import argparse
import json
import logging
import os
from typing import Dict, Optional

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

from pipeline import LOG_FORWARD_S_MAX, MASK_CONDITIONING_CHANNELS, PREDICTION_TYPE, TARGET_PROMPTS

logger = logging.getLogger(__name__)


def _save_exr_tensor(output_path: str, tensor_chw: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    array_hwc = tensor_chw.detach().float().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    iio.imwrite(output_path, array_hwc)


def _linear_to_display_png_tensor(tensor_chw_linear: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    clamped = tensor_chw_linear.detach().float().cpu().clamp(0.0, 1.0)
    inv_gamma = 1.0 / gamma
    return clamped.pow(inv_gamma)


def _linear_to_png_tensor(tensor_chw_linear: torch.Tensor) -> torch.Tensor:
    return tensor_chw_linear.detach().float().cpu().clamp(0.0, 1.0)


def build_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def maybe_enable_xformers(unet: UNet2DConditionModel) -> None:
    if not is_xformers_available():
        raise ValueError("xformers is not available. Install it before using this flag.")

    import xformers

    xformers_version = version.parse(xformers.__version__)
    if xformers_version == version.parse("0.0.16"):
        logger.warning(
            "xFormers 0.0.16 is known to be unstable for training on some GPUs."
        )
    unet.enable_xformers_memory_efficient_attention()


def expand_unet_conv_in_to_ten_channels(unet: UNet2DConditionModel) -> None:
    in_channels = int(unet.config.in_channels)
    target_in_channels = 8 + MASK_CONDITIONING_CHANNELS
    if in_channels == target_in_channels:
        return
    if in_channels != 8:
        raise ValueError(
            f"Expected an InstructPix2Pix U-Net with 8 or {target_in_channels} input channels, "
            f"but found {in_channels}."
        )

    old_conv = unet.conv_in
    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError(f"Expected unet.conv_in to be nn.Conv2d, got {type(old_conv)!r}")

    new_conv = nn.Conv2d(
        in_channels=target_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )

    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :8, :, :].copy_(old_conv.weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    unet.conv_in = new_conv
    unet.register_to_config(in_channels=target_in_channels)


def save_training_metadata(
    output_dir: str,
    args: argparse.Namespace,
    num_train_scenes: int,
    num_val_scenes: int,
) -> None:
    metadata = {
        "task": "intrinsic_terrain_decomposition",
        "target_prompts": list(TARGET_PROMPTS),
        "classifier_free_guidance_used": False,
        "prediction_type": PREDICTION_TYPE,
        "dataset_columns": {
            "Input": args.original_image_column,
            "A": args.albedo_column,
            "D": args.diffuse_column,
            "S": args.specular_column,
            "V": args.volume_column,
            "water_mask": args.water_mask_column,
            "scene_id": args.scene_id_column,
        },
        "resolution": args.resolution,
        "num_train_examples": num_train_scenes,
        "num_val_examples": num_val_scenes,
        "validation_split_ratio": 0.1,
        "loss_weights": {
            "diffusion": args.loss_weight_diffusion,
            "recon": args.loss_weight_recon,
            "direct": args.loss_weight_direct,
            "water": args.loss_weight_water,
        },
        "mask_conditioning": {
            "channels": ["water", "sky"],
            "num_channels": MASK_CONDITIONING_CHANNELS,
        },
        "scene_grouped_training": False,
        "public_dataset": args.dataset_name,
        "mask_policy": "use water_mask; zero conditioning only when the column is disabled",
        "strict_direct_loss_four_components": True,
        "target_encoding": "mixed_linear_unit_and_log_forward_s01",
        "component_target_encoding": {
            "Albedo": "linear_unit",
            "Diffuse Shading": "log_forward_s01",
            "Specular Shading": "linear_unit",
            "Volume": "linear_unit",
        },
        "log_forward_s_max": dict(LOG_FORWARD_S_MAX),
    }
    metadata_path = os.path.join(output_dir, "terrain_decomposition_config.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def resolve_resume_path(output_dir: str, resume_from_checkpoint: str) -> Optional[str]:
    if resume_from_checkpoint != "latest":
        return os.path.basename(resume_from_checkpoint)

    if not os.path.isdir(output_dir):
        return None

    checkpoint_dirs = [
        directory
        for directory in os.listdir(output_dir)
        if directory.startswith("checkpoint-") and directory.split("-")[-1].isdigit()
    ]
    checkpoint_dirs.sort(key=lambda item: int(item.split("-")[-1]))
    return checkpoint_dirs[-1] if checkpoint_dirs else None
