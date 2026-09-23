#!/usr/bin/env python3
"""Decompose a terrain photo and edit its atmosphere with the released weights.

    python demo.py --input_image demo_image/2.jpg --water_mask demo_image/2.png \
        --p_control 0 -3 0 --output_dir outputs/demo/2

Weights are downloaded from the Hugging Face model repo on first use.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "decomposition"))
sys.path.insert(0, str(ROOT / "atmosphere"))

from infer import load_model  # noqa: E402  (atmosphere/infer.py)
from inference import (  # noqa: E402  (decomposition/inference.py)
    build_pipeline,
    create_generator,
    resolve_dtype,
    resolve_resolution,
    save_linear_image_pair,
)
from model import apply_delta_spatial_filter  # noqa: E402
from pipeline import (  # noqa: E402
    compose_intrinsic_hdr,
    generate_corrected_intrinsic_tensors,
    load_image,
    load_mask_image,
    output_to_linear_components,
)

DEFAULT_MODEL_REPO = "ShunTatsukawa/TAID-Models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intrinsic decomposition + atmospheric editing demo.")
    parser.add_argument("--input_image", required=True, help="sRGB terrain photo.")
    parser.add_argument(
        "--water_mask",
        default=None,
        help="Optional one-hot RGB mask (R=water, G=terrain, B=sky). Omit to run without mask conditioning.",
    )
    parser.add_argument("--output_dir", default="outputs/demo")
    parser.add_argument(
        "--p_control",
        type=float,
        nargs=3,
        default=(0.0, -3.0, 0.0),
        metavar=("AIR", "AEROSOL", "OZONE"),
        help="Log-scale change of the atmospheric parameters. 0 keeps a parameter, <0 thins, >0 thickens.",
    )
    parser.add_argument("--model_repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--decomposition_model", default=None, help="Local checkpoint dir instead of the HF repo.")
    parser.add_argument("--atmosphere_checkpoint", default=None, help="Local .pt instead of the HF repo.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--correction_strength", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    return parser.parse_args()


def decompose(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    model_path = args.decomposition_model or os.path.join(
        snapshot_download(args.model_repo, allow_patterns=["decomposition/*"]), "decomposition"
    )
    weight_dtype = resolve_dtype(args.dtype, device)
    pipeline = build_pipeline(model_path, weight_dtype, device)

    image = load_image(image_path=args.input_image)
    if args.water_mask:
        water_mask = load_mask_image(image_path=args.water_mask).resize(image.size, resample=Image.NEAREST)
    else:
        water_mask = Image.new("RGB", image.size, color=(0, 0, 0))

    # Guidance settings of the paper's reconstruction-corrected ("corr") inference.
    outputs, _ = generate_corrected_intrinsic_tensors(
        image=image,
        water_mask=water_mask,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        vae=pipeline.vae,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        resolution=resolve_resolution(model_path, None),
        num_inference_steps=args.num_inference_steps,
        device=device,
        weight_dtype=weight_dtype,
        generator=create_generator(device, args.seed),
        input_is_srgb=True,
        correction_strength=args.correction_strength,
        correction_iters=1,
        correction_latent_blend=1.0,
        correction_start_step=max(0, args.num_inference_steps - 15),
        correction_end_step=None,
        correction_every=1,
        correction_schedule="cosine",
        residual_chroma=1.0,
        correction_max_delta=0.015,
        projection_mode="uniform",
        guidance_lowpass_downsample=8,
        guidance_grad_clip_rms=0.02,
        guidance_prior_weight=0.05,
        guidance_decode_batch_size=1,
        final_projection=True,
        final_projection_strength=0.0,
        component_mobility=(1.0, 1.0, 1.0, 1.0),
        shared_initial_noise=True,
        unet_batch_size=4,
        show_progress=True,
        progress_desc="Decomposition",
    )
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    linear = output_to_linear_components(outputs)
    return {
        "albedo": linear["Albedo"],
        "diffuse_shading": linear["Diffuse Shading"],
        "specular_shading": linear["Specular Shading"],
        "volume": linear["Volume"],
    }


@torch.no_grad()
def edit_atmosphere(
    args: argparse.Namespace, maps: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    checkpoint = args.atmosphere_checkpoint or hf_hub_download(args.model_repo, "atmosphere/terrain_difference.pt")
    model, ckpt = load_model(Path(checkpoint), device)
    model_args = ckpt.get("args", {})
    image_size = int(model_args.get("image_size", 512))
    clip = float(model_args.get("delta_log_p_clip", 3.0))
    clamp_delta = float(ckpt.get("clamp_delta", model_args.get("clamp_delta", 3.0)))

    # The editor runs at its training size; the predicted log change is then
    # resized back and applied to the full-resolution maps.
    x_full = torch.cat(
        [maps["diffuse_shading"], maps["specular_shading"], maps["volume"]], dim=0
    ).unsqueeze(0).clamp_min(0.0).to(device)
    x_cur = F.interpolate(x_full, size=(image_size, image_size), mode="bilinear", align_corners=False)
    delta_log_p = torch.tensor(args.p_control, dtype=torch.float32, device=device).clamp(-clip, clip)
    x_in = torch.cat([x_cur, delta_log_p.view(1, 3, 1, 1).expand(1, 3, image_size, image_size)], dim=1)

    delta = model(x_in).float().clamp(-clamp_delta, clamp_delta)
    delta = apply_delta_spatial_filter(
        delta,
        str(model_args.get("delta_spatial_mode", "global")),
        int(model_args.get("delta_blur_kernel", 31)),
        float(model_args.get("delta_blur_sigma", 0.0)),
    )
    delta = F.interpolate(delta, size=x_full.shape[-2:], mode="bilinear", align_corners=False)
    x_edit = (x_full * torch.exp(delta)).squeeze(0).cpu()
    return {
        "diffuse_shading": x_edit[0:3],
        "specular_shading": x_edit[3:6],
        "volume": x_edit[6:9],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)

    maps = decompose(args, device)
    for name, tensor in maps.items():
        save_linear_image_pair(tensor, name, str(out_dir / "decomposition"))
    save_linear_image_pair(compose_intrinsic_hdr(**maps), "reconstruction", str(out_dir / "decomposition"))

    edited = edit_atmosphere(args, maps, device)
    for name, tensor in edited.items():
        save_linear_image_pair(tensor, name, str(out_dir / "atmosphere"))
    save_linear_image_pair(
        compose_intrinsic_hdr(albedo=maps["albedo"], **edited), "edited", str(out_dir / "atmosphere")
    )


if __name__ == "__main__":
    main()
