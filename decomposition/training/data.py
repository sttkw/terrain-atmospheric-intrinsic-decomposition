"""TAID-Dataset loader for intrinsic decomposition training."""

from __future__ import annotations

import io
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.transforms import functional as TF

from pipeline import TARGET_PROMPTS
from training.args import build_hub_kwargs


PROMPT_COLUMNS = {
    "Albedo": "albedo_column",
    "Diffuse Shading": "diffuse_column",
    "Specular Shading": "specular_column",
    "Volume": "volume_column",
}


def _image_to_chw(value, name: str) -> torch.Tensor:
    if isinstance(value, PILImage.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(value, dict) and value.get("bytes") is not None:
        array = np.asarray(PILImage.open(io.BytesIO(value["bytes"])).convert("RGB"), dtype=np.float32) / 255.0
    else:
        raise TypeError(f"{name} must be a decoded Hugging Face Image, got {type(value)!r}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _npy_to_chw(value, name: str) -> torch.Tensor:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            array = np.load(value["path"], allow_pickle=False)
            value = None
        else:
            raise ValueError(f"{name} has neither bytes nor path")
    if value is not None:
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"{name} must contain NPY bytes, got {type(value)!r}")
        array = np.load(io.BytesIO(value), allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3-D, got {array.shape}")
    if array.shape[-1] in (1, 3):
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    if array.shape[0] != 3:
        raise ValueError(f"{name} must have three channels, got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def _resize_for_crop(tensor: torch.Tensor, resolution: int, interpolation: InterpolationMode) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    scale = resolution / min(height, width)
    size = [max(resolution, round(height * scale)), max(resolution, round(width * scale))]
    return TF.resize(tensor, size, interpolation=interpolation, antialias=interpolation != InterpolationMode.NEAREST)


def paired_augment_scene(
    original: torch.Tensor,
    targets: List[torch.Tensor],
    mask: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original = _resize_for_crop(original.float(), args.resolution, InterpolationMode.BILINEAR)
    targets = [TF.resize(x.float(), list(original.shape[-2:]), interpolation=InterpolationMode.BILINEAR, antialias=True) for x in targets]
    mask = _resize_for_crop(mask.float(), args.resolution, InterpolationMode.NEAREST)
    if args.center_crop:
        top = max(0, (original.shape[-2] - args.resolution) // 2)
        left = max(0, (original.shape[-1] - args.resolution) // 2)
        height = width = args.resolution
    else:
        top, left, height, width = RandomCrop.get_params(original, (args.resolution, args.resolution))
    original = TF.crop(original, top, left, height, width)
    targets = [TF.crop(x, top, left, height, width) for x in targets]
    mask = TF.crop(mask, top, left, height, width)
    if args.random_flip and torch.rand(()) < 0.5:
        original, targets, mask = TF.hflip(original), [TF.hflip(x) for x in targets], TF.hflip(mask)
    if args.random_vertical_flip and torch.rand(()) < 0.5:
        original, targets, mask = TF.vflip(original), [TF.vflip(x) for x in targets], TF.vflip(mask)
    if args.random_rotation_degrees > 0:
        angle = random.uniform(-args.random_rotation_degrees, args.random_rotation_degrees)
        original = TF.rotate(original, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        targets = [TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0) for x in targets]
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

    # D is distributed as linear HDR NPY but the published RGBX model uses
    # log1p(D)/log1p(5) as its VAE target space.
    targets[1] = torch.log1p(targets[1].clamp(0.0, 5.0)) / math.log1p(5.0)
    targets = [x.clamp(0.0, 1.0) for x in targets]
    return original.clamp(0.0, 1.0) * 2.0 - 1.0, mask.clamp(0.0, 1.0), torch.stack(targets) * 2.0 - 1.0


def build_scene_groups(dataset: Dataset, **_) -> List[Dict[str, int]]:
    return [{"row_index": index} for index in range(len(dataset))]


def split_scene_groups(scene_groups, seed: int, val_ratio: float):
    groups = list(scene_groups)
    random.Random(seed).shuffle(groups)
    count = max(1, round(len(groups) * val_ratio)) if len(groups) > 1 else 0
    return groups[count:], (groups[:count] or None)


class SceneGroupedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, scene_groups, args, **_):
        self.dataset = dataset
        self.scene_groups = scene_groups
        self.args = args

    def __len__(self):
        return len(self.scene_groups)

    def __getitem__(self, index):
        row = self.dataset[self.scene_groups[index]["row_index"]]
        targets = [
            _image_to_chw(row[self.args.albedo_column], self.args.albedo_column),
            _npy_to_chw(row[self.args.diffuse_column], self.args.diffuse_column),
            _image_to_chw(row[self.args.specular_column], self.args.specular_column),
            _image_to_chw(row[self.args.volume_column], self.args.volume_column),
        ]
        original = _image_to_chw(row[self.args.original_image_column], self.args.original_image_column)
        if self.args.water_mask_column and self.args.water_mask_column in row:
            mask = _image_to_chw(row[self.args.water_mask_column], self.args.water_mask_column)
        else:
            mask = torch.zeros_like(original)
        original, mask, targets = paired_augment_scene(original, targets, mask, self.args)
        return {
            "original_pixel_values": original.float(),
            "water_mask_pixel_values": mask.float(),
            "target_component_pixel_values": targets.float(),
        }


def collate_scene_batch(examples):
    return {
        key: torch.stack([example[key] for example in examples]).contiguous().float()
        for key in examples[0]
    }


def load_training_split(args):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.train_split,
        cache_dir=args.cache_dir,
        **build_hub_kwargs(args.hf_token),
    )
    if isinstance(dataset, DatasetDict):
        return dataset[args.train_split]
    return dataset


def validate_dataset_schema(dataset: Dataset, original_image_column: str, *_, **__):
    required = [original_image_column, "A", "D", "S", "V", "scene_id"]
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise ValueError(f"TAID-Dataset is missing columns: {missing}; available={dataset.column_names}")


def run_self_test():
    linear = np.linspace(0.0, 5.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    buffer = io.BytesIO()
    np.save(buffer, linear, allow_pickle=False)
    loaded = _npy_to_chw(buffer.getvalue(), "D")
    assert loaded.shape == (3, 4, 5)
    assert loaded.dtype == torch.float32
    print("TAID decomposition data self-test passed")
