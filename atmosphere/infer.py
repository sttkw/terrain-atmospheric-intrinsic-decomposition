#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
try:
    import cv2
except ImportError:  # pragma: no cover - fallback for non-OpenCV environments
    cv2 = None
    import imageio.v3 as iio

from model import DifferenceEditorUNet, apply_delta_spatial_filter


def read_exr_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read EXR: {path}")
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected RGB EXR at {path}, got shape {image.shape}")
        return np.ascontiguousarray(image[:, :, :3][:, :, ::-1].astype(np.float32, copy=True))

    image = iio.imread(path).astype(np.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB EXR at {path}, got shape {image.shape}")
    return np.ascontiguousarray(image[:, :, :3])


def load_map_tensor(path: Path, image_size: int, *, clamp_albedo: bool = False) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected tensor in {path}, got {type(tensor)!r}")
        tensor = tensor.detach().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim != 3:
            raise ValueError(f"Unsupported tensor shape in {path}: {tuple(tensor.shape)}")
    elif suffix == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3-D NPY array at {path}, got {array.shape}")
        if array.shape[-1] in (1, 3):
            array = np.moveaxis(array, -1, 0)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    elif suffix == ".exr":
        tensor = torch.from_numpy(read_exr_rgb(path)).permute(2, 0, 1).contiguous().float()
    else:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW 3ch tensor in {path}, got {tuple(tensor.shape)}")

    tensor = tensor.clamp(0.0, 1.0) if clamp_albedo else tensor.clamp_min(0.0)
    if image_size > 0 and tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.contiguous()


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu().clamp_min(0.0)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        torch.save(tensor.half(), path)
        return

    if suffix == ".npy":
        np.save(path, tensor.permute(1, 2, 0).numpy().astype(np.float32), allow_pickle=False)
        return

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Only 3ch tensors can be saved as {suffix}; use .pt for shape {tuple(tensor.shape)}.")

    if suffix == ".exr":
        array = tensor.permute(1, 2, 0).numpy().astype(np.float32)
        if cv2 is not None:
            params = []
            if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_HALF"):
                params = [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_HALF)]
            ok = cv2.imwrite(str(path), array[:, :, ::-1], params)
            if not ok:
                raise RuntimeError(f"Failed to write EXR: {path}")
        else:
            iio.imwrite(path, array)
        return

    display = tensor.clamp(0.0, 1.0).pow(1.0 / 2.2)
    array_u8 = np.round(display.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array_u8, mode="RGB").save(path)


def save_raw_pt_tensor(tensor: torch.Tensor, path: Path) -> None:
    if path.suffix.lower() != ".pt":
        raise ValueError("--out_delta must use a .pt path because delta-log values can be negative.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().float().cpu().half(), path)


def split_components(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x[:, 0:3], x[:, 3:6], x[:, 6:9]


def build_delta_log_p(args: argparse.Namespace, eps_p: float, device: torch.device) -> torch.Tensor:
    if args.p_control is not None:
        return torch.tensor(args.p_control, dtype=torch.float32, device=device)
    if args.delta_log_p is not None:
        return torch.tensor(args.delta_log_p, dtype=torch.float32, device=device)
    if args.p_cur is None or args.p_tgt is None:
        raise ValueError("Specify --p_control, --delta_log_p, or both --p_cur and --p_tgt.")

    p_cur = torch.tensor(args.p_cur, dtype=torch.float32, device=device)
    p_tgt = torch.tensor(args.p_tgt, dtype=torch.float32, device=device)
    if torch.any(p_cur + eps_p <= 0.0) or torch.any(p_tgt + eps_p <= 0.0):
        raise ValueError("--p_cur and --p_tgt must be positive after adding eps_p.")
    return torch.log(p_tgt + eps_p) - torch.log(p_cur + eps_p)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DifferenceEditorUNet, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")

    state = ckpt.get("editor_state", ckpt.get("model_state"))
    if state is None:
        raise KeyError("Checkpoint must contain 'editor_state' or 'model_state'.")

    model_args = ckpt.get("args", {})
    base_ch = int(model_args.get("base_ch", 32))
    in_ch = int(state["enc1.block.0.weight"].shape[1])
    out_ch = int(state["out_conv.weight"].shape[0])
    if in_ch != 12 or out_ch != 9:
        raise ValueError(f"Expected DifferenceEditorUNet 12ch->9ch, got {in_ch}ch->{out_ch}ch.")

    model = DifferenceEditorUNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run atmos difference editor inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--diff_shading", type=str, required=True)
    parser.add_argument("--spec_shading", type=str, required=True)
    parser.add_argument("--volume", type=str, required=True)
    parser.add_argument("--albedo", type=str, default=None, help="Optional albedo for reconstruction output.")

    parser.add_argument("--p_cur", type=float, nargs=3, metavar=("AIR", "AEROSOL", "OZONE"), default=None)
    parser.add_argument("--p_tgt", type=float, nargs=3, metavar=("AIR", "AEROSOL", "OZONE"), default=None)
    parser.add_argument("--delta_log_p", type=float, nargs=3, metavar=("D_AIR", "D_AEROSOL", "D_OZONE"), default=None)
    parser.add_argument(
        "--p_control",
        type=float,
        nargs=3,
        metavar=("AIR_CTRL", "AEROSOL_CTRL", "OZONE_CTRL"),
        default=None,
        help="Unbounded relative controls around 0. Values are used directly as delta_log_p.",
    )

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eps_p", type=float, default=None)
    parser.add_argument("--delta_log_p_clip", type=float, default=None)
    parser.add_argument("--clamp_delta", type=float, default=None)
    parser.add_argument("--delta_spatial_mode", type=str, default=None, choices=["none", "blur", "global"])
    parser.add_argument("--delta_blur_kernel", type=int, default=None)
    parser.add_argument("--delta_blur_sigma", type=float, default=None)

    parser.add_argument("--out_diff_shading", type=str, default="atmos/outputs/edited_diffuse_shading.pt")
    parser.add_argument("--out_spec_shading", type=str, default="atmos/outputs/edited_specular_shading.pt")
    parser.add_argument("--out_volume", type=str, default="atmos/outputs/edited_volume.pt")
    parser.add_argument("--out_reconstruction", type=str, default=None)
    parser.add_argument("--out_delta", type=str, default=None, help="Optional path to save the 9ch predicted delta-log tensor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    model, ckpt = load_model(checkpoint_path, device)

    model_args = ckpt.get("args", {})
    image_size = int(args.image_size if args.image_size is not None else model_args.get("image_size", 512))
    eps_p = float(args.eps_p if args.eps_p is not None else ckpt.get("eps_p", model_args.get("eps_p", 1e-6)))
    delta_log_p_clip = float(
        args.delta_log_p_clip
        if args.delta_log_p_clip is not None
        else model_args.get("delta_log_p_clip", 3.0)
    )
    clamp_delta = float(
        args.clamp_delta if args.clamp_delta is not None else ckpt.get("clamp_delta", model_args.get("clamp_delta", 3.0))
    )
    delta_spatial_mode = str(
        args.delta_spatial_mode
        if args.delta_spatial_mode is not None
        else model_args.get("delta_spatial_mode", "global")
    )
    delta_blur_kernel = int(
        args.delta_blur_kernel
        if args.delta_blur_kernel is not None
        else model_args.get("delta_blur_kernel", 31)
    )
    delta_blur_sigma = float(
        args.delta_blur_sigma
        if args.delta_blur_sigma is not None
        else model_args.get("delta_blur_sigma", 0.0)
    )
    if eps_p <= 0.0:
        raise ValueError("--eps_p must be positive.")
    if delta_log_p_clip < 0.0:
        raise ValueError("--delta_log_p_clip must be >= 0.")
    if clamp_delta <= 0.0:
        raise ValueError("--clamp_delta must be positive.")
    if delta_spatial_mode not in {"none", "blur", "global"}:
        raise ValueError("--delta_spatial_mode must be one of: none, blur, global.")
    if delta_blur_kernel < 1 or delta_blur_kernel % 2 == 0:
        raise ValueError("--delta_blur_kernel must be a positive odd integer.")
    if delta_blur_sigma < 0.0:
        raise ValueError("--delta_blur_sigma must be >= 0.")

    diff = load_map_tensor(Path(args.diff_shading), image_size)
    spec = load_map_tensor(Path(args.spec_shading), image_size)
    volume = load_map_tensor(Path(args.volume), image_size)
    x_cur = torch.cat([diff, spec, volume], dim=0).unsqueeze(0).to(device)

    delta_log_p = build_delta_log_p(args, eps_p, device)
    if delta_log_p_clip > 0.0:
        delta_log_p = delta_log_p.clamp(-delta_log_p_clip, delta_log_p_clip)
    _, _, h, w = x_cur.shape
    delta_log_p_map = delta_log_p.view(1, 3, 1, 1).expand(1, 3, h, w)
    x_in = torch.cat([x_cur, delta_log_p_map.to(dtype=x_cur.dtype)], dim=1)

    with torch.no_grad():
        pred_delta = model(x_in).float()
        pred_delta_clip = pred_delta.clamp(-clamp_delta, clamp_delta)
        pred_delta_low = apply_delta_spatial_filter(
            pred_delta_clip,
            delta_spatial_mode,
            delta_blur_kernel,
            delta_blur_sigma,
        )
        x_edit = x_cur.float() * torch.exp(pred_delta_low)

    out_diff, out_spec, out_volume = split_components(x_edit.cpu())
    save_tensor(out_diff.squeeze(0), Path(args.out_diff_shading))
    save_tensor(out_spec.squeeze(0), Path(args.out_spec_shading))
    save_tensor(out_volume.squeeze(0), Path(args.out_volume))
    if args.out_delta is not None:
        save_raw_pt_tensor(pred_delta_low.squeeze(0), Path(args.out_delta))

    if args.albedo is not None and args.out_reconstruction is not None:
        albedo = load_map_tensor(Path(args.albedo), image_size, clamp_albedo=True).to(device)
        reconstruction = albedo.float() * out_diff.squeeze(0).to(device).float()
        reconstruction = reconstruction + out_spec.squeeze(0).to(device).float() + out_volume.squeeze(0).to(device).float()
        save_tensor(reconstruction.cpu(), Path(args.out_reconstruction))

    print(f"Saved edited diff shading to: {args.out_diff_shading}")
    print(f"Saved edited spec shading to: {args.out_spec_shading}")
    print(f"Saved edited volume to: {args.out_volume}")
    if args.out_delta is not None:
        print(f"Saved predicted delta-log tensor to: {args.out_delta}")
    if args.albedo is not None and args.out_reconstruction is not None:
        print(f"Saved reconstruction to: {args.out_reconstruction}")
    print("delta_log_p:", " ".join(f"{v:.6f}" for v in delta_log_p.detach().cpu().tolist()))
    print(
        f"delta spatial filter: mode={delta_spatial_mode}, "
        f"kernel={delta_blur_kernel}, sigma={delta_blur_sigma:.6f}"
    )


if __name__ == "__main__":
    main()
