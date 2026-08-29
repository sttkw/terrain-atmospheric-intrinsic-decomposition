#!/usr/bin/env python3
import argparse
import csv
import io
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import DifferenceEditorUNet, apply_delta_spatial_filter

load_dotenv()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset_by_seed(hf_ds, seed_column: str, val_ratio: float, seed: int):
    if seed_column not in hf_ds.column_names:
        raise ValueError(f"Dataset must contain '{seed_column}' column for seed-wise split.")

    seed_values = list(hf_ds[seed_column])
    unique_seeds = sorted(set(int(s) for s in seed_values))
    if len(unique_seeds) < 2:
        raise ValueError("Need at least 2 unique seeds to perform train/val split.")

    rng = random.Random(seed)
    rng.shuffle(unique_seeds)

    n_val = max(1, int(len(unique_seeds) * val_ratio))
    n_val = min(n_val, len(unique_seeds) - 1)

    val_seeds = sorted(unique_seeds[:n_val])
    val_seed_set = set(val_seeds)

    train_indices = [i for i, s in enumerate(seed_values) if int(s) not in val_seed_set]
    val_indices = [i for i, s in enumerate(seed_values) if int(s) in val_seed_set]

    return hf_ds.select(train_indices), hf_ds.select(val_indices), val_seeds


class DifferencePairDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        image_size: int,
        seed_column: str,
        diff_column: str,
        spec_column: str,
        volume_column: str,
        density_column: str,
        aerosol_column: str,
        ozone_column: str,
        fixed_target_pairs: bool,
    ):
        self.ds = hf_ds
        self.image_size = image_size
        self.seed_column = seed_column
        self.diff_column = diff_column
        self.spec_column = spec_column
        self.volume_column = volume_column
        self.density_column = density_column
        self.aerosol_column = aerosol_column
        self.ozone_column = ozone_column
        self.fixed_target_pairs = fixed_target_pairs

        required = {
            seed_column,
            diff_column,
            spec_column,
            volume_column,
            density_column,
            aerosol_column,
            ozone_column,
        }
        missing = [name for name in required if name not in self.ds.column_names]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.seed_to_indices: Dict[int, List[int]] = {}
        for i, seed_value in enumerate(self.ds[self.seed_column]):
            self.seed_to_indices.setdefault(int(seed_value), []).append(i)

    @staticmethod
    def _load_component(value, column_name: str) -> torch.Tensor:
        array = None
        if isinstance(value, Image.Image):
            array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
        if isinstance(value, dict):
            value_bytes = value.get("bytes")
            value_path = value.get("path")
            if value_bytes is not None:
                array = np.load(io.BytesIO(value_bytes), allow_pickle=False)
            elif value_path:
                array = np.load(value_path, allow_pickle=False)
            else:
                raise ValueError(f"{column_name} entry must contain bytes or path")
        elif isinstance(value, (bytes, bytearray)):
            array = np.load(io.BytesIO(value), allow_pickle=False)
        elif isinstance(value, str):
            array = np.load(value, allow_pickle=False)
        if array is None:
            raise TypeError(f"Unsupported {column_name} value type: {type(value)!r}")
        array = np.asarray(array, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim != 3:
            raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)}")

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 channels after load, got {tuple(tensor.shape)}")
        return tensor.contiguous()

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_size <= 0 or tensor.shape[-2:] == (self.image_size, self.image_size):
            return tensor
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _load_components(self, row) -> torch.Tensor:
        diff = self._load_component(row[self.diff_column], self.diff_column).clamp_min(0.0)
        spec = self._load_component(row[self.spec_column], self.spec_column).clamp_min(0.0)
        volume = self._load_component(row[self.volume_column], self.volume_column).clamp_min(0.0)
        return self._resize(torch.cat([diff, spec, volume], dim=0))

    def _load_params(self, row) -> torch.Tensor:
        return torch.tensor(
            [
                float(row[self.density_column]),
                float(row[self.aerosol_column]),
                float(row[self.ozone_column]),
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row_i = self.ds[idx]
        seed_value = int(row_i[self.seed_column])
        candidates = self.seed_to_indices[seed_value]

        if len(candidates) == 1:
            tgt_idx = idx
        elif self.fixed_target_pairs:
            pos = candidates.index(idx)
            tgt_idx = candidates[(pos + 1) % len(candidates)]
        else:
            tgt_idx = idx
            while tgt_idx == idx:
                tgt_idx = random.choice(candidates)

        row_j = self.ds[tgt_idx]
        return {
            "x_i": self._load_components(row_i),
            "x_j": self._load_components(row_j),
            "p_i": self._load_params(row_i),
            "p_j": self._load_params(row_j),
        }


def init_csv_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_delta",
                "train_x",
                "train_mae_diff",
                "train_mae_spec",
                "train_mae_volume",
                "val_total",
                "val_delta",
                "val_x",
                "val_mae_diff",
                "val_mae_spec",
                "val_mae_volume",
                "is_best",
            ]
        )


def append_csv_log(
    path: Path,
    epoch: int,
    train_m: Dict[str, float],
    val_m: Dict[str, float],
    is_best: int,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{train_m['loss_total']:.8f}",
                f"{train_m['loss_delta']:.8f}",
                f"{train_m['loss_x']:.8f}",
                f"{train_m['mae_diff']:.8f}",
                f"{train_m['mae_spec']:.8f}",
                f"{train_m['mae_volume']:.8f}",
                f"{val_m['loss_total']:.8f}",
                f"{val_m['loss_delta']:.8f}",
                f"{val_m['loss_x']:.8f}",
                f"{val_m['mae_diff']:.8f}",
                f"{val_m['mae_spec']:.8f}",
                f"{val_m['mae_volume']:.8f}",
                is_best,
            ]
        )


def save_val_seeds_csv(path: Path, val_seeds: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["val_seed"])
        for seed in val_seeds:
            writer.writerow([seed])


def create_versioned_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    latest = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            suffix = child.name[len(prefix) :]
            if suffix.isdigit():
                latest = max(latest, int(suffix))
    run_dir = root / f"{prefix}{latest + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    scheduler=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "editor_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "args": {key: value for key, value in vars(args).items() if key != "hf_token"},
        "eps_p": float(args.eps_p),
        "eps_x": float(args.eps_x),
        "clamp_delta": float(args.clamp_delta),
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def split_components(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x[:, 0:3], x[:, 3:6], x[:, 6:9]


def _zero_metrics() -> Dict[str, float]:
    return {
        "loss_total": 0.0,
        "loss_delta": 0.0,
        "loss_x": 0.0,
        "mae_diff": 0.0,
        "mae_spec": 0.0,
        "mae_volume": 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    train: bool,
    eps_p: float,
    eps_x: float,
    delta_log_p_clip: float,
    clamp_delta: float,
    delta_spatial_mode: str,
    delta_blur_kernel: int,
    delta_blur_sigma: float,
    lambda_delta: float,
    lambda_x: float,
    amp_dtype: torch.dtype | None,
) -> Dict[str, float]:
    model.train(mode=train)
    totals = _zero_metrics()
    total_count = 0

    for batch in loader:
        x_i = batch["x_i"].to(device)
        x_j = batch["x_j"].to(device)
        p_i = batch["p_i"].to(device)
        p_j = batch["p_j"].to(device)
        bs, _, h, w = x_i.shape

        delta_log_p = torch.log(p_j.float() + eps_p) - torch.log(p_i.float() + eps_p)
        if delta_log_p_clip > 0.0:
            delta_log_p = delta_log_p.clamp(-delta_log_p_clip, delta_log_p_clip)
        delta_log_p_map = delta_log_p[:, :, None, None].expand(-1, -1, h, w)
        x_in = torch.cat([x_i, delta_log_p_map.to(dtype=x_i.dtype)], dim=1)

        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=(amp_dtype is not None and device.type == "cuda"),
            ):
                pred_delta = model(x_in)

            pred_delta_clip = pred_delta.float().clamp(-clamp_delta, clamp_delta)
            pred_delta_low = apply_delta_spatial_filter(
                pred_delta_clip,
                delta_spatial_mode,
                delta_blur_kernel,
                delta_blur_sigma,
            )
            x_edit = x_i.float() * torch.exp(pred_delta_low)

            gt_delta = torch.log(x_j.float() + eps_x) - torch.log(x_i.float() + eps_x)
            gt_delta_clip = gt_delta.clamp(-clamp_delta, clamp_delta)
            gt_delta_low = apply_delta_spatial_filter(
                gt_delta_clip,
                delta_spatial_mode,
                delta_blur_kernel,
                delta_blur_sigma,
            )

            loss_delta = F.l1_loss(pred_delta_low, gt_delta_low)
            loss_x = F.l1_loss(x_edit.float(), x_j.float())
            loss_total = lambda_delta * loss_delta + lambda_x * loss_x

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss_total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_total.backward()
                    optimizer.step()

        edit_diff, edit_spec, edit_volume = split_components(x_edit.float())
        tgt_diff, tgt_spec, tgt_volume = split_components(x_j.float())
        batch_metrics = {
            "loss_total": loss_total.item(),
            "loss_delta": loss_delta.item(),
            "loss_x": loss_x.item(),
            "mae_diff": F.l1_loss(edit_diff, tgt_diff).item(),
            "mae_spec": F.l1_loss(edit_spec, tgt_spec).item(),
            "mae_volume": F.l1_loss(edit_volume, tgt_volume).item(),
        }

        for key, value in batch_metrics.items():
            totals[key] += value * bs
        total_count += bs

    denom = max(1, total_count)
    return {key: value / denom for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train atmos difference editor: delta log P -> delta log components.")
    parser.add_argument("--dataset", type=str, default="ShunTatsukawa/TAID-AtmosEdit")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--seed_column", type=str, default="seed")
    parser.add_argument("--diff_column", type=str, default="D")
    parser.add_argument("--spec_column", type=str, default="S")
    parser.add_argument("--volume_column", type=str, default="V")
    parser.add_argument("--density_column", type=str, default="s_density")
    parser.add_argument("--aerosol_column", type=str, default="s_aerosol")
    parser.add_argument("--ozone_column", type=str, default="s_ozone")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--lr_min_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eps_p", type=float, default=1e-6)
    parser.add_argument("--eps_x", type=float, default=1e-4)
    parser.add_argument("--delta_log_p_clip", type=float, default=3.0)
    parser.add_argument("--clamp_delta", type=float, default=3.0)
    parser.add_argument("--delta_spatial_mode", type=str, default="global", choices=["none", "blur", "global"])
    parser.add_argument("--delta_blur_kernel", type=int, default=31)
    parser.add_argument("--delta_blur_sigma", type=float, default=0.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--lambda_x", type=float, default=1.0)

    parser.add_argument("--amp", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpoint_root", type=str, default="outputs/atmosphere/checkpoints")
    parser.add_argument("--version_prefix", type=str, default="diff_ver")
    parser.add_argument("--save_name", type=str, default="terrain_difference.pt")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_csv", type=str, default="outputs/atmosphere/train_log.csv")
    parser.add_argument("--tb_root", type=str, default="outputs/atmosphere/tensorboard")
    parser.add_argument("--tb_name", type=str, default="terrain_difference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")
    if args.eps_p <= 0.0 or args.eps_x <= 0.0:
        raise ValueError("--eps_p and --eps_x must be positive.")
    if args.delta_log_p_clip < 0.0:
        raise ValueError("--delta_log_p_clip must be >= 0.")
    if args.clamp_delta <= 0.0:
        raise ValueError("--clamp_delta must be positive.")
    if args.delta_blur_kernel < 1 or args.delta_blur_kernel % 2 == 0:
        raise ValueError("--delta_blur_kernel must be a positive odd integer.")
    if args.delta_blur_sigma < 0.0:
        raise ValueError("--delta_blur_sigma must be >= 0.")
    if args.lr_min_ratio < 0.0:
        raise ValueError("--lr_min_ratio must be >= 0.")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    scaler = None

    if args.amp == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif args.amp == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    hf_ds_all = load_dataset(args.dataset, split=args.split, token=args.hf_token)

    train_hf, val_hf, val_seeds = split_dataset_by_seed(
        hf_ds_all,
        args.seed_column,
        args.val_ratio,
        args.seed,
    )

    print(f"Seed-wise split: train_samples={len(train_hf)}, val_samples={len(val_hf)}")
    print(f"Val seed count: {len(val_seeds)}")
    print(f"Val seeds: {val_seeds}")

    train_ds = DifferencePairDataset(
        train_hf,
        image_size=args.image_size,
        seed_column=args.seed_column,
        diff_column=args.diff_column,
        spec_column=args.spec_column,
        volume_column=args.volume_column,
        density_column=args.density_column,
        aerosol_column=args.aerosol_column,
        ozone_column=args.ozone_column,
        fixed_target_pairs=False,
    )

    val_ds = DifferencePairDataset(
        val_hf,
        image_size=args.image_size,
        seed_column=args.seed_column,
        diff_column=args.diff_column,
        spec_column=args.spec_column,
        volume_column=args.volume_column,
        density_column=args.density_column,
        aerosol_column=args.aerosol_column,
        ozone_column=args.ozone_column,
        fixed_target_pairs=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DifferenceEditorUNet(in_ch=12, out_ch=9, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.lr * args.lr_min_ratio,
        )

    run_dir = create_versioned_dir(Path(args.checkpoint_root), args.version_prefix)
    save_path = run_dir / args.save_name

    val_seed_csv_path = run_dir / "val_seeds.csv"
    save_val_seeds_csv(val_seed_csv_path, val_seeds)

    log_csv_path = Path(args.log_csv)
    tb_run_dir = Path(args.tb_root) / f"{args.tb_name}_{run_dir.name}"

    writer = SummaryWriter(log_dir=str(tb_run_dir))
    init_csv_log(log_csv_path)

    print(f"Checkpoint run dir: {run_dir}")
    print(f"Validation seed CSV path: {val_seed_csv_path}")
    print(f"CSV log path: {log_csv_path}")
    print(f"TensorBoard log dir: {tb_run_dir}")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train=True,
            eps_p=args.eps_p,
            eps_x=args.eps_x,
            delta_log_p_clip=args.delta_log_p_clip,
            clamp_delta=args.clamp_delta,
            delta_spatial_mode=args.delta_spatial_mode,
            delta_blur_kernel=args.delta_blur_kernel,
            delta_blur_sigma=args.delta_blur_sigma,
            lambda_delta=args.lambda_delta,
            lambda_x=args.lambda_x,
            amp_dtype=amp_dtype,
        )

        val_m = run_epoch(
            model,
            val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            train=False,
            eps_p=args.eps_p,
            eps_x=args.eps_x,
            delta_log_p_clip=args.delta_log_p_clip,
            clamp_delta=args.clamp_delta,
            delta_spatial_mode=args.delta_spatial_mode,
            delta_blur_kernel=args.delta_blur_kernel,
            delta_blur_sigma=args.delta_blur_sigma,
            lambda_delta=args.lambda_delta,
            lambda_x=args.lambda_x,
            amp_dtype=amp_dtype,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_total={train_m['loss_total']:.6f} "
            f"train_delta={train_m['loss_delta']:.6f} "
            f"train_x={train_m['loss_x']:.6f} "
            f"val_total={val_m['loss_total']:.6f} "
            f"val_delta={val_m['loss_delta']:.6f} "
            f"val_x={val_m['loss_x']:.6f} "
            f"lr={current_lr:.8g}"
        )

        for key, value in train_m.items():
            writer.add_scalar(f"{key}/train", value, epoch)
        for key, value in val_m.items():
            writer.add_scalar(f"{key}/val", value, epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)

        if scheduler is not None:
            scheduler.step()

        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = save_path.with_name(
                f"{save_path.stem}_epoch{epoch:03d}{save_path.suffix}"
            )
            save_checkpoint(periodic_path, model, optimizer, args, epoch, scheduler=scheduler)
            print(f"  -> periodic checkpoint saved to {periodic_path}")

        is_best = 0
        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            save_checkpoint(save_path, model, optimizer, args, epoch, scheduler=scheduler)
            print(f"  -> best checkpoint saved to {save_path}")
            is_best = 1

        append_csv_log(log_csv_path, epoch, train_m, val_m, is_best)

    writer.close()

    print("Difference training completed")
    print(f"Best val_total: {best_val:.6f}")


if __name__ == "__main__":
    main()
