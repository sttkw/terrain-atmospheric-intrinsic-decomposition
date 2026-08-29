import argparse
import json
import logging
import os
from typing import Dict, Optional

import torch

from pipeline import TARGET_PROMPTS, canonicalize_prompt

logger = logging.getLogger(__name__)


def resolve_hf_token() -> Optional[str]:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token
    return None


def build_hub_kwargs(token: Optional[str]) -> Dict[str, str]:
    if token:
        return {"token": token}
    return {}


def sanitize_tracker_value(value):
    if value is None:
        return "none"
    if isinstance(value, (bool, int, float, str, torch.Tensor)):
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_tracker_config(args: argparse.Namespace) -> Dict[str, object]:
    raw_config = {**vars(args), "target_prompts": list(TARGET_PROMPTS), "cfg_used": False}
    return {key: sanitize_tracker_value(value) for key, value in raw_config.items()}


def _validate_validation_args(args: argparse.Namespace) -> None:
    if args.validation_image_path and args.validation_water_mask_path:
        image_paths = [path.strip() for path in args.validation_image_path.split(",") if path.strip()]
        mask_paths = [path.strip() for path in args.validation_water_mask_path.split(",") if path.strip()]
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                "The number of --validation_image_path entries must match "
                "--validation_water_mask_path entries when using comma-separated paths."
            )


def _validate_data_source_args(args: argparse.Namespace) -> None:
    if not args.self_test and args.dataset_name is None:
        raise ValueError("Provide --dataset_name.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune InstructPix2Pix for intrinsic terrain decomposition."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Path or model id of an InstructPix2Pix checkpoint.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ShunTatsukawa/TAID-Dataset",
        help="Dataset name on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Optional dataset config.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="Input",
        help="Column containing the source terrain image.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default=None,
        help="Column containing the target intrinsic map.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default=None,
        help="Column containing one of the four intrinsic keywords.",
    )
    parser.add_argument(
        "--water_mask_column",
        type=str,
        default="water_mask",
        help="Column containing the water area mask image.",
    )
    parser.add_argument(
        "--scene_id_column",
        type=str,
        default="scene_id",
        help=(
            "Optional scene identifier column. "
            "When present, scene grouping uses this id instead of image hashing."
        ),
    )
    parser.add_argument("--albedo_column", default="A")
    parser.add_argument("--diffuse_column", default="D")
    parser.add_argument("--specular_column", default="S")
    parser.add_argument("--volume_column", default="V")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/decomposition/rgbx",
        help="Directory to write checkpoints and the final model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for datasets and models.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=resolve_hf_token(),
        help="Hugging Face access token for private datasets or models. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square resolution used for training and validation.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Center crop after resizing instead of random crop.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Apply paired horizontal flips.",
    )
    parser.add_argument(
        "--random_vertical_flip",
        action="store_true",
        help="Apply paired vertical flips.",
    )
    parser.add_argument(
        "--random_rotation_degrees",
        type=float,
        default=0.0,
        help="Maximum absolute paired rotation angle in degrees.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=6,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Override the total number of training steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap the number of training samples for debugging.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the U-Net.",
    )
    parser.add_argument(
        "--vae_decode_batch_size",
        type=int,
        default=4,
        help=(
            "Micro-batch size for VAE decode during loss computation. "
            "Smaller values reduce peak VRAM at the cost of throughput."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate by effective batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use bitsandbytes AdamW8bit.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 matmul on Ampere GPUs.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Track an exponential moving average of the U-Net.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        help="Optional revision for loading the non-EMA U-Net.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Relative logging directory under output_dir.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Tracker backend. Use "none" to disable trackers.',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Distributed training local rank.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save an Accelerator checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of Accelerator checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Checkpoint path or "latest".',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xFormers attention if installed.",
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        help="Enable VAE slicing to reduce decode memory usage.",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enable VAE tiling to reduce decode memory usage.",
    )
    parser.add_argument(
        "--validation_image_path",
        type=str,
        default=None,
        help="Local path to a terrain image used for validation inference.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One of the four target prompts used for periodic validation.",
    )
    parser.add_argument(
        "--validation_all_prompts",
        action="store_true",
        help="Generate validation images for all target prompts each validation cycle.",
    )
    parser.add_argument(
        "--validation_water_mask_path",
        type=str,
        default=None,
        help="Local path to a water mask image used for validation inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="How many validation samples to generate.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps for validation inference.",
    )
    parser.add_argument(
        "--loss_weight_diffusion",
        type=float,
        default=1.0,
        help="Weight for diffusion denoising loss.",
    )
    parser.add_argument(
        "--loss_weight_recon",
        type=float,
        default=0.5,
        help="Weight for physical reconstruction loss: |(A*Ds+Gs+V)-I|.",
    )
    parser.add_argument(
        "--loss_weight_direct",
        type=float,
        default=1.0,
        help="Weight for direct supervision loss in target component space.",
    )
    parser.add_argument(
        "--loss_weight_water",
        type=float,
        default=2.0,
        help=(
            "Weight for water-region A/D/V gradient, water-terrain V boundary, "
            "and water-region specular direct loss."
        ),
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run lightweight preprocessing tests without loading a model.",
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.validation_prompt is not None:
        args.validation_prompt = canonicalize_prompt(args.validation_prompt)

    if args.validation_all_prompts and args.validation_prompt is not None:
        logger.warning(
            "Both --validation_all_prompts and --validation_prompt were provided. "
            "--validation_all_prompts takes precedence."
        )

    _validate_validation_args(args)

    if args.random_rotation_degrees < 0:
        raise ValueError("--random_rotation_degrees must be non-negative.")

    _validate_data_source_args(args)

    return args
