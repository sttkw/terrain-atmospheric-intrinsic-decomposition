import argparse
import cv2
import json
import os
from typing import Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from PIL import Image

from pipeline import (
    PREDICTION_TYPE,
    TARGET_PROMPTS,
    canonicalize_prompt,
    component_space_to_linear,
    compose_intrinsic_hdr,
    generate_corrected_intrinsic_tensors,
    generate_intrinsic_tensor,
    load_image,
    load_mask_image,
    output_to_linear_components,
)

DEFAULT_BASE_MODEL_PATH = "timbrooks/instruct-pix2pix"


def validate_exactly_one(local_value: Optional[str], remote_value: Optional[str], local_name: str, remote_name: str) -> None:
    if bool(local_value) == bool(remote_value):
        raise ValueError(f"Specify exactly one of --{local_name} or --{remote_name}.")


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16

    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def is_pipeline_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model_index.json"))


def is_unet_checkpoint_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "unet", "config.json"))
        and (
            os.path.isfile(os.path.join(path, "unet", "diffusion_pytorch_model.safetensors"))
            or os.path.isfile(os.path.join(path, "unet", "diffusion_pytorch_model.bin"))
        )
    )


def resolve_checkpoint_base_model_path(model_path: str, base_model_path: Optional[str]) -> str:
    if base_model_path:
        return base_model_path

    parent_dir = os.path.dirname(os.path.abspath(model_path))
    if is_pipeline_dir(parent_dir):
        return parent_dir

    return os.environ.get("RGBX_BASE_MODEL_PATH", DEFAULT_BASE_MODEL_PATH)


def resolve_resolution(model_path: str, resolution_arg: Optional[int], base_model_path: Optional[str] = None) -> int:
    if resolution_arg is not None:
        return resolution_arg

    metadata_dirs = [model_path]
    if is_unet_checkpoint_dir(model_path):
        metadata_dirs.append(os.path.dirname(os.path.abspath(model_path)))
    if base_model_path:
        metadata_dirs.append(base_model_path)

    seen_dirs = set()
    for metadata_dir in metadata_dirs:
        if metadata_dir in seen_dirs:
            continue
        seen_dirs.add(metadata_dir)
        metadata_path = os.path.join(metadata_dir, "terrain_decomposition_config.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            resolution = metadata.get("resolution")
            if isinstance(resolution, int) and resolution > 0:
                return resolution

    raise ValueError(
        "Inference resolution is required. Provide --resolution or use a model directory "
        "or checkpoint parent containing terrain_decomposition_config.json with a positive integer resolution."
    )


def load_inputs(args: argparse.Namespace) -> tuple[Image.Image, Image.Image]:
    image = load_image(image_path=args.input_image, image_url=args.input_image_url)
    water_mask = (
        load_mask_image(image_path=args.water_mask, image_url=args.water_mask_url)
        if args.water_mask or args.water_mask_url
        else Image.new("RGB", image.size, color=(0, 0, 0))
    )
    if water_mask.size != image.size:
        water_mask = water_mask.resize(image.size, resample=Image.NEAREST)
    return image, water_mask


def build_pipeline(
    model_path: str,
    weight_dtype: torch.dtype,
    device: torch.device,
    base_model_path: Optional[str] = None,
):
    checkpoint_unet = None
    pipeline_path = model_path
    if is_unet_checkpoint_dir(model_path) and not is_pipeline_dir(model_path):
        pipeline_path = resolve_checkpoint_base_model_path(model_path, base_model_path)
        checkpoint_unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=weight_dtype,
        )

    pipeline_kwargs = {
        "torch_dtype": weight_dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if checkpoint_unet is not None:
        pipeline_kwargs["unet"] = checkpoint_unet

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        pipeline_path,
        **pipeline_kwargs,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.register_to_config(prediction_type=PREDICTION_TYPE)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.eval()
    pipeline.unet.eval()
    pipeline.vae.eval()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    return pipeline.to(device)


def create_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _linear_to_display_rgb(tensor_chw_linear: torch.Tensor, gamma: float = 2.2) -> np.ndarray:
    linear = tensor_chw_linear.detach().float().cpu().clamp(0.0, 1.0)
    display = linear.pow(1.0 / gamma)
    display_hwc = (display.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return display_hwc


def _save_exr_half_from_linear_tensor(linear_tensor_chw: torch.Tensor, exr_path: str) -> None:
    exr_hwc = linear_tensor_chw.detach().float().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    exr_bgr = exr_hwc[:, :, ::-1]

    params = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_HALF"):
        params = [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_HALF)]

    ok = cv2.imwrite(exr_path, exr_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to write EXR: {exr_path}")


def save_output_images(
    output_tensor_chw: torch.Tensor,
    output_path: str,
    prompt: str,
) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    base, ext = os.path.splitext(output_path)
    ext = ext.lower()
    if ext == ".png":
        png_path = output_path
        exr_path = f"{base}.exr"
    else:
        exr_path = output_path if ext == ".exr" else f"{output_path}.exr"
        png_path = f"{os.path.splitext(exr_path)[0]}.png"

    output_linear = component_space_to_linear(output_tensor_chw, prompt)

    _save_exr_half_from_linear_tensor(output_linear, exr_path)

    png_rgb = _linear_to_display_rgb(output_linear)
    Image.fromarray(png_rgb, mode="RGB").save(png_path)

    print(f"saved: {exr_path}")
    print(f"saved: {png_path}")
    return exr_path, png_path


def save_linear_image_pair(linear_tensor_chw: torch.Tensor, stem: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    exr_path = os.path.join(output_dir, f"{stem}.exr")
    png_path = os.path.join(output_dir, f"{stem}.png")
    _save_exr_half_from_linear_tensor(linear_tensor_chw, exr_path)
    Image.fromarray(_linear_to_display_rgb(linear_tensor_chw), mode="RGB").save(png_path)
    print(f"saved: {exr_path}")
    print(f"saved: {png_path}")


def save_corrected_outputs(
    outputs: dict[str, torch.Tensor],
    source_image_linear: torch.Tensor,
    output_dir: str,
) -> None:
    linear_outputs = output_to_linear_components(outputs)
    save_linear_image_pair(linear_outputs["Albedo"], "albedo", output_dir)
    save_linear_image_pair(linear_outputs["Diffuse Shading"], "diffuse_shading", output_dir)
    save_linear_image_pair(linear_outputs["Specular Shading"], "specular_shading", output_dir)
    save_linear_image_pair(linear_outputs["Volume"], "volume", output_dir)

    reconstruction = compose_intrinsic_hdr(
        albedo=linear_outputs["Albedo"],
        diffuse_shading=linear_outputs["Diffuse Shading"],
        specular_shading=linear_outputs["Specular Shading"],
        volume=linear_outputs["Volume"],
    )
    save_linear_image_pair(reconstruction, "reconstruction", output_dir)
    recon_l1 = (reconstruction.clamp(0.0, 1.0) - source_image_linear.clamp(0.0, 1.0)).abs().mean()
    print(f"reconstruction_l1_clamped: {float(recon_l1):.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGBX intrinsic decomposition inference in normal or corrected mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "corr"],
        help="normal generates one prompt map; corr jointly generates all maps with DPS reconstruction guidance.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a fine-tuned pipeline directory or an Accelerator checkpoint directory.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help=(
            "Base pipeline path/model id used when --model_path points to a checkpoint-* directory. "
            "Defaults to RGBX_BASE_MODEL_PATH or timbrooks/instruct-pix2pix."
        ),
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Local input terrain image path.",
    )
    parser.add_argument(
        "--input_image_url",
        type=str,
        default=None,
        help="Optional remote image URL.",
    )
    parser.add_argument(
        "--water_mask",
        type=str,
        default=None,
        help="Local path to a water mask image.",
    )
    parser.add_argument(
        "--water_mask_url",
        type=str,
        default=None,
        help="Optional remote water mask URL.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        choices=list(TARGET_PROMPTS),
        help="Target intrinsic map to generate: Albedo, Diffuse Shading, Specular Shading, or Volume.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the generated intrinsic map.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save all maps and reconstruction in corr mode.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Inference resolution. If omitted, read from terrain_decomposition_config.json.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of denoising steps. Defaults to 20 in normal mode and 50 in corr mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Weight dtype for inference.",
    )
    parser.add_argument(
        "--input_is_linear",
        action="store_true",
        help="Use when the input image is already linear. By default, input is treated as sRGB.",
    )
    parser.add_argument("--correction_strength", type=float, default=0.02)
    parser.add_argument("--correction_iters", type=int, default=1)
    parser.add_argument("--correction_latent_blend", type=float, default=1.0)
    parser.add_argument("--correction_start_step", type=int, default=None)
    parser.add_argument("--correction_end_step", type=int, default=None)
    parser.add_argument("--correction_every", type=int, default=1)
    parser.add_argument(
        "--correction_schedule",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine"],
        help="Per-step correction strength schedule between start/end steps.",
    )
    parser.add_argument(
        "--residual_chroma",
        type=float,
        default=1.0,
        help="0 uses luminance-only residual guidance; 1 uses full RGB residual guidance.",
    )
    parser.add_argument(
        "--correction_max_delta",
        type=float,
        default=0.015,
        help="Clamp per-iteration projection step in linear HDR units. Only used by optional final projection.",
    )
    parser.add_argument(
        "--guidance_lowpass_downsample",
        type=int,
        default=8,
        help="Downsample factor for low-frequency reconstruction guidance. 1 uses full resolution.",
    )
    parser.add_argument(
        "--guidance_grad_clip_rms",
        type=float,
        default=0.02,
        help="Maximum RMS of each normalized latent guidance update. Use <=0 to disable.",
    )
    parser.add_argument(
        "--guidance_prior_weight",
        type=float,
        default=0.05,
        help="Latent x0 prior weight for multi-iteration guidance within a denoise step.",
    )
    parser.add_argument(
        "--guidance_decode_batch_size",
        type=int,
        default=1,
        help="Micro-batch size for gradient-tracked VAE decode during guidance.",
    )
    parser.add_argument(
        "--unet_batch_size",
        type=int,
        default=4,
        help="Micro-batch size for the four-component UNet forward in corr mode. 4 keeps the original full batch.",
    )
    parser.add_argument(
        "--projection_mode",
        type=str,
        default="uniform",
        choices=["uniform", "sensitivity"],
        help="uniform splits reconstruction residual evenly by mobility; sensitivity uses a Jacobian projection.",
    )
    parser.add_argument(
        "--final_projection_strength",
        type=float,
        default=0.0,
        help="Optional weak output-space projection after denoising. 0 disables it.",
    )
    parser.add_argument(
        "--component_mobility",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("ALBEDO", "DIFFUSE", "SPECULAR", "VOLUME"),
    )
    parser.add_argument(
        "--independent_initial_noise",
        action="store_true",
        help="Use different initial noise for each component in corr mode.",
    )
    parser.add_argument("--disable_final_projection", action="store_true")
    args = parser.parse_args()

    validate_exactly_one(args.input_image, args.input_image_url, "input_image", "input_image_url")
    if args.water_mask and args.water_mask_url:
        raise ValueError("Specify at most one of --water_mask or --water_mask_url.")

    if args.num_inference_steps is None:
        args.num_inference_steps = 20 if args.mode == "normal" else 50
    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be positive.")
    if args.correction_start_step is None:
        args.correction_start_step = max(0, args.num_inference_steps - 15)

    if args.mode == "normal":
        if args.prompt is None:
            raise ValueError("--prompt is required in normal mode.")
        if args.output_path is None:
            raise ValueError("--output_path is required in normal mode.")
        args.prompt = canonicalize_prompt(args.prompt)
    elif args.output_dir is None:
        raise ValueError("--output_dir is required in corr mode.")
    return args


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    weight_dtype = resolve_dtype(args.dtype, device)
    resolution = resolve_resolution(args.model_path, args.resolution, args.base_model_path)

    image, water_mask = load_inputs(args)
    pipeline = build_pipeline(args.model_path, weight_dtype, device, args.base_model_path)
    generator = create_generator(device, args.seed)

    if args.mode == "normal":
        output_tensor = generate_intrinsic_tensor(
            prompt=args.prompt,
            image=image,
            water_mask=water_mask,
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            resolution=resolution,
            num_inference_steps=args.num_inference_steps,
            device=device,
            weight_dtype=weight_dtype,
            generator=generator,
            input_is_srgb=not args.input_is_linear,
            show_progress=True,
            progress_desc=f"Inference [{args.prompt}]",
        )

        save_output_images(output_tensor, args.output_path, args.prompt)
        return

    outputs, source_image_linear = generate_corrected_intrinsic_tensors(
        image=image,
        water_mask=water_mask,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        vae=pipeline.vae,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        resolution=resolution,
        num_inference_steps=args.num_inference_steps,
        device=device,
        weight_dtype=weight_dtype,
        generator=generator,
        input_is_srgb=not args.input_is_linear,
        correction_strength=args.correction_strength,
        correction_iters=args.correction_iters,
        correction_latent_blend=args.correction_latent_blend,
        correction_start_step=args.correction_start_step,
        correction_end_step=args.correction_end_step,
        correction_every=args.correction_every,
        correction_schedule=args.correction_schedule,
        residual_chroma=args.residual_chroma,
        correction_max_delta=args.correction_max_delta,
        projection_mode=args.projection_mode,
        guidance_lowpass_downsample=args.guidance_lowpass_downsample,
        guidance_grad_clip_rms=args.guidance_grad_clip_rms,
        guidance_prior_weight=args.guidance_prior_weight,
        guidance_decode_batch_size=args.guidance_decode_batch_size,
        final_projection=not args.disable_final_projection,
        final_projection_strength=args.final_projection_strength,
        component_mobility=tuple(args.component_mobility),
        shared_initial_noise=not args.independent_initial_noise,
        unet_batch_size=args.unet_batch_size,
        show_progress=True,
        progress_desc="DPS-guided inference",
    )
    save_corrected_outputs(outputs, source_image_linear, args.output_dir)


if __name__ == "__main__":
    main()
