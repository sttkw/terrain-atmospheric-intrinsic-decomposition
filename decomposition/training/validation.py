import argparse
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from pipeline import (
    TARGET_PROMPTS,
    component_space_to_linear,
    compose_intrinsic_hdr,
    generate_intrinsic_tensor,
    load_image,
    load_mask_image,
)
from training.utils import (
    _linear_to_display_png_tensor,
    _save_exr_tensor,
)

ValidationPair = Tuple[str, Image.Image, Image.Image]


def load_validation_pairs(args: argparse.Namespace, logger) -> List[ValidationPair]:
    validation_pairs: List[ValidationPair] = []
    if not args.validation_image_path:
        return validation_pairs

    validation_image_paths = [
        path.strip() for path in args.validation_image_path.split(",") if path.strip()
    ]
    validation_mask_paths = [] if not args.validation_water_mask_path else [
        path.strip() for path in args.validation_water_mask_path.split(",") if path.strip()
    ]
    if not validation_mask_paths:
        validation_mask_paths = [None] * len(validation_image_paths)
    logger.info("Validation input is treated as already linear LDR.")
    for pair_index, (image_path, mask_path) in enumerate(
        zip(validation_image_paths, validation_mask_paths)
    ):
        validation_image = load_image(image_path=image_path, image_url=None)
        validation_water_mask = (
            load_mask_image(image_path=mask_path, image_url=None)
            if mask_path is not None
            else Image.new("RGB", validation_image.size, color=(0, 0, 0))
        )
        scene_slug = os.path.splitext(os.path.basename(image_path))[0] or f"val-{pair_index:02d}"
        validation_pairs.append((scene_slug, validation_image, validation_water_mask))
    return validation_pairs


@torch.no_grad()
def export_validation_images(
    *,
    args: argparse.Namespace,
    validation_pairs: List[ValidationPair],
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
    accelerator,
    weight_dtype: torch.dtype,
    generator: torch.Generator,
    global_step: int,
) -> None:
    validation_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    for scene_slug, validation_image, validation_water_mask in validation_pairs:
        scene_validation_dir = os.path.join(
            validation_dir,
            f"step-{global_step:06d}-{scene_slug}",
        )
        os.makedirs(scene_validation_dir, exist_ok=True)

        for sample_index in range(args.num_validation_images):
            component_outputs: Dict[str, torch.Tensor] = {}
            for prompt_name in TARGET_PROMPTS:
                component_outputs[prompt_name] = generate_intrinsic_tensor(
                    prompt=prompt_name,
                    image=validation_image,
                    water_mask=validation_water_mask,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    scheduler=scheduler,
                    resolution=args.resolution,
                    num_inference_steps=args.validation_num_inference_steps,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    generator=generator,
                    input_is_srgb=True,
                )

            albedo = component_outputs["Albedo"]
            diffuse_hdr = component_space_to_linear(component_outputs["Diffuse Shading"], "Diffuse Shading")
            specular_hdr = component_space_to_linear(component_outputs["Specular Shading"], "Specular Shading")
            volume_hdr = component_space_to_linear(component_outputs["Volume"], "Volume")

            recon_hdr = compose_intrinsic_hdr(
                albedo=albedo,
                diffuse_shading=diffuse_hdr,
                specular_shading=specular_hdr,
                volume=volume_hdr,
            )
            recon_png = recon_hdr.clamp(0.0, 1.0)

            sample_prefix = (
                f"sample-{sample_index:02d}-"
                if args.num_validation_images > 1
                else ""
            )

            _save_exr_tensor(os.path.join(scene_validation_dir, f"{sample_prefix}albedo.exr"), albedo)
            _save_exr_tensor(
                os.path.join(scene_validation_dir, f"{sample_prefix}diffuse_shading.exr"),
                diffuse_hdr,
            )
            _save_exr_tensor(
                os.path.join(scene_validation_dir, f"{sample_prefix}specular_shading.exr"),
                specular_hdr,
            )
            _save_exr_tensor(
                os.path.join(scene_validation_dir, f"{sample_prefix}volume.exr"),
                volume_hdr,
            )
            _save_exr_tensor(
                os.path.join(scene_validation_dir, f"{sample_prefix}reconstruction.exr"),
                recon_hdr,
            )

            TF.to_pil_image(_linear_to_display_png_tensor(albedo)).save(
                os.path.join(scene_validation_dir, f"{sample_prefix}albedo.png")
            )
            TF.to_pil_image(_linear_to_display_png_tensor(diffuse_hdr)).save(
                os.path.join(scene_validation_dir, f"{sample_prefix}diffuse_shading.png")
            )
            TF.to_pil_image(_linear_to_display_png_tensor(specular_hdr)).save(
                os.path.join(scene_validation_dir, f"{sample_prefix}specular_shading.png")
            )
            TF.to_pil_image(_linear_to_display_png_tensor(volume_hdr)).save(
                os.path.join(scene_validation_dir, f"{sample_prefix}volume.png")
            )
            TF.to_pil_image(_linear_to_display_png_tensor(recon_png)).save(
                os.path.join(scene_validation_dir, f"{sample_prefix}reconstruction.png")
            )
