import math
import requests
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Tuple

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF
from transformers import CLIPTextModel, CLIPTokenizer

TARGET_PROMPTS: Tuple[str, ...] = (
    "Albedo",
    "Diffuse Shading",
    "Specular Shading",
    "Volume",
)
PROMPT_TO_INDEX = {prompt: index for index, prompt in enumerate(TARGET_PROMPTS)}
PREDICTION_TYPE = "v_prediction"
LOG_FORWARD_S_MAX = {
    "Diffuse Shading": 5.0,
}
MASK_WATER_INDEX = 0
MASK_TERRAIN_INDEX = 1
MASK_SKY_INDEX = 2
MASK_CONDITIONING_CHANNELS = 2
PROMPT_LOOKUP = {prompt.lower(): prompt for prompt in TARGET_PROMPTS}
PROMPT_ALIASES = {
    "shading": "Diffuse Shading",
    "diffuse": "Diffuse Shading",
    "gloss": "Specular Shading",
    "glossy": "Specular Shading",
    "spec": "Specular Shading",
    "specular": "Specular Shading",
}


def canonicalize_prompt(prompt: str) -> str:
    normalized = prompt.strip().lower()
    normalized = PROMPT_ALIASES.get(normalized, normalized)
    if normalized not in PROMPT_LOOKUP:
        allowed = ", ".join(TARGET_PROMPTS)
        raise ValueError(f"Unsupported prompt '{prompt}'. Expected one of: {allowed}.")
    return PROMPT_LOOKUP[normalized]


def to_unit_range(image_tensor: torch.Tensor) -> torch.Tensor:
    return ((image_tensor / 2.0) + 0.5).clamp(0.0, 1.0)


def _log_forward_s_max(prompt: str, like: torch.Tensor) -> torch.Tensor:
    canonical_prompt = canonicalize_prompt(prompt)
    if canonical_prompt not in LOG_FORWARD_S_MAX:
        raise ValueError(f"Prompt '{canonical_prompt}' does not use log-forward S01 encoding.")
    return torch.as_tensor(
        LOG_FORWARD_S_MAX[canonical_prompt],
        device=like.device,
        dtype=like.dtype,
    )


def hdr_to_log_forward_space(image_tensor: torch.Tensor, prompt: str) -> torch.Tensor:
    s_max = _log_forward_s_max(prompt, image_tensor)
    return (torch.log1p(image_tensor.clamp(min=0.0)) / torch.log1p(s_max)).clamp(0.0, 1.0)


def log_forward_space_to_hdr(image_tensor: torch.Tensor, prompt: str) -> torch.Tensor:
    s_max = _log_forward_s_max(prompt, image_tensor)
    return torch.expm1(image_tensor.clamp(0.0, 1.0) * torch.log1p(s_max))


def component_space_to_linear(image_tensor: torch.Tensor, prompt: str) -> torch.Tensor:
    canonical_prompt = canonicalize_prompt(prompt)
    if canonical_prompt in LOG_FORWARD_S_MAX:
        return log_forward_space_to_hdr(image_tensor, canonical_prompt)
    return image_tensor.clamp(0.0, 1.0)


def linear_to_component_space(image_tensor: torch.Tensor, prompt: str) -> torch.Tensor:
    canonical_prompt = canonicalize_prompt(prompt)
    if canonical_prompt in LOG_FORWARD_S_MAX:
        return hdr_to_log_forward_space(image_tensor, canonical_prompt)
    return image_tensor.clamp(0.0, 1.0)


def select_mask_conditioning_channels(mask_onehot: torch.Tensor) -> torch.Tensor:
    if mask_onehot.shape[1] <= MASK_SKY_INDEX:
        raise ValueError(
            "Expected mask channels [water, terrain, sky]. "
            f"Got shape {tuple(mask_onehot.shape)}."
        )
    return torch.cat(
        [
            mask_onehot[:, MASK_WATER_INDEX:MASK_WATER_INDEX + 1],
            mask_onehot[:, MASK_SKY_INDEX:MASK_SKY_INDEX + 1],
        ],
        dim=1,
    )


def compose_intrinsic_image(
    albedo: torch.Tensor,
    diffuse_shading: torch.Tensor,
    specular_shading: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    return (albedo * diffuse_shading + specular_shading + volume).clamp(0.0, 1.0)


def compose_intrinsic_hdr(
    albedo: torch.Tensor,
    diffuse_shading: torch.Tensor,
    specular_shading: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    return albedo * diffuse_shading + specular_shading + volume


def _open_image_from_path_or_url(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Image.Image:
    if image_path is not None:
        return Image.open(image_path)
    if image_url is not None:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw)
    raise ValueError("Either image_path or image_url must be provided.")


def load_image(image_path: Optional[str] = None, image_url: Optional[str] = None) -> Image.Image:
    image = _open_image_from_path_or_url(image_path=image_path, image_url=image_url)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_mask_image(image_path: Optional[str] = None, image_url: Optional[str] = None) -> Image.Image:
    image = _open_image_from_path_or_url(image_path=image_path, image_url=image_url)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def srgb_pil_to_linear_pil(image: Image.Image, gamma: float = 2.2) -> Image.Image:
    srgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0
    linear = np.power(np.clip(srgb, 0.0, 1.0), gamma)
    linear_u8 = (linear * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(linear_u8, mode="RGB")


def prepare_image(
    image: Image.Image,
    resolution: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if size is not None:
        image = image.resize(size, resample=Image.BICUBIC)
    elif resolution is not None:
        image = image.resize((resolution, resolution), resample=Image.BICUBIC)
    tensor = TF.to_tensor(image)
    return tensor * 2.0 - 1.0


def prepare_linear_image(
    image: Image.Image,
    resolution: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if size is not None:
        image = image.resize(size, resample=Image.BICUBIC)
    elif resolution is not None:
        image = image.resize((resolution, resolution), resample=Image.BICUBIC)
    return TF.to_tensor(image).clamp(0.0, 1.0)


def prepare_mask_for_vae(
    mask_image: Image.Image,
    resolution: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    mask_image = ImageOps.exif_transpose(mask_image).convert("L")
    if size is not None:
        mask_image = mask_image.resize(size, resample=Image.NEAREST)
    elif resolution is not None:
        mask_image = mask_image.resize((resolution, resolution), resample=Image.NEAREST)
    mask_tensor = TF.to_tensor(mask_image)
    mask_tensor = mask_tensor.repeat(3, 1, 1)
    return mask_tensor * 2.0 - 1.0


def prepare_segmentation_onehot(
    mask_image: Image.Image,
    resolution: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Convert an RGB segmentation map to one-hot channels [water, terrain, sky]."""
    mask_image = ImageOps.exif_transpose(mask_image).convert("RGB")
    if size is not None:
        mask_image = mask_image.resize(size, resample=Image.NEAREST)
    elif resolution is not None:
        mask_image = mask_image.resize((resolution, resolution), resample=Image.NEAREST)

    mask_tensor = TF.to_tensor(mask_image)
    class_indices = torch.argmax(mask_tensor, dim=0)
    one_hot = F.one_hot(class_indices, num_classes=3).permute(2, 0, 1).float()

    # Ignore pixels that are effectively unlabeled (e.g. rotation fill values).
    valid = (mask_tensor.max(dim=0).values >= 0.5).float().unsqueeze(0)
    return one_hot * valid


def tokenize_prompts(tokenizer: CLIPTokenizer, prompts: Iterable[str]) -> torch.Tensor:
    tokenized = tokenizer(
        list(prompts),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized.input_ids


def get_autocast_context(device: torch.device, weight_dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if weight_dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=weight_dtype)


def round_to_multiple_of_8(value: int) -> int:
    return max(8, int(round(value / 8.0)) * 8)


def resolve_inference_size(source_size: Tuple[int, int], target_short_side: int) -> Tuple[int, int]:
    width, height = source_size
    short_side = min(width, height)
    if short_side <= 0:
        raise ValueError("Input image must have positive size.")
    scale = float(target_short_side) / float(short_side)
    return (
        round_to_multiple_of_8(int(round(width * scale))),
        round_to_multiple_of_8(int(round(height * scale))),
    )


def decode_latents_to_unit_maps(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    return to_unit_range(decoded.float())


def decode_latents_to_unit_maps_chunked(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    decode_batch_size: int,
) -> torch.Tensor:
    if decode_batch_size <= 0 or decode_batch_size >= latents.shape[0]:
        return decode_latents_to_unit_maps(vae, latents)

    decoded_chunks = []
    for latent_chunk in latents.split(decode_batch_size, dim=0):
        decoded_chunks.append(decode_latents_to_unit_maps(vae, latent_chunk))
    return torch.cat(decoded_chunks, dim=0)


def resize_sky_mask(mask_onehot: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    sky_mask = mask_onehot[:, MASK_SKY_INDEX:MASK_SKY_INDEX + 1].float()
    if sky_mask.shape[-2:] != like.shape[-2:]:
        sky_mask = F.interpolate(sky_mask, size=like.shape[-2:], mode="nearest")
    return sky_mask.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)


def enforce_sky_volume_only(
    *,
    albedo: torch.Tensor,
    diffuse_shading: torch.Tensor,
    specular_shading: torch.Tensor,
    volume: torch.Tensor,
    source_image_linear: torch.Tensor,
    sky_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    non_sky = 1.0 - sky_mask
    return (
        albedo * non_sky,
        diffuse_shading * non_sky,
        specular_shading * non_sky,
        volume * non_sky + source_image_linear * sky_mask,
    )


def project_intrinsic_components_to_input(
    *,
    component_maps: torch.Tensor,
    source_image_linear: torch.Tensor,
    mask_onehot: torch.Tensor,
    strength: float,
    num_iters: int,
    component_mobility: Tuple[float, float, float, float],
    residual_chroma: float,
    max_delta: float,
    projection_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project [A, D01, S, V] toward I = A * D + S + V in linear space."""
    original_dtype = component_maps.dtype
    maps = component_maps.float().clamp(0.0, 1.0)
    source = source_image_linear.float().clamp(0.0, 1.0)
    if source.shape[-2:] != maps.shape[-2:]:
        source = F.interpolate(source, size=maps.shape[-2:], mode="bilinear", align_corners=False)
    source = source.to(device=maps.device, dtype=maps.dtype)

    sky_mask = resize_sky_mask(mask_onehot, maps[0:1])
    non_sky = 1.0 - sky_mask
    mobility = torch.as_tensor(component_mobility, device=maps.device, dtype=maps.dtype).clamp(min=0.0)
    albedo_mobility, diffuse_mobility, specular_mobility, volume_mobility = mobility

    albedo = maps[PROMPT_TO_INDEX["Albedo"]:PROMPT_TO_INDEX["Albedo"] + 1]
    diffuse_shading = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Diffuse Shading"]:PROMPT_TO_INDEX["Diffuse Shading"] + 1],
        "Diffuse Shading",
    ).float()
    specular_shading = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Specular Shading"]:PROMPT_TO_INDEX["Specular Shading"] + 1],
        "Specular Shading",
    ).float()
    volume = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Volume"]:PROMPT_TO_INDEX["Volume"] + 1],
        "Volume",
    ).float()

    diffuse_max = _log_forward_s_max("Diffuse Shading", diffuse_shading)
    projection_strength = max(0.0, float(strength))

    for _ in range(max(0, int(num_iters))):
        albedo, diffuse_shading, specular_shading, volume = enforce_sky_volume_only(
            albedo=albedo,
            diffuse_shading=diffuse_shading,
            specular_shading=specular_shading,
            volume=volume,
            source_image_linear=source,
            sky_mask=sky_mask,
        )
        reconstruction = compose_intrinsic_hdr(albedo, diffuse_shading, specular_shading, volume)
        residual = (source - reconstruction) * non_sky * projection_strength
        if residual_chroma < 1.0:
            residual_luma = (
                0.2126 * residual[:, 0:1]
                + 0.7152 * residual[:, 1:2]
                + 0.0722 * residual[:, 2:3]
            )
            residual = residual_luma + max(0.0, residual_chroma) * (residual - residual_luma)

        if projection_mode == "uniform":
            weight_sum = mobility.sum().clamp_min(eps)
            albedo_weight = albedo_mobility / weight_sum
            diffuse_weight = diffuse_mobility / weight_sum
            specular_weight = specular_mobility / weight_sum
            volume_weight = volume_mobility / weight_sum

            delta_albedo = residual * albedo_weight / diffuse_shading.clamp_min(eps)
            delta_diffuse = residual * diffuse_weight / albedo.clamp_min(eps)
            delta_specular = residual * specular_weight
            delta_volume = residual * volume_weight
            if max_delta > 0.0:
                clamp_value = float(max_delta)
                delta_albedo = delta_albedo.clamp(-clamp_value, clamp_value)
                delta_diffuse = delta_diffuse.clamp(-clamp_value, clamp_value)
                delta_specular = delta_specular.clamp(-clamp_value, clamp_value)
                delta_volume = delta_volume.clamp(-clamp_value, clamp_value)

            albedo = (albedo + delta_albedo).clamp(0.0, 1.0)
            diffuse_shading = (diffuse_shading + delta_diffuse).clamp(0.0, diffuse_max)
            specular_shading = (specular_shading + delta_specular).clamp(0.0, 1.0)
            volume = (volume + delta_volume).clamp(0.0, 1.0)
        elif projection_mode == "sensitivity":
            jac_albedo = diffuse_shading
            jac_diffuse = albedo
            denom = (
                albedo_mobility * jac_albedo.square()
                + diffuse_mobility * jac_diffuse.square()
                + specular_mobility
                + volume_mobility
                + eps
            )
            step = residual / denom
            if max_delta > 0.0:
                step = step.clamp(-float(max_delta), float(max_delta))

            albedo = (albedo + albedo_mobility * jac_albedo * step).clamp(0.0, 1.0)
            diffuse_shading = (diffuse_shading + diffuse_mobility * jac_diffuse * step).clamp(
                0.0, diffuse_max
            )
            specular_shading = (specular_shading + specular_mobility * step).clamp(0.0, 1.0)
            volume = (volume + volume_mobility * step).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported projection mode: {projection_mode}")

    albedo, diffuse_shading, specular_shading, volume = enforce_sky_volume_only(
        albedo=albedo,
        diffuse_shading=diffuse_shading,
        specular_shading=specular_shading,
        volume=volume,
        source_image_linear=source,
        sky_mask=sky_mask,
    )
    corrected = torch.cat(
        [
            albedo,
            linear_to_component_space(diffuse_shading, "Diffuse Shading"),
            linear_to_component_space(specular_shading, "Specular Shading"),
            linear_to_component_space(volume, "Volume"),
        ],
        dim=0,
    )
    return corrected.clamp(0.0, 1.0).to(dtype=original_dtype)


def enforce_sky_only_on_unit_maps(
    *,
    component_maps: torch.Tensor,
    source_image_linear: torch.Tensor,
    mask_onehot: torch.Tensor,
) -> torch.Tensor:
    maps = component_maps.clone()
    sky_mask = resize_sky_mask(mask_onehot, maps[0:1])
    non_sky = 1.0 - sky_mask

    maps[PROMPT_TO_INDEX["Albedo"]:PROMPT_TO_INDEX["Albedo"] + 1] *= non_sky
    maps[PROMPT_TO_INDEX["Diffuse Shading"]:PROMPT_TO_INDEX["Diffuse Shading"] + 1] *= non_sky
    maps[PROMPT_TO_INDEX["Specular Shading"]:PROMPT_TO_INDEX["Specular Shading"] + 1] *= non_sky
    return maps.clamp(0.0, 1.0)


def timestep_to_index(timestep) -> int:
    if isinstance(timestep, torch.Tensor):
        return int(timestep.detach().cpu().item())
    return int(timestep)


def should_apply_correction(
    *,
    step_index: int,
    total_steps: int,
    start_step: int,
    end_step: Optional[int],
    every: int,
) -> bool:
    end = total_steps if end_step is None else min(end_step, total_steps)
    if step_index < max(0, start_step) or step_index >= end:
        return False
    return every <= 1 or ((step_index - max(0, start_step)) % every == 0)


def scheduled_correction_strength(
    *,
    base_strength: float,
    schedule: str,
    step_index: int,
    total_steps: int,
    start_step: int,
    end_step: Optional[int],
) -> float:
    end = total_steps if end_step is None else min(end_step, total_steps)
    start = max(0, start_step)
    if base_strength <= 0.0 or step_index < start or step_index >= end:
        return 0.0
    if schedule == "constant":
        return float(base_strength)

    span = max(1, end - start - 1)
    progress = min(1.0, max(0.0, float(step_index - start) / float(span)))
    if schedule == "linear":
        scale = progress
    elif schedule == "cosine":
        scale = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unsupported correction schedule: {schedule}")
    return float(base_strength) * scale


def lowpass_tensor(tensor: torch.Tensor, downsample: int) -> torch.Tensor:
    if downsample <= 1:
        return tensor
    pooled = F.avg_pool2d(
        tensor,
        kernel_size=downsample,
        stride=downsample,
        ceil_mode=True,
    )
    return F.interpolate(pooled, size=tensor.shape[-2:], mode="bilinear", align_corners=False)


def reconstruction_guidance_loss(
    *,
    component_maps: torch.Tensor,
    source_image_linear: torch.Tensor,
    mask_onehot: torch.Tensor,
    residual_chroma: float,
    lowpass_downsample: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    maps = component_maps.float().clamp(0.0, 1.0)
    source = source_image_linear.float().clamp(0.0, 1.0)
    if source.shape[-2:] != maps.shape[-2:]:
        source = F.interpolate(source, size=maps.shape[-2:], mode="bilinear", align_corners=False)
    source = source.to(device=maps.device, dtype=maps.dtype)

    albedo = maps[PROMPT_TO_INDEX["Albedo"]:PROMPT_TO_INDEX["Albedo"] + 1]
    diffuse_shading = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Diffuse Shading"]:PROMPT_TO_INDEX["Diffuse Shading"] + 1],
        "Diffuse Shading",
    ).float()
    specular_shading = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Specular Shading"]:PROMPT_TO_INDEX["Specular Shading"] + 1],
        "Specular Shading",
    ).float()
    volume = component_space_to_linear(
        maps[PROMPT_TO_INDEX["Volume"]:PROMPT_TO_INDEX["Volume"] + 1],
        "Volume",
    ).float()
    reconstruction = compose_intrinsic_hdr(albedo, diffuse_shading, specular_shading, volume)

    reconstruction_lf = lowpass_tensor(reconstruction, lowpass_downsample)
    source_lf = lowpass_tensor(source, lowpass_downsample)
    residual = reconstruction_lf - source_lf
    if residual_chroma < 1.0:
        residual_luma = (
            0.2126 * residual[:, 0:1]
            + 0.7152 * residual[:, 1:2]
            + 0.0722 * residual[:, 2:3]
        )
        residual = residual_luma + max(0.0, residual_chroma) * (residual - residual_luma)

    valid_mask = (mask_onehot.float().sum(dim=1, keepdim=True) >= 0.5).to(
        device=residual.device,
        dtype=residual.dtype,
    )
    if valid_mask.shape[-2:] != residual.shape[-2:]:
        valid_mask = F.interpolate(valid_mask, size=residual.shape[-2:], mode="nearest")
    loss_map = torch.sqrt(residual.square() + eps * eps) * valid_mask
    denom = valid_mask.sum().clamp_min(1.0) * residual.shape[1]
    return loss_map.sum() / denom


def normalized_guidance_update(
    grad: torch.Tensor,
    step_size: float,
    max_update_rms: float,
) -> torch.Tensor:
    grad_rms = grad.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    update = grad / grad_rms.to(dtype=grad.dtype)
    update = update * float(step_size)
    if max_update_rms > 0.0:
        update_rms = update.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        update_scale = (float(max_update_rms) / update_rms).clamp(max=1.0)
        update = update * update_scale.to(dtype=update.dtype)
    return update


def apply_dps_guidance_to_latents(
    *,
    vae: AutoencoderKL,
    scheduler: DDPMScheduler,
    current_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    source_image_linear: torch.Tensor,
    mask_onehot: torch.Tensor,
    guidance_scale: float,
    guidance_iters: int,
    guidance_lowpass_downsample: int,
    guidance_grad_clip_rms: float,
    guidance_prior_weight: float,
    guidance_decode_batch_size: int,
    residual_chroma: float,
) -> torch.Tensor:
    if guidance_scale <= 0.0 or guidance_iters <= 0:
        return current_latents.detach()

    guided_latents = current_latents.detach()
    prior_pred_original_sample = None
    detached_noise_pred = noise_pred.detach()
    for _ in range(guidance_iters):
        latents_for_grad = guided_latents.detach().requires_grad_(True)
        step_output = scheduler.step(detached_noise_pred, timestep, latents_for_grad)
        pred_original_sample = getattr(step_output, "pred_original_sample", None)
        if pred_original_sample is None:
            pred_original_sample = latents_for_grad
        if prior_pred_original_sample is None:
            prior_pred_original_sample = pred_original_sample.detach()

        decoded_maps = decode_latents_to_unit_maps_chunked(
            vae,
            pred_original_sample,
            decode_batch_size=guidance_decode_batch_size,
        )
        loss = reconstruction_guidance_loss(
            component_maps=decoded_maps,
            source_image_linear=source_image_linear,
            mask_onehot=mask_onehot.float(),
            residual_chroma=residual_chroma,
            lowpass_downsample=guidance_lowpass_downsample,
        )
        if guidance_prior_weight > 0.0:
            loss = loss + float(guidance_prior_weight) * F.mse_loss(
                pred_original_sample.float(),
                prior_pred_original_sample.float(),
            )

        grad = torch.autograd.grad(loss, latents_for_grad, retain_graph=False, create_graph=False)[0]
        update = normalized_guidance_update(
            grad=grad,
            step_size=guidance_scale,
            max_update_rms=guidance_grad_clip_rms,
        )
        guided_latents = (latents_for_grad - update.to(dtype=latents_for_grad.dtype)).detach()
        del decoded_maps, loss, grad, update, latents_for_grad, step_output, pred_original_sample
    return guided_latents


def predict_noise_unet_chunked(
    *,
    unet: UNet2DConditionModel,
    model_input: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0 or batch_size >= model_input.shape[0]:
        return unet(
            model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    noise_chunks = []
    for start in range(0, model_input.shape[0], batch_size):
        end = start + batch_size
        noise_chunks.append(
            unet(
                model_input[start:end],
                timestep,
                encoder_hidden_states=encoder_hidden_states[start:end],
            ).sample
        )
    return torch.cat(noise_chunks, dim=0)


@torch.no_grad()
def generate_intrinsic_tensor(
    *,
    prompt: str,
    image: Image.Image,
    water_mask: Image.Image,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    resolution: int,
    num_inference_steps: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    generator: Optional[torch.Generator],
    input_is_srgb: bool = False,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> torch.Tensor:
    def _round_to_multiple_of_8(value: int) -> int:
        return max(8, int(round(value / 8.0)) * 8)

    def _resolve_inference_size(source_size: Tuple[int, int], target_short_side: int) -> Tuple[int, int]:
        width, height = source_size
        short_side = min(width, height)
        if short_side <= 0:
            raise ValueError("Input image must have positive size.")
        scale = float(target_short_side) / float(short_side)
        resized_width = _round_to_multiple_of_8(int(round(width * scale)))
        resized_height = _round_to_multiple_of_8(int(round(height * scale)))
        return resized_width, resized_height

    canonical_prompt = canonicalize_prompt(prompt)
    inference_size = _resolve_inference_size(image.size, resolution)
    if input_is_srgb:
        image = srgb_pil_to_linear_pil(image)
    image_tensor = prepare_image(image, size=inference_size).unsqueeze(0).to(
        device=device,
        dtype=weight_dtype,
    )
    mask_onehot = prepare_segmentation_onehot(water_mask, size=inference_size).unsqueeze(0).to(
        device=device, dtype=weight_dtype
    )
    input_ids = tokenize_prompts(tokenizer, [canonical_prompt]).to(device)

    encoder_hidden_states = text_encoder(input_ids)[0]
    image_latents = vae.encode(image_tensor).latent_dist.mode()

    latent_height = image_latents.shape[-2]
    latent_width = image_latents.shape[-1]
    mask_condition = select_mask_conditioning_channels(mask_onehot)
    mask_condition_latents = F.interpolate(
        mask_condition,
        size=(latent_height, latent_width),
        mode="nearest",
    )

    latents = torch.randn(
        (1, unet.config.out_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    scheduler = scheduler.__class__.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    timestep_iterator = scheduler.timesteps
    if show_progress:
        timestep_iterator = tqdm(
            scheduler.timesteps,
            desc=progress_desc or f"Inference [{canonical_prompt}]",
            total=len(scheduler.timesteps),
        )

    autocast_context = get_autocast_context(device, weight_dtype)
    with autocast_context:
        for timestep in timestep_iterator:
            latent_model_input = scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat([latent_model_input, image_latents, mask_condition_latents], dim=1)
            noise_pred = unet(
                model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            latents = scheduler.step(noise_pred, timestep, latents, generator=generator).prev_sample

        decoded = vae.decode(latents / vae.config.scaling_factor).sample

    decoded = ((decoded / 2.0) + 0.5).clamp(0, 1)
    output = decoded[0].detach().float().cpu()

    if canonical_prompt in ("Albedo", "Diffuse Shading", "Specular Shading"):
        # Sky is represented by Volume only.
        sky_mask = mask_onehot[0, 2].detach().float().cpu() >= 0.5
        output[:, sky_mask] = 0.0

    return output


def generate_corrected_intrinsic_tensors(
    *,
    image: Image.Image,
    water_mask: Image.Image,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    resolution: int,
    num_inference_steps: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    generator: Optional[torch.Generator],
    input_is_srgb: bool,
    correction_strength: float,
    correction_iters: int,
    correction_latent_blend: float,
    correction_start_step: int,
    correction_end_step: Optional[int],
    correction_every: int,
    correction_schedule: str,
    residual_chroma: float,
    correction_max_delta: float,
    projection_mode: str,
    guidance_lowpass_downsample: int,
    guidance_grad_clip_rms: float,
    guidance_prior_weight: float,
    guidance_decode_batch_size: int,
    final_projection: bool,
    final_projection_strength: float,
    component_mobility: Tuple[float, float, float, float],
    shared_initial_noise: bool,
    unet_batch_size: int = 0,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inference_size = resolve_inference_size(image.size, resolution)
    model_image = srgb_pil_to_linear_pil(image) if input_is_srgb else image

    image_tensor = prepare_image(model_image, size=inference_size).unsqueeze(0).to(
        device=device,
        dtype=weight_dtype,
    )
    source_image_linear = prepare_linear_image(model_image, size=inference_size).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    mask_onehot = prepare_segmentation_onehot(water_mask, size=inference_size).unsqueeze(0).to(
        device=device,
        dtype=weight_dtype,
    )

    input_ids = tokenize_prompts(tokenizer, TARGET_PROMPTS).to(device)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]
        image_latents_single = vae.encode(image_tensor).latent_dist.mode()
    latent_height = image_latents_single.shape[-2]
    latent_width = image_latents_single.shape[-1]

    image_latents = image_latents_single.repeat(len(TARGET_PROMPTS), 1, 1, 1)
    mask_condition = select_mask_conditioning_channels(mask_onehot)
    mask_condition_latents = F.interpolate(
        mask_condition,
        size=(latent_height, latent_width),
        mode="nearest",
    ).repeat(len(TARGET_PROMPTS), 1, 1, 1)

    latent_shape = (1, unet.config.out_channels, latent_height, latent_width)
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    if shared_initial_noise:
        latents = latents.repeat(len(TARGET_PROMPTS), 1, 1, 1)
    else:
        extra_latents = torch.randn(
            (len(TARGET_PROMPTS) - 1, unet.config.out_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = torch.cat([latents, extra_latents], dim=0)

    scheduler = scheduler.__class__.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    timesteps = scheduler.timesteps
    timestep_iterator = timesteps
    if show_progress:
        timestep_iterator = tqdm(
            timesteps,
            desc=progress_desc or "DPS-guided inference",
            total=len(timesteps),
        )

    autocast_context = get_autocast_context(device, weight_dtype)
    for step_index, timestep in enumerate(timestep_iterator):
        with torch.no_grad(), autocast_context:
            latent_model_input = scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat([latent_model_input, image_latents, mask_condition_latents], dim=1)
            noise_pred = predict_noise_unet_chunked(
                unet=unet,
                model_input=model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                batch_size=max(0, int(unet_batch_size)),
            )

        guided_latents = latents
        if should_apply_correction(
            step_index=step_index,
            total_steps=len(timesteps),
            start_step=correction_start_step,
            end_step=correction_end_step,
            every=max(1, correction_every),
        ):
            effective_strength = scheduled_correction_strength(
                base_strength=correction_strength * correction_latent_blend,
                schedule=correction_schedule,
                step_index=step_index,
                total_steps=len(timesteps),
                start_step=correction_start_step,
                end_step=correction_end_step,
            )
            with autocast_context:
                guided_latents = apply_dps_guidance_to_latents(
                    vae=vae,
                    scheduler=scheduler,
                    current_latents=latents,
                    noise_pred=noise_pred,
                    timestep=timestep,
                    source_image_linear=source_image_linear,
                    mask_onehot=mask_onehot.float(),
                    guidance_scale=effective_strength,
                    guidance_iters=correction_iters,
                    guidance_lowpass_downsample=guidance_lowpass_downsample,
                    guidance_grad_clip_rms=guidance_grad_clip_rms,
                    guidance_prior_weight=guidance_prior_weight,
                    guidance_decode_batch_size=guidance_decode_batch_size,
                    residual_chroma=residual_chroma,
                )

        with torch.no_grad():
            step_output = scheduler.step(noise_pred, timestep, guided_latents, generator=generator)
            latents = step_output.prev_sample.detach()

    with torch.no_grad(), autocast_context:
        final_maps = decode_latents_to_unit_maps_chunked(
            vae,
            latents,
            decode_batch_size=max(1, guidance_decode_batch_size),
        )

    if final_projection and final_projection_strength > 0.0:
        final_maps = project_intrinsic_components_to_input(
            component_maps=final_maps,
            source_image_linear=source_image_linear,
            mask_onehot=mask_onehot.float(),
            strength=final_projection_strength,
            num_iters=max(correction_iters, 1),
            component_mobility=component_mobility,
            residual_chroma=residual_chroma,
            max_delta=correction_max_delta,
            projection_mode=projection_mode,
        )
    final_maps = enforce_sky_only_on_unit_maps(
        component_maps=final_maps,
        source_image_linear=source_image_linear,
        mask_onehot=mask_onehot.float(),
    )

    outputs = {
        prompt: final_maps[PROMPT_TO_INDEX[prompt]].detach().float().cpu()
        for prompt in TARGET_PROMPTS
    }
    return outputs, source_image_linear[0].detach().float().cpu()


def output_to_linear_components(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        prompt: component_space_to_linear(outputs[prompt], prompt)
        for prompt in TARGET_PROMPTS
    }


@torch.no_grad()
def generate_intrinsic_map(
    *,
    prompt: str,
    image: Image.Image,
    water_mask: Image.Image,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    resolution: int,
    num_inference_steps: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    generator: Optional[torch.Generator],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Image.Image:
    output = generate_intrinsic_tensor(
        prompt=prompt,
        image=image,
        water_mask=water_mask,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        device=device,
        weight_dtype=weight_dtype,
        generator=generator,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return TF.to_pil_image(output)
