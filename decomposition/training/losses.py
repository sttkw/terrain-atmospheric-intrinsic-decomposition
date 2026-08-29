import argparse
from typing import Dict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from pipeline import (
    TARGET_PROMPTS,
    component_space_to_linear,
    compose_intrinsic_hdr,
    select_mask_conditioning_channels,
    to_unit_range,
    tokenize_prompts,
)

PROMPT_TO_INDEX = {prompt_name: index for index, prompt_name in enumerate(TARGET_PROMPTS)}


def _sqrt_alpha_timestep_weights(
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=timesteps.device,
        dtype=torch.float32,
    )
    return torch.sqrt(alphas_cumprod[timesteps].clamp(min=1e-6, max=1.0))


def _masked_mean_per_scene(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted_sum = (values * mask).sum(dim=(1, 2, 3))
    denominator = (mask.sum(dim=(1, 2, 3)) * values.shape[1]).clamp_min(1.0)
    return weighted_sum / denominator


def _water_gradient_loss_per_scene(
    image: torch.Tensor,
    water_mask: torch.Tensor,
) -> torch.Tensor:
    grad_x = (image[..., :, 1:] - image[..., :, :-1]).abs()
    grad_y = (image[..., 1:, :] - image[..., :-1, :]).abs()
    mask_x = water_mask[..., :, 1:] * water_mask[..., :, :-1]
    mask_y = water_mask[..., 1:, :] * water_mask[..., :-1, :]
    loss_x = _masked_mean_per_scene(grad_x, mask_x)
    loss_y = _masked_mean_per_scene(grad_y, mask_y)
    return 0.5 * (loss_x + loss_y)


def _water_terrain_boundary_loss_per_scene(
    image: torch.Tensor,
    water_mask: torch.Tensor,
    terrain_mask: torch.Tensor,
) -> torch.Tensor:
    diff_x = (image[..., :, 1:] - image[..., :, :-1]).abs()
    diff_y = (image[..., 1:, :] - image[..., :-1, :]).abs()
    mask_x = (
        water_mask[..., :, 1:] * terrain_mask[..., :, :-1]
        + terrain_mask[..., :, 1:] * water_mask[..., :, :-1]
    )
    mask_y = (
        water_mask[..., 1:, :] * terrain_mask[..., :-1, :]
        + terrain_mask[..., 1:, :] * water_mask[..., :-1, :]
    )
    weighted_sum = (diff_x * mask_x).sum(dim=(1, 2, 3)) + (diff_y * mask_y).sum(
        dim=(1, 2, 3)
    )
    denominator = (
        (mask_x.sum(dim=(1, 2, 3)) + mask_y.sum(dim=(1, 2, 3))) * image.shape[1]
    ).clamp_min(1.0)
    return weighted_sum / denominator


def _predict_x0_from_model_prediction(
    noise_scheduler: DDPMScheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).clamp(min=1e-6, max=1.0)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=0.0))

    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (noisy_latents - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
    if prediction_type == "v_prediction":
        return sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * model_pred
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def _decode_prediction_to_map(
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    decode_batch_size: int,
) -> torch.Tensor:
    x0_latents = _predict_x0_from_model_prediction(
        noise_scheduler=noise_scheduler,
        noisy_latents=noisy_latents,
        model_pred=model_pred,
        timesteps=timesteps,
    )
    vae_dtype = next(vae.parameters()).dtype
    latents_for_decode = (x0_latents / vae.config.scaling_factor).to(dtype=vae_dtype)
    if decode_batch_size <= 0 or decode_batch_size >= latents_for_decode.shape[0]:
        decoded = vae.decode(latents_for_decode).sample
    else:
        decoded_chunks = []
        for latent_chunk in latents_for_decode.split(decode_batch_size, dim=0):
            decoded_chunks.append(vae.decode(latent_chunk).sample)
        decoded = torch.cat(decoded_chunks, dim=0)
    return to_unit_range(decoded.float())


@torch.no_grad()
def build_fixed_prompt_hidden_states(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    hidden_states_by_prompt: Dict[str, torch.Tensor] = {}
    for prompt_name in TARGET_PROMPTS:
        prompt_ids = tokenize_prompts(tokenizer, [prompt_name]).to(device)
        hidden_states_by_prompt[prompt_name] = text_encoder(prompt_ids)[0].detach()
    return hidden_states_by_prompt


def _expand_prompt_hidden_states_for_scene_batch(
    fixed_prompt_hidden_states: Dict[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    stacked_prompt_states = torch.stack(
        [fixed_prompt_hidden_states[prompt_name].squeeze(0) for prompt_name in TARGET_PROMPTS],
        dim=0,
    )
    return stacked_prompt_states.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
        batch_size * len(TARGET_PROMPTS),
        stacked_prompt_states.shape[1],
        stacked_prompt_states.shape[2],
    )


def compute_training_losses(
    *,
    batch: Dict[str, torch.Tensor],
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    args: argparse.Namespace,
    fixed_prompt_hidden_states: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    target_component_pixel_values = batch["target_component_pixel_values"].to(dtype=weight_dtype)
    original_pixel_values = batch["original_pixel_values"].to(dtype=weight_dtype)
    mask_pixel_values = batch["water_mask_pixel_values"].to(dtype=weight_dtype)
    batch_size = original_pixel_values.shape[0]

    flat_target_pixel_values = target_component_pixel_values.reshape(
        batch_size * len(TARGET_PROMPTS),
        target_component_pixel_values.shape[2],
        target_component_pixel_values.shape[3],
        target_component_pixel_values.shape[4],
    )

    latents = vae.encode(flat_target_pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size * len(TARGET_PROMPTS),),
        device=latents.device,
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    original_image_latents = vae.encode(original_pixel_values).latent_dist.mode()
    original_image_latents = original_image_latents.unsqueeze(1).expand(
        -1,
        len(TARGET_PROMPTS),
        -1,
        -1,
        -1,
    )
    original_image_latents = original_image_latents.reshape(
        batch_size * len(TARGET_PROMPTS),
        original_image_latents.shape[2],
        original_image_latents.shape[3],
        original_image_latents.shape[4],
    )
    mask_condition_values = select_mask_conditioning_channels(mask_pixel_values)
    mask_condition_latents = F.interpolate(
        mask_condition_values,
        size=original_image_latents.shape[-2:],
        mode="nearest",
    )
    mask_condition_latents = mask_condition_latents.unsqueeze(1).expand(
        -1,
        len(TARGET_PROMPTS),
        -1,
        -1,
        -1,
    )
    mask_condition_latents = mask_condition_latents.reshape(
        batch_size * len(TARGET_PROMPTS),
        mask_condition_latents.shape[2],
        mask_condition_latents.shape[3],
        mask_condition_latents.shape[4],
    )
    prompt_hidden_states = _expand_prompt_hidden_states_for_scene_batch(
        fixed_prompt_hidden_states=fixed_prompt_hidden_states,
        batch_size=batch_size,
    )

    model_input = torch.cat(
        [noisy_latents, original_image_latents, mask_condition_latents],
        dim=1,
    )
    model_pred = unet(
        model_input,
        timesteps,
        encoder_hidden_states=prompt_hidden_states,
    ).sample

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            "Unsupported prediction type: "
            f"{noise_scheduler.config.prediction_type}"
        )

    diffusion_per_component = (model_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
    predicted_component_maps = _decode_prediction_to_map(
        vae=vae,
        noise_scheduler=noise_scheduler,
        noisy_latents=noisy_latents,
        model_pred=model_pred,
        timesteps=timesteps,
        decode_batch_size=args.vae_decode_batch_size,
    )

    predicted_component_maps = predicted_component_maps.reshape(
        batch_size,
        len(TARGET_PROMPTS),
        predicted_component_maps.shape[1],
        predicted_component_maps.shape[2],
        predicted_component_maps.shape[3],
    )
    target_component_maps = to_unit_range(target_component_pixel_values.float())

    predicted_albedo = predicted_component_maps[:, PROMPT_TO_INDEX["Albedo"]]
    predicted_diffuse_s01 = predicted_component_maps[:, PROMPT_TO_INDEX["Diffuse Shading"]]
    predicted_specular_unit = predicted_component_maps[:, PROMPT_TO_INDEX["Specular Shading"]]
    predicted_volume_unit = predicted_component_maps[:, PROMPT_TO_INDEX["Volume"]]

    predicted_diffuse_hdr = component_space_to_linear(predicted_diffuse_s01, "Diffuse Shading")
    predicted_specular_hdr = component_space_to_linear(predicted_specular_unit, "Specular Shading")
    predicted_volume_hdr = component_space_to_linear(predicted_volume_unit, "Volume")

    reconstruction_hdr = compose_intrinsic_hdr(
        albedo=predicted_albedo,
        diffuse_shading=predicted_diffuse_hdr,
        specular_shading=predicted_specular_hdr,
        volume=predicted_volume_hdr,
    )
    reconstruction = reconstruction_hdr.clamp(0.0, 1.0)
    source_image_linear = to_unit_range(original_pixel_values.float())
    timestep_weights = _sqrt_alpha_timestep_weights(
        noise_scheduler=noise_scheduler,
        timesteps=timesteps,
    ).reshape(batch_size, len(TARGET_PROMPTS))
    scene_timestep_weights = timestep_weights.mean(dim=1)
    albedo_timestep_weights = timestep_weights[:, PROMPT_TO_INDEX["Albedo"]]
    diffuse_timestep_weights = timestep_weights[:, PROMPT_TO_INDEX["Diffuse Shading"]]
    specular_timestep_weights = timestep_weights[:, PROMPT_TO_INDEX["Specular Shading"]]
    volume_timestep_weights = timestep_weights[:, PROMPT_TO_INDEX["Volume"]]
    recon_per_scene = (
        (reconstruction - source_image_linear).abs().mean(dim=(1, 2, 3))
        * scene_timestep_weights
    )

    target_albedo = target_component_maps[:, PROMPT_TO_INDEX["Albedo"]]
    target_diffuse_s01 = target_component_maps[:, PROMPT_TO_INDEX["Diffuse Shading"]]
    target_specular_unit = target_component_maps[:, PROMPT_TO_INDEX["Specular Shading"]]
    target_volume_unit = target_component_maps[:, PROMPT_TO_INDEX["Volume"]]

    direct_per_prompt = torch.stack(
        [
            (predicted_albedo - target_albedo).abs().mean(dim=(1, 2, 3)),
            (predicted_diffuse_s01 - target_diffuse_s01).abs().mean(dim=(1, 2, 3)),
            (predicted_specular_unit - target_specular_unit).abs().mean(dim=(1, 2, 3)),
            (predicted_volume_unit - target_volume_unit).abs().mean(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    direct_per_scene = (direct_per_prompt * timestep_weights).sum(dim=1)
    water_mask = mask_pixel_values[:, 0:1].float()
    terrain_mask = mask_pixel_values[:, 1:2].float()
    water_albedo_gradient = _water_gradient_loss_per_scene(
        predicted_albedo,
        water_mask,
    )
    water_diffuse_gradient = _water_gradient_loss_per_scene(
        predicted_diffuse_s01,
        water_mask,
    )
    water_volume_gradient = _water_gradient_loss_per_scene(
        predicted_volume_unit,
        water_mask,
    )
    water_volume_boundary = _water_terrain_boundary_loss_per_scene(
        predicted_volume_unit,
        water_mask,
        terrain_mask,
    )
    water_specular_direct = _masked_mean_per_scene(
        (predicted_specular_unit - target_specular_unit).abs(),
        water_mask,
    )
    water_per_scene = (
        albedo_timestep_weights * water_albedo_gradient
        + diffuse_timestep_weights * water_diffuse_gradient
        + specular_timestep_weights * water_specular_direct
        + volume_timestep_weights * (water_volume_gradient + water_volume_boundary)
    )

    diffusion_per_scene = diffusion_per_component.reshape(batch_size, len(TARGET_PROMPTS)).mean(dim=1)

    diffusion_per_prompt = diffusion_per_component.reshape(batch_size, len(TARGET_PROMPTS)).mean(dim=0)

    total_per_scene = (
        args.loss_weight_diffusion * diffusion_per_scene
        + args.loss_weight_recon * recon_per_scene
        + args.loss_weight_direct * direct_per_scene
        + args.loss_weight_water * water_per_scene
    )

    return {
        "loss": total_per_scene.mean(),
        "per_scene_loss": total_per_scene,
        "diffusion_loss": diffusion_per_scene.mean(),
        "recon_loss": recon_per_scene.mean(),
        "direct_loss": direct_per_scene.mean(),
        "water_loss": water_per_scene.mean(),
        "diffusion_loss_per_prompt": diffusion_per_prompt,
    }


@torch.no_grad()
def evaluate_validation_losses(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    val_dataloader: torch.utils.data.DataLoader,
    weight_dtype: torch.dtype,
    accelerator: Accelerator,
    args: argparse.Namespace,
    fixed_prompt_hidden_states: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    unet.eval()

    total_loss = 0.0
    total_batches = 0
    total_diffusion_loss = 0.0
    total_recon_loss = 0.0
    total_direct_loss = 0.0
    total_water_loss = 0.0
    prompt_diffusion_sums = {prompt_name: 0.0 for prompt_name in TARGET_PROMPTS}

    for batch in val_dataloader:
        loss_outputs = compute_training_losses(
            batch=batch,
            vae=vae,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            args=args,
            fixed_prompt_hidden_states=fixed_prompt_hidden_states,
        )
        loss = loss_outputs["loss"]
        per_scene_losses = loss_outputs["per_scene_loss"]
        scene_batch_size = per_scene_losses.shape[0]

        gathered_loss = accelerator.gather(loss.repeat(scene_batch_size)).mean()
        gathered_diffusion_loss = accelerator.gather(
            loss_outputs["diffusion_loss"].repeat(scene_batch_size)
        ).mean()
        gathered_recon_loss = accelerator.gather(
            loss_outputs["recon_loss"].repeat(scene_batch_size)
        ).mean()
        gathered_direct_loss = accelerator.gather(
            loss_outputs["direct_loss"].repeat(scene_batch_size)
        ).mean()
        gathered_water_loss = accelerator.gather(
            loss_outputs["water_loss"].repeat(scene_batch_size)
        ).mean()
        total_loss += gathered_loss.item()
        total_diffusion_loss += gathered_diffusion_loss.item()
        total_recon_loss += gathered_recon_loss.item()
        total_direct_loss += gathered_direct_loss.item()
        total_water_loss += gathered_water_loss.item()
        total_batches += 1

        gathered_prompt_diffusion = accelerator.gather(
            loss_outputs["diffusion_loss_per_prompt"].unsqueeze(0)
        )
        prompt_means = gathered_prompt_diffusion.float().mean(dim=0).cpu()
        for prompt_index, prompt_name in enumerate(TARGET_PROMPTS):
            prompt_diffusion_sums[prompt_name] += float(prompt_means[prompt_index].item())

    metrics = {
        "val_loss": total_loss / max(total_batches, 1),
        "val_loss_diffusion": total_diffusion_loss / max(total_batches, 1),
        "val_loss_recon": total_recon_loss / max(total_batches, 1),
        "val_loss_direct": total_direct_loss / max(total_batches, 1),
        "val_loss_water": total_water_loss / max(total_batches, 1),
    }
    for prompt_name in TARGET_PROMPTS:
        metrics[f"val_loss_diffusion_{prompt_name.lower()}"] = prompt_diffusion_sums[prompt_name] / max(
            total_batches, 1
        )
    return metrics
