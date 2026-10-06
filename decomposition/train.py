#!/usr/bin/env python
# coding=utf-8

"""Fine-tune InstructPix2Pix for intrinsic terrain decomposition.

This script follows the project specification:
- one inference produces one intrinsic property map
- prompt conditioning is restricted to four fixed keywords
- classifier-free guidance is not used for validation inference
- paired augmentations are applied consistently to source/target images
"""

import logging
import math
import os
import random

import accelerate
import datasets
import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from pipeline import PREDICTION_TYPE
from training.args import build_hub_kwargs, build_tracker_config, parse_args
from training.data import (
    SceneGroupedDataset,
    build_scene_groups,
    collate_scene_batch,
    load_training_split,
    run_self_test,
    split_scene_groups,
    validate_dataset_schema,
)
from training.losses import (
    PROMPT_TO_INDEX,
    build_fixed_prompt_hidden_states,
    compute_training_losses,
    evaluate_validation_losses,
)
from training.utils import (
    build_generator,
    expand_unet_conv_in_to_ten_channels,
    maybe_enable_xformers,
    resolve_resume_path,
    save_training_metadata,
)
from training.validation import export_validation_images, load_validation_pairs

check_min_version("0.37.0")

logger = get_logger(__name__, log_level="INFO")


def main() -> None:
    args = parse_args()

    hub_kwargs = build_hub_kwargs(args.hf_token)

    if args.self_test:
        run_self_test()
        return

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        logging_dir=logging_dir,
    )
    log_with = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use --report_to wandb.")
        import wandb  # noqa: F401

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        **hub_kwargs,
    )
    noise_scheduler.register_to_config(prediction_type=PREDICTION_TYPE)
    inference_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    inference_scheduler.register_to_config(prediction_type=PREDICTION_TYPE)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        **hub_kwargs,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        **hub_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        **hub_kwargs,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
        **hub_kwargs,
    )
    expand_unet_conv_in_to_ten_channels(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_vae_slicing:
        vae.enable_slicing()
    if args.enable_vae_tiling:
        vae.enable_tiling()

    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )
    else:
        ema_unet = None

    if args.enable_xformers_memory_efficient_attention:
        maybe_enable_xformers(unet)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if ema_unet is not None:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for model in models:
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if ema_unet is not None:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNet2DConditionModel,
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            while models:
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir,
                    subfolder="unet",
                )
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Install bitsandbytes before using --use_8bit_adam."
            ) from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset_all = load_training_split(args)
    validate_dataset_schema(
        train_dataset_all,
        args.original_image_column,
        args.edited_image_column,
        args.edit_prompt_column,
        args.water_mask_column,
    )

    scene_groups_all = build_scene_groups(
        dataset=train_dataset_all,
        original_image_column=args.original_image_column,
        prompt_column=args.edit_prompt_column,
        water_mask_column=args.water_mask_column,
        scene_id_column=args.scene_id_column,
    )

    if args.max_train_samples is not None:
        max_scene_samples = min(args.max_train_samples, len(scene_groups_all))
        scene_groups_all = random.Random(args.seed).sample(scene_groups_all, max_scene_samples)

    train_scene_groups, val_scene_groups = split_scene_groups(
        scene_groups=scene_groups_all,
        seed=args.seed,
        val_ratio=0.1,
    )

    train_dataset = SceneGroupedDataset(
        dataset=train_dataset_all,
        scene_groups=train_scene_groups,
        args=args,
        original_image_column=args.original_image_column,
        edited_image_column=args.edited_image_column,
        water_mask_column=args.water_mask_column,
    )
    val_dataset = None
    if val_scene_groups is not None and len(val_scene_groups) > 0:
        val_dataset = SceneGroupedDataset(
            dataset=train_dataset_all,
            scene_groups=val_scene_groups,
            args=args,
            original_image_column=args.original_image_column,
            edited_image_column=args.edited_image_column,
            water_mask_column=args.water_mask_column,
        )

    dataloader_common_kwargs = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_num_workers > 0,
        "persistent_workers": args.dataloader_num_workers > 0,
    }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_scene_batch,
        batch_size=args.train_batch_size,
        **dataloader_common_kwargs,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=collate_scene_batch,
            batch_size=args.train_batch_size,
            **dataloader_common_kwargs,
        )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if val_dataloader is not None:
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            unet,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    if ema_unet is not None:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    fixed_prompt_hidden_states = build_fixed_prompt_hidden_states(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=accelerator.device,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "intrinsic-terrain-decomposition",
            config=build_tracker_config(args),
        )
        save_training_metadata(
            args.output_dir,
            args,
            num_train_scenes=len(train_scene_groups),
            num_val_scenes=0 if val_scene_groups is None else len(val_scene_groups),
        )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num scenes = {len(train_scene_groups)}")
    logger.info(
        "  Num validation scenes = "
        f"{0 if val_scene_groups is None else len(val_scene_groups)}"
    )
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        "  Total train batch size (parallel, distributed, accumulation) = "
        f"{total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        resume_path = resolve_resume_path(args.output_dir, args.resume_from_checkpoint)
        if resume_path is None:
            raise FileNotFoundError(
                f"Checkpoint '{args.resume_from_checkpoint}' was not found in {args.output_dir}."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_path}")
            # Avoid accelerator.load_state to prevent restoring sampler/dataloader/random
            # states which can cause heavy seeks on sharded parquet datasets.
            ckpt_dir = os.path.join(args.output_dir, resume_path)

            unet_path = os.path.join(ckpt_dir, "unet")
            if not os.path.isdir(unet_path):
                raise FileNotFoundError(f"UNet checkpoint not found at {unet_path}")

            restored_unet = UNet2DConditionModel.from_pretrained(unet_path)
            accelerator.unwrap_model(unet).load_state_dict(restored_unet.state_dict())
            del restored_unet
            accelerator.print("Loaded UNet weights from checkpoint.")

            opt_path = os.path.join(ckpt_dir, "optimizer.bin")
            sch_path = os.path.join(ckpt_dir, "scheduler.bin")
            if os.path.isfile(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(opt_state)
                accelerator.print("Restored optimizer state from checkpoint.")
            if os.path.isfile(sch_path):
                sch_state = torch.load(sch_path, map_location="cpu", weights_only=True)
                lr_scheduler.load_state_dict(sch_state)
                accelerator.print("Restored scheduler state from checkpoint.")

            global_step = int(resume_path.split("-")[-1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    generator = build_generator(accelerator.device, args.seed)
    validation_pairs = load_validation_pairs(args, logger)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
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

                avg_loss = accelerator.gather(loss.repeat(scene_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                # clip_grad_norm_ with max_norm=0 scales every gradient to zero,
                # so 0 has to mean "no clipping" rather than being passed through.
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if ema_unet is not None:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "train_loss_diffusion": loss_outputs["diffusion_loss"].detach().item(),
                        "train_loss_recon": loss_outputs["recon_loss"].detach().item(),
                        "train_loss_direct": loss_outputs["direct_loss"].detach().item(),
                        "train_loss_water": loss_outputs["water_loss"].detach().item(),
                        "train_loss_diffusion_albedo": loss_outputs["diffusion_loss_per_prompt"][PROMPT_TO_INDEX["Albedo"]].detach().item(),
                        "train_loss_diffusion_diffuse_shading": loss_outputs["diffusion_loss_per_prompt"][PROMPT_TO_INDEX["Diffuse Shading"]].detach().item(),
                        "train_loss_diffusion_specular_shading": loss_outputs["diffusion_loss_per_prompt"][PROMPT_TO_INDEX["Specular Shading"]].detach().item(),
                        "train_loss_diffusion_volume": loss_outputs["diffusion_loss_per_prompt"][PROMPT_TO_INDEX["Volume"]].detach().item(),
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    should_run_validation_images = len(validation_pairs) > 0
                    if should_run_validation_images:
                        logger.info("Running validation export for mixed component-space outputs and reconstruction.")

                        validation_unet = accelerator.unwrap_model(unet)
                        if ema_unet is not None:
                            ema_unet.store(validation_unet.parameters())
                            ema_unet.copy_to(validation_unet.parameters())

                        export_validation_images(
                            args=args,
                            validation_pairs=validation_pairs,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vae=vae,
                            unet=validation_unet,
                            scheduler=inference_scheduler,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            generator=generator,
                            global_step=global_step,
                        )

                        if ema_unet is not None:
                            ema_unet.restore(validation_unet.parameters())

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if val_dataloader is not None:
                        validation_metrics = evaluate_validation_losses(
                            unet=unet,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            val_dataloader=val_dataloader,
                            weight_dtype=weight_dtype,
                            accelerator=accelerator,
                            args=args,
                            fixed_prompt_hidden_states=fixed_prompt_hidden_states,
                        )
                        accelerator.log(validation_metrics, step=global_step)

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_unet = accelerator.unwrap_model(unet)
        if ema_unet is not None:
            ema_unet.copy_to(final_unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=final_unet,
            scheduler=noise_scheduler,
            revision=args.revision,
            **hub_kwargs,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
