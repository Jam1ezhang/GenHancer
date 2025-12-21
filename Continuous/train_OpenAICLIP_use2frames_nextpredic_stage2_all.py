import argparse
import logging
import math
import os
import re
import random
import shutil
import time
from pathlib import Path
from contextlib import nullcontext
from copy import deepcopy

from einops import repeat, rearrange
from safetensors.torch import save_file

# ----------------------------
# NCCL / distributed env
# ----------------------------
os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")
os.environ.setdefault("NCCL_TOPO_FILE", "")

os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "7200000")

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from torchvision import transforms

from src.flux.util import load_ae, load_flow_model2
from image_datasets.dataset_video import loader

from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip

from peft import LoraConfig, get_peft_model

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD  = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD  = 0.5

NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE  = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


class VisualPromptAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class SuperModel(nn.Module):
    def __init__(self, clip_vis, dit):
        super().__init__()
        self.clip_vis = clip_vis
        self.dit = dit
        self.visual_adapter = VisualPromptAdapter(in_dim=1024, out_dim=4096)

    def get_clip_vis(self):
        return self.clip_vis

    def get_dit(self):
        return self.dit


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 training: use frame1+frame2 -> pred frame3")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parsed_args = parser.parse_args()
    return parsed_args.config


def create_spatio_temporal_ids(h, w, time_step, device):
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()
    flat_t = torch.full_like(flat_h, fill_value=time_step)
    ids = torch.stack([flat_t, flat_h, flat_w], dim=1)  # [h*w, 3]
    return ids


def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)

    is_schnell = args.model_name == "flux-schnell"
    args.clip_config.seq_t5 = 256 if is_schnell else 512

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

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

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------
    # Load models
    # ----------------------------
    dit = load_flow_model2(args.model_name, device="cpu")
    vae = load_ae(args.model_name, device=accelerator.device)
    clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)

    # contiguous projections for OpenAICLIP-336px
    if args.clip_config.clip_image_size == 336:
        clip_vis.model.visual_projection.weight = torch.nn.Parameter(clip_vis.model.visual_projection.weight.contiguous())
        clip_vis.model.text_projection.weight   = torch.nn.Parameter(clip_vis.model.text_projection.weight.contiguous())

    # ----------------------------
    # LoRA for CLIP (stage2)
    # ----------------------------
    lora_config = LoraConfig(
        r=args.lora_config.r,
        lora_alpha=args.lora_config.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_config.lora_dropout,
        bias=args.lora_config.bias,
    )
    clip_vis.model = get_peft_model(clip_vis.model, lora_config)
    clip_vis.model.print_trainable_parameters()

    # super model = clip_vis + dit + visual_adapter
    super_model = SuperModel(clip_vis, dit)

    # ----------------------------
    # Load stage1 checkpoints (project_clip, visual_adapter, dit)
    # ----------------------------
    if hasattr(args, "load_dir") and hasattr(args, "load_step"):
        print("Loading projection params from stage1...")
        load_path_project_clip = os.path.join(args.load_dir, f"checkpoint-project-clip-{args.load_step}.bin")
        clip_vis.project_clip.load_state_dict(torch.load(load_path_project_clip, map_location="cpu"))

        load_path_visual_adapter = os.path.join(args.load_dir, f"checkpoint-visual-adapter-{args.load_step}.bin")
        super_model.visual_adapter.load_state_dict(torch.load(load_path_visual_adapter, map_location="cpu"))
        print("Loading projection params successfully!")

        print("Loading dit params from stage1...")
        load_path_dit = os.path.join(args.load_dir, f"checkpoint-dit-{args.load_step}.bin")
        dit.load_state_dict(torch.load(load_path_dit, map_location="cpu"))
        print("Loading dit params successfully!")

    # ----------------------------
    # Trainability setup
    # ----------------------------
    vae.requires_grad_(False)

    dit.requires_grad_(True)
    dit = dit.to(torch.bfloat16).to(accelerator.device)
    dit.train()

    # CLIP with LoRA: must be train() and MUST allow grads through vision_model forward
    clip_vis.train()

    # adapter must be trainable
    super_model.visual_adapter.requires_grad_(True)

    # Collect params to optimize
    params_to_optimize = [p for p in super_model.parameters() if p.requires_grad]

    total_params = sum(p.numel() for p in super_model.parameters())
    trainable_params = sum(p.numel() for p in params_to_optimize)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)

    # Scheduler and steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(int(3e6) / args.data_config.train_batch_size) / args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    global_step = 0
    first_epoch = 0

    super_model, optimizer, _, lr_scheduler = accelerator.prepare(
        super_model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # resume
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            ckpt_dirs = []
            for d in dirs:
                m = re.fullmatch(r"checkpoint-(\d+)", d)
                if m is not None and os.path.isdir(os.path.join(args.output_dir, d)):
                    ckpt_dirs.append((int(m.group(1)), d))
            path = max(ckpt_dirs, key=lambda x: x[0])[1] if ckpt_dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # monitoring
    last_data_load_time = time.time()
    data_load_timeout = 60.0
    step_timeout = 300.0

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            try:
                # data load monitor
                current_time = time.time()
                data_load_duration = current_time - last_data_load_time
                if data_load_duration > data_load_timeout:
                    logger.warning(
                        f"[Rank {accelerator.process_index}] Data loading timeout: "
                        f"step={step}, duration={data_load_duration:.2f}s > {data_load_timeout}s"
                    )
                last_data_load_time = current_time

                # sanity: required keys
                if (not batch) or (len(batch) == 0):
                    logger.warning(f"[Rank {accelerator.process_index}] Empty batch at step {step}, skipping")
                    continue
                if ("start_frame" not in batch) or ("middle_frame" not in batch) or ("end_frame" not in batch):
                    logger.warning(f"[Rank {accelerator.process_index}] Missing required fields in batch at step {step}")
                    continue

                step_start_time = time.time()

                with accelerator.accumulate(super_model):
                    # -------------------------------------------------
                    # Stage2 target: use frame1+frame2 -> predict frame3
                    # -------------------------------------------------
                    cond0  = batch["start_frame"].to(accelerator.device)   # frame1
                    cond1  = batch["middle_frame"].to(accelerator.device)  # frame2
                    target = batch["end_frame"].to(accelerator.device)     # frame3 (GT)

                    # VAE encode target (no grad)
                    with torch.no_grad():
                        x_1 = vae.encode(NORMALIZE_VAE(target).to(torch.float32))

                    # CLIP normalize
                    img0 = NORMALIZE_CLIP(cond0).to(weight_dtype)
                    img1 = NORMALIZE_CLIP(cond1).to(weight_dtype)

                    # IMPORTANT: stage2 must allow grads through CLIP vision_model (LoRA needs this)
                    out0 = super_model.clip_vis.model.vision_model(img0, output_hidden_states=True)
                    out1 = super_model.clip_vis.model.vision_model(img1, output_hidden_states=True)

                    patches0 = out0.last_hidden_state[:, 1:, :]  # [B, L, 1024]
                    patches1 = out1.last_hidden_state[:, 1:, :]  # [B, L, 1024]

                    vec0 = super_model.clip_vis.model.visual_projection(out0.pooler_output)
                    vec1 = super_model.clip_vis.model.visual_projection(out1.pooler_output)
                    vec_fused = (vec0 + vec1) / 2

                    # concat two frames patches
                    visual_context_raw = torch.cat([patches0, patches1], dim=1)  # [B, 2L, 1024]

                    # adapter to text space
                    txt_replacement = super_model.visual_adapter(visual_context_raw)  # [B, 2L, 4096]

                    # txt_ids for rope: time_step 0/1
                    L = patches0.shape[1]
                    side = int(L ** 0.5)
                    assert side * side == L, f"CLIP patch tokens not square: L={L}"

                    ids0 = create_spatio_temporal_ids(side, side, time_step=0, device=accelerator.device)  # frame1
                    ids1 = create_spatio_temporal_ids(side, side, time_step=1, device=accelerator.device)  # frame2
                    ids_cat = torch.cat([ids0, ids1], dim=0)  # [2L, 3]

                    bs = target.shape[0]
                    txt_ids = repeat(ids_cat, "l d -> b l d", b=bs).to(weight_dtype)

                    assert txt_replacement.shape[1] == txt_ids.shape[1], \
                        f"txt len mismatch: txt={txt_replacement.shape[1]} vs txt_ids={txt_ids.shape[1]}"

                    # img_ids for target image tokens
                    dummy_out = prepare_clip(
                        super_model.clip_vis,
                        img0,                    # any correct-shaped clip-normalized img is ok
                        x_1.to(weight_dtype),
                    )
                    target_img_ids = dummy_out["img_ids"].to(weight_dtype)

                    # mark target frame time_step=2 (third frame)
                    target_img_ids[..., 0] = 2.0

                    inp = {
                        "img_ids": target_img_ids,
                        "txt": txt_replacement,
                        "txt_ids": txt_ids,
                        "vec": vec_fused,
                    }

                    # diffusion/flow training
                    x_1_tokens = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                    t = torch.sigmoid(torch.randn((bs,), device=accelerator.device) * args.scale_factor)
                    x_0 = torch.randn_like(x_1_tokens).to(accelerator.device)
                    x_t = (1 - t[:, None, None]) * x_1_tokens + t[:, None, None] * x_0

                    guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                    model_pred = super_model.dit(
                        img=x_t.to(weight_dtype),
                        img_ids=inp["img_ids"].to(weight_dtype),
                        txt=inp["txt"].to(weight_dtype),
                        txt_ids=inp["txt_ids"].to(weight_dtype),
                        y=inp["vec"].to(weight_dtype),
                        timesteps=t.to(weight_dtype),
                        guidance=guidance_vec.to(weight_dtype),
                    )

                    loss = F.mse_loss(model_pred.float(), (x_0 - x_1_tokens).float(), reduction="mean")

                    train_loss += loss.detach().item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # step monitor
                step_duration = time.time() - step_start_time
                if step_duration > step_timeout:
                    logger.error(
                        f"[Rank {accelerator.process_index}] Step timeout: "
                        f"step={step}, duration={step_duration:.2f}s > {step_timeout}s"
                    )

                if accelerator.sync_gradients:
                    if global_step % 10 == 0:
                        logger.info(
                            f"[Rank {accelerator.process_index}] Step {global_step}: "
                            f"loss={train_loss:.4f}, step_time={step_duration:.3f}s, data_load_time={data_load_duration:.3f}s"
                        )

                    progress_bar.update(1)
                    global_step += 1

                    accelerator.log(
                        {"train_loss": train_loss, "step_time": step_duration, "data_load_time": data_load_duration},
                        step=global_step,
                    )
                    train_loss = 0.0

                    # checkpoint
                    if (global_step % args.checkpointing_steps == 0) or (global_step in [50, 100, 200, 300, 500, 1000, 2000, 3000]):
                        if accelerator.is_main_process:
                            unwrapped_super_model = accelerator.unwrap_model(super_model)

                            # save DiT
                            save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                            torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)

                            # save CLIP (LoRA merged)
                            if args.clip_config.clip_image_size == 336:
                                save_path_clip = os.path.join(args.output_dir, f"clip-vit-large-patch14-336-{global_step}")
                            else:
                                save_path_clip = os.path.join(args.output_dir, f"clip-vit-large-patch14-{global_step}")

                            save_model = deepcopy(unwrapped_super_model.clip_vis.model).merge_and_unload()
                            save_model.save_pretrained(save_path_clip, safe_serialization=False)

                            # save adapters / projection
                            save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                            save_path_visual_adapter = os.path.join(args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin")
                            torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                            torch.save(deepcopy(unwrapped_super_model.visual_adapter).state_dict(), save_path_visual_adapter)

                            # save optimizer
                            save_path_optimizer = os.path.join(args.output_dir, f"optimizer-state-{global_step}.bin")
                            torch.save(optimizer.state_dict(), save_path_optimizer)

                            logger.info(f"Saved checkpoint at step {global_step}")
                            logger.info(f"  - DiT: {save_path_dit}")
                            logger.info(f"  - CLIP (LoRA merged): {save_path_clip}")
                            logger.info(f"  - Visual Adapter: {save_path_visual_adapter}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        unwrapped_super_model = accelerator.unwrap_model(super_model)

                        save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)

                        if args.clip_config.clip_image_size == 336:
                            save_path_clip = os.path.join(args.output_dir, f"clip-vit-large-patch14-336-{global_step}")
                        else:
                            save_path_clip = os.path.join(args.output_dir, f"clip-vit-large-patch14-{global_step}")
                        save_model = deepcopy(unwrapped_super_model.clip_vis.model).merge_and_unload()
                        save_model.save_pretrained(save_path_clip, safe_serialization=False)

                        save_path_visual_adapter = os.path.join(args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.visual_adapter).state_dict(), save_path_visual_adapter)

                        logger.info(f"Final checkpoint saved at step {global_step}")
                    break

            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f"RuntimeError at step {step}: {error_msg}", exc_info=True)
                if ("NCCL" in error_msg) or ("timeout" in error_msg.lower()) or ("distributed" in error_msg.lower()):
                    logger.error("Distributed training error detected. Stopping training.")
                    break
                raise
            except Exception as e:
                logger.error(f"Unexpected error at step {step}: {type(e).__name__}: {e}", exc_info=True)
                raise

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
