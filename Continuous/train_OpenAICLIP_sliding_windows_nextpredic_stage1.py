#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/data/zby/GenHancer/Continuous/train_OpenAICLIP_sliding_windows_nextpredic_stage1.py

Stage1 约束（按你最新要求）：
- clip_vis 完全 frozen（不训练，不回传梯度）
- 训练 dit + visual_adapter
- 数据：full_frames（batch['frames'] + batch['frame_mask']）
- 滑动窗口：给定 (frame_t, frame_{t+1}, frame_{t+2}) 作为 cond，预测 frame_{t+3}
- 必须使用 frame_mask 裁掉 padding，否则滑窗会把 0-padding 当真帧训脏
- 控制 effective batch：每个视频最多采样 K 个窗口（避免 bs_eff 爆炸）

建议你在 yaml 里加：
data_config:
  return_mode: full_frames
  max_frames_per_video: 200
以及可选：
window_cond: 3
window_stride: 1
max_windows_per_video: 8
"""

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

from einops import rearrange, repeat
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# --------------------------
# NCCL / distributed env
# --------------------------
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

# --------------------------
# Imports
# --------------------------
import datasets
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from torchvision import transforms

from src.flux.util import load_ae, load_flow_model2
from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip

# 你原项目里的 dataloader
from image_datasets.dataset_video_sliding_window import loader  # noqa

if is_wandb_available():
    import wandb  # noqa

logger = get_logger(__name__, log_level="INFO")

# --------------------------
# Normalization
# --------------------------
OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


# --------------------------
# Models
# --------------------------
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
    """
    Stage1: clip_vis frozen, only for feature extraction
    Trainable: dit + visual_adapter
    """
    def __init__(self, clip_vis, dit, in_dim=1024, out_dim=4096):
        super().__init__()
        self.clip_vis = clip_vis
        self.dit = dit
        self.visual_adapter = VisualPromptAdapter(in_dim=in_dim, out_dim=out_dim)


# --------------------------
# Utils
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sliding-window next-frame prediction training (stage1).")
    parser.add_argument("--config", type=str, required=True, help="path to config (yaml)")
    parsed_args = parser.parse_args()
    return parsed_args.config


def create_spatio_temporal_ids(h, w, time_step, device):
    """
    生成 [h*w, 3] : [time_step, row_idx, col_idx]
    """
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()
    flat_t = torch.full_like(flat_h, fill_value=time_step)
    ids = torch.stack([flat_t, flat_h, flat_w], dim=1)
    return ids


def build_windows_with_mask(
    frames: torch.Tensor,
    frame_mask: torch.Tensor,
    window_cond: int = 3,
    window_stride: int = 1,
    max_windows_per_video: int = 8,
):
    """
    frames: [B,T,C,H,W] (padded)
    frame_mask: [B,T] bool (True means valid frame)
    Returns:
      cond0,cond1,cond2: [bs_eff,C,H,W]
      target: [bs_eff,C,H,W]
      avg_nw: 平均每视频窗口数（float）
      bs_eff: 有效窗口总数
    """
    assert frames.ndim == 5, f"expect [B,T,C,H,W], got {frames.shape}"
    assert frame_mask.ndim == 2, f"expect [B,T], got {frame_mask.shape}"

    B, T, C, H, W = frames.shape
    cond0_all, cond1_all, cond2_all, target_all = [], [], [], []
    nw_list = []

    for i in range(B):
        Ti = int(frame_mask[i].sum().item())
        if Ti < window_cond + 1:
            continue

        f = frames[i, :Ti]  # [Ti,C,H,W]
        starts = list(range(0, Ti - window_cond, window_stride))
        if len(starts) <= 0:
            continue

        if max_windows_per_video is not None and max_windows_per_video > 0 and len(starts) > max_windows_per_video:
            starts = random.sample(starts, k=max_windows_per_video)
            starts.sort()

        nw_i = len(starts)
        nw_list.append(nw_i)

        cond0_all.append(torch.stack([f[s + 0] for s in starts], dim=0))
        cond1_all.append(torch.stack([f[s + 1] for s in starts], dim=0))
        cond2_all.append(torch.stack([f[s + 2] for s in starts], dim=0))
        target_all.append(torch.stack([f[s + window_cond] for s in starts], dim=0))

    if len(target_all) == 0:
        return None

    cond0 = torch.cat(cond0_all, dim=0)
    cond1 = torch.cat(cond1_all, dim=0)
    cond2 = torch.cat(cond2_all, dim=0)
    target = torch.cat(target_all, dim=0)

    avg_nw = float(sum(nw_list)) / float(max(1, len(nw_list)))
    bs_eff = int(target.shape[0])
    return cond0, cond1, cond2, target, avg_nw, bs_eff


# --------------------------
# Main
# --------------------------
def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)

    is_schnell = args.model_name == "flux-schnell"
    args.clip_config.seq_t5 = 256 if is_schnell else 512  # 保持你原逻辑

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

    # --------------------------
    # Load models
    # --------------------------
    dit = load_flow_model2(args.model_name, device="cpu")
    vae = load_ae(args.model_name, device=accelerator.device)
    clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)

    # contiguous projections for OpenAICLIP-336px
    if getattr(args.clip_config, "clip_image_size", None) == 336:
        clip_vis.model.visual_projection.weight = torch.nn.Parameter(
            clip_vis.model.visual_projection.weight.contiguous()
        )
        clip_vis.model.text_projection.weight = torch.nn.Parameter(
            clip_vis.model.text_projection.weight.contiguous()
        )

    # Stage1: clip_vis frozen
    clip_vis.eval()
    clip_vis.requires_grad_(False)

    # Train: dit + adapter
    vae.requires_grad_(False)
    dit.requires_grad_(True)
    dit = dit.to(torch.bfloat16).to(accelerator.device)
    dit.train()

    # Super model
    # 注意：in_dim 取决于你 CLIP patch dim；你之前用的是 1024（CLIP-L）
    super_model = SuperModel(clip_vis, dit, in_dim=1024, out_dim=4096)
    super_model.visual_adapter.requires_grad_(True)

    params_to_optimize = [p for p in super_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --------------------------
    # Data
    # --------------------------
    train_dataloader = loader(**args.data_config)

    # steps math（保持你原写法）
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

    # IMPORTANT: prepare
    super_model, optimizer, _, lr_scheduler = accelerator.prepare(
        super_model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    # weight dtype
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

    # resume（保持你原逻辑：accelerate state ckpt）
    initial_global_step = 0
    if getattr(args, "resume_from_checkpoint", None):
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            ckpt_dirs = []
            import re as _re
            for d in dirs:
                m = _re.fullmatch(r"checkpoint-(\d+)", d)
                if m is not None:
                    ckpt_dirs.append((int(m.group(1)), d))
            ckpt_dirs.sort(key=lambda x: x[0])
            path = ckpt_dirs[-1][1] if len(ckpt_dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

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

    # sliding window settings
    window_cond = int(getattr(args, "window_cond", 3))
    window_stride = int(getattr(args, "window_stride", 1))
    max_windows_per_video = int(getattr(args, "max_windows_per_video", 8))

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

                if not batch or len(batch) == 0:
                    logger.warning(f"[Rank {accelerator.process_index}] Empty batch at step {step}, skipping")
                    continue

                step_start_time = time.time()

                with accelerator.accumulate(super_model):

                    # ---------------------------------------------------------
                    # 1) build sliding windows (mask-aware)
                    # ---------------------------------------------------------
                    if "frames" not in batch or "frame_mask" not in batch:
                        raise KeyError(
                            "Sliding-window training requires batch['frames'] and batch['frame_mask'] "
                            "(from loader(return_mode=full_frames))."
                        )

                    frames = batch["frames"].to(accelerator.device)        # [B,T,C,H,W] padded
                    frame_mask = batch["frame_mask"].to(accelerator.device)  # [B,T] bool

                    win = build_windows_with_mask(
                        frames=frames,
                        frame_mask=frame_mask,
                        window_cond=window_cond,
                        window_stride=window_stride,
                        max_windows_per_video=max_windows_per_video,
                    )
                    if win is None:
                        logger.warning("No valid windows in this batch after mask; skipping.")
                        continue

                    cond0, cond1, cond2, target, avg_nw, bs_eff = win

                    # ---------------------------------------------------------
                    # 2) VAE encode target (GT)  (frozen)
                    # ---------------------------------------------------------
                    with torch.no_grad():
                        x_1 = vae.encode(NORMALIZE_VAE(target).to(torch.float32))

                    # ---------------------------------------------------------
                    # 3) CLIP feature extraction (frozen, stage1)
                    # ---------------------------------------------------------
                    img0 = NORMALIZE_CLIP(cond0).to(weight_dtype)
                    img1 = NORMALIZE_CLIP(cond1).to(weight_dtype)
                    img2 = NORMALIZE_CLIP(cond2).to(weight_dtype)

                    with torch.no_grad():
                        out0 = super_model.clip_vis.model.vision_model(img0, output_hidden_states=True)
                        out1 = super_model.clip_vis.model.vision_model(img1, output_hidden_states=True)
                        out2 = super_model.clip_vis.model.vision_model(img2, output_hidden_states=True)

                        patches0 = out0.last_hidden_state[:, 1:, :]  # [B*, L, 1024]
                        patches1 = out1.last_hidden_state[:, 1:, :]
                        patches2 = out2.last_hidden_state[:, 1:, :]

                        # 全局 vec（stage1 frozen）
                        vec0 = super_model.clip_vis.model.visual_projection(out0.pooler_output)
                        vec1 = super_model.clip_vis.model.visual_projection(out1.pooler_output)
                        vec2 = super_model.clip_vis.model.visual_projection(out2.pooler_output)
                        vec_fused = (vec0 + vec1 + vec2) / 3.0  # [B*, D]

                    # cond 的视觉 token = 三帧 patch concat -> adapter（可训练）
                    visual_context_raw = torch.cat([patches0, patches1, patches2], dim=1)  # [B*, 3L, 1024]
                    txt_replacement = super_model.visual_adapter(visual_context_raw)       # [B*, 3L, 4096]

                    # ---------------------------------------------------------
                    # 4) txt_ids: time_step=0/1/2
                    # ---------------------------------------------------------
                    L = patches0.shape[1]
                    side = int(L ** 0.5)
                    assert side * side == L, f"CLIP patch tokens not square: L={L}"

                    ids0 = create_spatio_temporal_ids(side, side, time_step=0, device=accelerator.device)
                    ids1 = create_spatio_temporal_ids(side, side, time_step=1, device=accelerator.device)
                    ids2 = create_spatio_temporal_ids(side, side, time_step=2, device=accelerator.device)
                    ids_cat = torch.cat([ids0, ids1, ids2], dim=0)  # [3L, 3]

                    txt_ids = repeat(ids_cat, "l d -> b l d", b=bs_eff).to(weight_dtype)

                    assert txt_replacement.shape[1] == txt_ids.shape[1], \
                        f"txt len mismatch: txt={txt_replacement.shape[1]} vs txt_ids={txt_ids.shape[1]}"

                    # ---------------------------------------------------------
                    # 5) img_ids: target frame time_step=3
                    # ---------------------------------------------------------
                    with torch.no_grad():
                        dummy_out = prepare_clip(
                            super_model.clip_vis,
                            img0,               # shape 正确即可
                            x_1.to(weight_dtype)
                        )
                        target_img_ids = dummy_out["img_ids"].to(weight_dtype)
                        target_img_ids[..., 0] = 3.0

                    inp = {
                        "img_ids": target_img_ids,
                        "txt": txt_replacement,
                        "txt_ids": txt_ids,
                        "vec": vec_fused,
                    }

                    # ---------------------------------------------------------
                    # 6) FLUX training objective（保持你原公式）
                    # ---------------------------------------------------------
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                    t = torch.sigmoid(torch.randn((bs_eff,), device=accelerator.device) * args.scale_factor)
                    x_0 = torch.randn_like(x_1).to(accelerator.device)
                    x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0

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

                    loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                    train_loss += loss.detach().item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # monitor step time
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
                            f"loss={train_loss:.4f}, step_time={step_duration:.3f}s, "
                            f"data_load_time={data_load_duration:.3f}s, "
                            f"avg_windows_per_video={avg_nw:.2f}, bs_eff={bs_eff}"
                        )

                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "step_time": step_duration,
                            "data_load_time": data_load_duration,
                            "avg_windows_per_video": float(avg_nw),
                            "bs_eff": float(bs_eff),
                        },
                        step=global_step,
                    )
                    train_loss = 0.0

                    # checkpoint（保持你原保存方式：只存 dit + adapter；clip 不存也行）
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            unwrapped_super_model = accelerator.unwrap_model(super_model)
                            save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                            save_path_visual_adapter = os.path.join(
                                args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin"
                            )
                            save_path_optimizer = os.path.join(args.output_dir, f"optimizer-state-{global_step}.bin")

                            torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                            torch.save(deepcopy(unwrapped_super_model.visual_adapter).state_dict(), save_path_visual_adapter)
                            torch.save(optimizer.state_dict(), save_path_optimizer)

                            logger.info(f"Saved checkpoint at step {global_step}")

                progress_bar.set_postfix(step_loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

                if global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        unwrapped_super_model = accelerator.unwrap_model(super_model)
                        save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                        save_path_visual_adapter = os.path.join(args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin")
                        torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                        torch.save(deepcopy(unwrapped_super_model.visual_adapter).state_dict(), save_path_visual_adapter)
                    break

            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f"RuntimeError at step {step}: {error_msg}", exc_info=True)
                if "NCCL" in error_msg or "timeout" in error_msg.lower() or "distributed" in error_msg.lower():
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
