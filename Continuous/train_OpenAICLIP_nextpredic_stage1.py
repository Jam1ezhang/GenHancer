import argparse
import logging
import math
import os
import re
import random
import shutil
import time
from einops import repeat
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

# 设置NCCL环境变量以避免超时问题
# 注意：这些设置会作为默认值，如果shell脚本中已设置环境变量，则不会覆盖
os.environ.setdefault("NCCL_TIMEOUT", "7200")  # 设置超时时间为2小时（7200秒）
# PyTorch 2.3.0+ 使用新的环境变量名
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")  # 新版本使用这个
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # 旧版本兼容
os.environ.setdefault("NCCL_DEBUG", "WARN")  # 启用NCCL调试信息（WARN级别减少日志）

# GPU通信配置 - 根据NUMA拓扑优化
# 对于同一NUMA节点内的GPU，应该启用P2P以获得最佳性能
os.environ.setdefault("NCCL_P2P_DISABLE", "0")  # 启用P2P（同一NUMA节点内GPU应该支持）
os.environ.setdefault("NCCL_SHM_DISABLE", "0")  # 启用共享内存
os.environ.setdefault("NCCL_IB_DISABLE", "1")  # 禁用InfiniBand（如果没有）
# 对于单机多GPU，ring算法通常更稳定
os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")  # 使用ring算法
# NUMA亲和性优化
os.environ.setdefault("NCCL_TOPO_FILE", "")  # 让NCCL自动检测拓扑

# PyTorch分布式超时（毫秒）- 这是关键！必须在导入torch.distributed之前设置
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "7200000")  # 2小时（7200000毫秒）

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, unpack
from src.flux.util import configs, load_ae, load_flow_model2

from image_datasets.dataset_video import loader
from torchvision import transforms

from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


class VisualPromptAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        # 使用两层 MLP 进行特征空间转换
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)

class SuperModel(nn.Module):
    def __init__(self, clip_vis, dit):
        super().__init__()
        self.clip_vis = clip_vis
        self.dit = dit
        
        # --- 新增：初始化适配器 ---
        # 注意：这里假设使用 CLIP-Large (1024维) 和 FLUX.1 (4096维)
        # 如果是 SigLIP (1152维)，请修改 in_dim=1152
        self.visual_adapter = VisualPromptAdapter(in_dim=1024, out_dim=4096)
    
    def get_clip_vis(self):
        return self.clip_vis
    
    def get_dit(self):
        return self.dit

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    parsed_args = parser.parse_args()
    return parsed_args.config

def create_spatio_temporal_ids(h, w, time_step, device):
    """
    生成形状为 [h*w, 3] 的坐标张量。
    最后一维 3 代表: [time_step, row_idx, col_idx]
    """
    # 1. 生成空间网格 (h, w)
    # torch.meshgrid 返回 (H, W) 的坐标矩阵
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device), 
        torch.arange(w, device=device), 
        indexing='ij'
    )
    
    # Flatten 到一维序列 [h*w]
    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()
    
    # 2. 生成时间维度 (全都是 time_step)
    flat_t = torch.full_like(flat_h, fill_value=time_step)
    
    # 3. 堆叠成 [h*w, 3] -> (t, h, w)
    ids = torch.stack([flat_t, flat_h, flat_w], dim=1)
    
    return ids

def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)
    
    is_schnell = args.model_name == "flux-schnell"
    args.clip_config.seq_t5 = 256 if is_schnell else 512   # NOTE!!!
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
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


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit = load_flow_model2(args.model_name, device="cpu")
    vae = load_ae(args.model_name, device=accelerator.device)
    clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)


    # contiguous projections for OpenAICLIP-336px
    if args.clip_config.clip_image_size == 336:
        clip_vis.model.visual_projection.weight = torch.nn.Parameter(clip_vis.model.visual_projection.weight.contiguous())
        clip_vis.model.text_projection.weight = torch.nn.Parameter(clip_vis.model.text_projection.weight.contiguous())


    vae.requires_grad_(False)
    dit.requires_grad_(True)
    dit = dit.to(torch.bfloat16)
    dit.to(accelerator.device)
    clip_vis.train()
    dit.train()

    for name_, param in clip_vis.named_parameters():
        if 'project_clip' in name_:
            param.requires_grad = True
        else:
            param.requires_grad = False
    

    # super model = clip_vis + dit + visual_adapter
    super_model = SuperModel(clip_vis, dit)
    
    # Ensure visual_adapter is trainable
    super_model.visual_adapter.requires_grad_(True)

    params_to_optimize = [p for p in super_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)
    # Scheduler and math around the number of training steps.
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

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
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

    # 添加监控变量
    last_data_load_time = time.time()
    data_load_timeout = 60.0  # 数据加载超时阈值（秒）
    last_step_time = time.time()
    step_timeout = 300.0  # 单步训练超时阈值（秒）
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            try:
                # 监控数据加载时间
                current_time = time.time()
                data_load_duration = current_time - last_data_load_time
                if data_load_duration > data_load_timeout:
                    logger.warning(
                        f"[Rank {accelerator.process_index}] Data loading timeout: "
                        f"step={step}, duration={data_load_duration:.2f}s > {data_load_timeout}s"
                    )
                last_data_load_time = current_time
                
                # 检查batch是否有效
                if not batch or len(batch) == 0:
                    logger.warning(
                        f"[Rank {accelerator.process_index}] Empty batch at step {step}, skipping"
                    )
                    continue
                
                # 检查batch中的关键字段
                if 'start_frame' not in batch or 'middle_frame' not in batch:
                    logger.warning(
                        f"[Rank {accelerator.process_index}] Missing required fields in batch at step {step}"
                    )
                    continue
                
                step_start_time = time.time()
                
                with accelerator.accumulate(super_model):
                    # 从视频数据中提取帧和文本
                    current_frame = batch['start_frame'].to(accelerator.device)  # 当前帧
                    next_frame = batch['middle_frame'].to(accelerator.device)    # 下一帧
                    
                    # 使用VAE对next_frame进行编码
                    with torch.no_grad():
                        x_1 = vae.encode(NORMALIZE_VAE(next_frame).to(torch.float32))

                    current_img_norm = NORMALIZE_CLIP(current_frame).to(weight_dtype)
                    
                    with torch.no_grad():
                        # 获取 Current Frame 特征
                        # clip_vis.model 是 HF 的 CLIPModel
                        out_current = super_model.clip_vis.model.vision_model(current_img_norm, output_hidden_states=True)
                        # last_hidden_state: [B, Seq_Len+1, 1024] (Seq_Len=256 for 336px image)
                        # 去掉索引 0 的 CLS token，只保留 spatial tokens
                        patches_current = out_current.last_hidden_state[:, 1:, :] 

                        # 提取全局向量 vec (y) - 用于全局风格调制
                        # 使用 visual_projection 投影 CLS token
                        vec_current = super_model.clip_vis.model.visual_projection(out_current.pooler_output)
                        vec_fused = vec_current  # 只使用当前帧的全局向量

                    # 使用当前帧的 Patch 特征
                    visual_context_raw = patches_current

                    # 5. 通过 Adapter 映射到 Text 空间 (1024 -> 4096)
                    # 这是我们要训练的部分，所以要有梯度
                    txt_replacement = super_model.visual_adapter(visual_context_raw)

                    # 6. 构建 txt_ids (位置编码)
                    # 这是一个全 0 张量，或者你可以构建真实的空间坐标
                    # 形状: [B, 576, 3]
                    H_patch, W_patch = 24,24
                    # 构建 Current Frame 的 IDs (t=0)
                    ids_current = create_spatio_temporal_ids(H_patch, W_patch, time_step=0, device=accelerator.device)
                    # [576, 3]

                    # 扩展 Batch 维度
                    # 结果形状 [B, 576, 3]
                    bs = next_frame.shape[0]
                    txt_ids = repeat(ids_current, "l d -> b l d", b=bs).to(weight_dtype)
                    # 7. 构建 img_ids (针对 target image)
                    # 我们可以复用 prepare_clip 的一部分逻辑，或者手动构建
                    # 这里为了简单，调用一次 prepare_clip 仅为了获取 img_ids，计算量很小
                    dummy_out = prepare_clip(
                        super_model.clip_vis, 
                        current_img_norm, # 输入什么不重要，只要 shape 对
                        x_1.to(weight_dtype)
                    )
                    target_img_ids = dummy_out['img_ids']
                    target_img_ids[..., 0] = 1.0  # 下一帧的时间步设为 1

                    # 8. 组装最终输入
                    inp = {
                        'img_ids': target_img_ids,
                        'txt': txt_replacement,  # <--- 核心：现在是 512 个视觉 Token
                        'txt_ids': txt_ids,
                        'vec': vec_fused         # 全局上下文
                    }
                    
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    t = torch.sigmoid(torch.randn((bs,), device=accelerator.device) * args.scale_factor)
                    x_0 = torch.randn_like(x_1).to(accelerator.device)
                    x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                    guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                    # Predict the noise residual and compute loss
                    model_pred = super_model.dit(
                        img=x_t.to(weight_dtype),
                        img_ids=inp['img_ids'].to(weight_dtype),
                        txt=inp['txt'].to(weight_dtype),
                        txt_ids=inp['txt_ids'].to(weight_dtype),
                        y=inp['vec'].to(weight_dtype),
                        timesteps=t.to(weight_dtype),
                        guidance=guidance_vec.to(weight_dtype),
                    )

                    loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                    # Accumulate loss for logging (local, no synchronization here)
                    train_loss += loss.detach().item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # 监控单步训练时间
                step_duration = time.time() - step_start_time
                if step_duration > step_timeout:
                    logger.error(
                        f"[Rank {accelerator.process_index}] Step timeout: "
                        f"step={step}, duration={step_duration:.2f}s > {step_timeout}s"
                    )
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # Note: We don't gather loss here to avoid blocking on data loading inconsistencies
                    # Gradients are already synchronized through backward(), so training is correct
                    # Each rank logs its local loss (which should be similar across ranks)
                    
                    # 每10步记录一次详细监控信息
                    if global_step % 10 == 0:
                        logger.info(
                            f"[Rank {accelerator.process_index}] Step {global_step}: "
                            f"loss={train_loss:.4f}, step_time={step_duration:.3f}s, "
                            f"data_load_time={data_load_duration:.3f}s"
                        )
                    
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({
                        "train_loss": train_loss,
                        "step_time": step_duration,
                        "data_load_time": data_load_duration
                    }, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            unwrapped_super_model = accelerator.unwrap_model(super_model)
                            save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                            save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                            save_path_visual_adapter = os.path.join(args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin")
                            save_path_optimizer = os.path.join(args.output_dir, f"optimizer-state-{global_step}.bin")
                            
                            torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                            torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                            torch.save(deepcopy(unwrapped_super_model.visual_adapter).state_dict(), save_path_visual_adapter)
                            torch.save(optimizer.state_dict(), save_path_optimizer)

                            logger.info(f"Saved checkpoint at step {global_step}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        unwrapped_super_model = accelerator.unwrap_model(super_model)
                        save_path_dit = os.path.join(args.output_dir, f"checkpoint-dit-{global_step}.bin")
                        save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                        save_path_visual_adapter = os.path.join(args.output_dir, f"checkpoint-visual-adapter-{global_step}.bin")
                        
                        torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                        torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
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

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
