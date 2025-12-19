#!/bin/bash

# ============================================
# OpenAI CLIP 336px Video Stage2 All Training
# ============================================

export AE="/home/user/gptdata/zym/codespace_hallucination/ckpts/FLUX.1-dev/ae.safetensors"

# ============================================
# NCCL环境变量配置 - 解决GPU间通信问题
# ============================================
# 这些设置在代码中已经配置，这里注释掉避免重复
# 如果训练时遇到NCCL超时问题，可以取消注释并调整参数

# export NCCL_TIMEOUT=7200  # 2小时超时（7200秒）
# export NCCL_P2P_DISABLE=1  # 禁用P2P通信，使用网络回退
# export NCCL_DEBUG=WARN  # 设置为WARN减少日志量
# export TORCH_DISTRIBUTED_TIMEOUT=7200000  # PyTorch分布式超时（毫秒）

echo "=========================================="
echo "Starting OpenAI CLIP-336 Video Stage2 All Training"
echo "Load from: output_OpenAICLIP_336_video_stage1_313/"
echo "Output to: output_OpenAICLIP_336_video_stage2_all_load313/"
echo "=========================================="

accelerate launch \
    --config_file "train_configs/accelerate_config.yaml" \
    train_OpenAICLIP_video_stage2_all.py \
    --config "train_configs/test_OpenAICLIP_336_video_stage2_all.yaml"

echo "=========================================="
echo "Training completed!"
echo "=========================================="

