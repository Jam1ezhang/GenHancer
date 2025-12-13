export AE="/home/user/gptdata/zym/codespace_hallucination/ckpts/FLUX.1-dev/ae.safetensors"

# ============================================
# NCCL环境变量配置 - 解决GPU间通信问题
# ============================================

# 超时设置（单位：秒，注意PyTorch内部会转换为毫秒）
# export NCCL_TIMEOUT=7200  # 2小时超时（7200秒）

# # 通信方式配置
# # 如果GPU间无法通过P2P通信，强制使用共享内存或网络
# export NCCL_P2P_DISABLE=1  # 禁用P2P通信，使用网络回退
# export NCCL_SHM_DISABLE=0  # 启用共享内存（如果GPU在同一台机器上）
# export NCCL_IB_DISABLE=0   # 启用InfiniBand（如果有）
# export NCCL_SOCKET_IFNAME=lo  # 使用loopback接口（单机多GPU）

# # 通信后端优先级：优先使用共享内存，然后网络
# export NCCL_NET_GDR_LEVEL=0  # 禁用GPU Direct RDMA（如果不可用）
# export NCCL_NET_GDR_READ=0   # 禁用GDR读取

# # 错误处理和调试
# export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
# export NCCL_DEBUG=WARN  # 设置为WARN减少日志量，如需调试可改为INFO

# # 其他优化设置
# export NCCL_TREE_THRESHOLD=0  # 强制使用ring算法（对单机多GPU更稳定）
# export NCCL_MIN_NCHANNELS=4   # 最小通信通道数
# export NCCL_MAX_NCHANNELS=16  # 最大通信通道数
# export NCCL_BUFFSIZE=2097152  # 缓冲区大小（2MB）

# # DeepSpeed相关超时设置
# export DEEPSPEED_COMM_TIMEOUT=7200  # DeepSpeed通信超时（秒）

# # PyTorch分布式超时（毫秒）
# export TORCH_DISTRIBUTED_TIMEOUT=7200000  # 2小时（7200000毫秒）

# echo "NCCL环境变量已设置："
# echo "  NCCL_TIMEOUT=$NCCL_TIMEOUT"
# echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
# echo "  NCCL_DEBUG=$NCCL_DEBUG"
accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_video_stage1.py --config "train_configs/test_OpenAICLIP_336_video_stage1.yaml" 
