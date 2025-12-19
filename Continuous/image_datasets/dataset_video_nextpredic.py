import os
import sys
import tarfile
import io
import random
import logging
from typing import Optional, Sequence, Dict, Union, Tuple
from dataclasses import dataclass
import glob  # <--- 新增这一行
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

# 引入 webdataset
try:
    import webdataset as wds
except ImportError:
    print("Error: webdataset library is missing. Please install it using 'pip install webdataset'")
    sys.exit(1)

# 配置日志
logger = logging.getLogger(__name__)

OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    try:
        return image.convert('RGB')
    except Exception:
        return image

def to_tensor_func(image):
    try:
        return ToTensor()(image)
    except Exception:
        return image.float()

def image_transform(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    
    if not isinstance(mean, (list, tuple)): mean = (mean,) * 3
    if not isinstance(std, (list, tuple)): std = (std,) * 3

    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            to_tensor_func,
            Normalize(mean=mean, std=std)
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            to_tensor_func,
            Normalize(mean=mean, std=std)
        ])

# ==========================================
# 核心修复：自定义 WebDataset 分组与处理逻辑
# ==========================================

def group_by_directory(data):
    """
    生成器：将扁平的文件流按照目录名聚合成视频样本。
    假设 tar 包中的文件是按目录顺序排列的（WebDataset 的标准行为）。
    
    添加了超时和错误处理机制，防止卡住：
    - 如果某个视频的buffer等待时间过长，强制yield
    - 如果遇到格式异常的数据，跳过并继续
    """
    import time
    
    current_video_id = None
    video_buffer = {}
    last_video_start_time = None
    max_wait_time = 30.0  # 最多等待30秒
    max_samples_per_video = 1000  # 每个视频最多处理1000个sample
    
    sample_count_since_yield = 0

    try:
        for sample in data:
            sample_count_since_yield += 1
            
            # 超时保护：如果等待太久，强制yield当前buffer并重置
            if last_video_start_time is not None:
                wait_time = time.time() - last_video_start_time
                if wait_time > max_wait_time or sample_count_since_yield > max_samples_per_video:
                    if video_buffer:
                        logger.warning(
                            f"Timeout/overflow in group_by_directory: video_id={current_video_id}, "
                            f"wait_time={wait_time:.2f}s, samples={sample_count_since_yield}, "
                            f"yielding incomplete buffer with {len(video_buffer)} items"
                        )
                        video_buffer['__key__'] = current_video_id
                        yield video_buffer
                        video_buffer = {}
                        sample_count_since_yield = 0
                    last_video_start_time = None
                    current_video_id = None
            
            # sample 是 webdataset 返回的字典
            # key 示例: '150593/frame_0' 或 '150593/txt'
            try:
                key = sample.get("__key__")
                if not key:
                    continue
                    
                # 提取目录ID (如 '150593')
                parts = key.split('/')
                if len(parts) < 2:
                    # 如果文件不在子目录中，暂时忽略或根据需要处理
                    continue
                    
                video_id = parts[0]
                filename = parts[-1] # 如 'frame_0' 或 'txt'
                
                # 如果遇到了新的视频ID，先yield出上一个视频的所有数据
                if video_id != current_video_id:
                    if current_video_id is not None and video_buffer:
                        # 检查buffer是否有基本内容（至少有一个frame）
                        has_frames = any('frame_' in k for k in video_buffer.keys())
                        if has_frames:
                            video_buffer['__key__'] = current_video_id
                            yield video_buffer
                        else:
                            logger.debug(f"Skipping empty video buffer for {current_video_id}")
                    
                    current_video_id = video_id
                    video_buffer = {}
                    last_video_start_time = time.time()
                    sample_count_since_yield = 0
                    
                # 将当前文件数据合并到 buffer 中
                # WebDataset 会把扩展名作为 key (例如 'jpg', 'txt', 'png')
                # 我们将其重命名为 'frame_0.jpg' 这样的格式以便后续识别
                for ext, content in sample.items():
                    if ext.startswith("__"): 
                        continue # 跳过元数据
                    
                    # 构建新的唯一键，例如: 'frame_0.jpg'
                    new_key = f"{filename}.{ext}"
                    video_buffer[new_key] = content
                    
            except Exception as e:
                # 如果处理sample时出错，记录并继续
                sample_key = sample.get("__key__", "UNKNOWN")
                logger.warning(f"Error processing sample {sample_key} in group_by_directory: {e}")
                continue

        # 处理最后一个样本
        if current_video_id is not None and video_buffer:
            has_frames = any('frame_' in k for k in video_buffer.keys())
            if has_frames:
                video_buffer['__key__'] = current_video_id
                yield video_buffer
            else:
                logger.debug(f"Skipping empty final video buffer for {current_video_id}")
                
    except Exception as e:
        # 如果生成器本身出错，记录并尝试yield当前buffer
        logger.error(f"Critical error in group_by_directory: {e}", exc_info=True)
        if current_video_id is not None and video_buffer:
            has_frames = any('frame_' in k for k in video_buffer.keys())
            if has_frames:
                video_buffer['__key__'] = current_video_id
                yield video_buffer


def process_wds_sample(sample, img_processor):
    """
    将聚合后的二进制数据解码为图像 Tensor 和文本。
    """
    frames = {}
    text = ""
    
    try:
        # 1. 遍历字典，分离出图像帧和文本
        for key, content in sample.items():
            if key.startswith("__"): continue
            
            # 处理图像 (支持 jpg, png 等)
            if 'frame_' in key and any(ext in key for ext in ['jpg', 'png', 'jpeg', 'webp']):
                # 提取帧号: 'frame_0.jpg' -> 0
                try:
                    frame_part = key.split('.')[0] # frame_0
                    idx = int(frame_part.split('_')[1])
                    frames[idx] = content
                except Exception:
                    pass
            
            # 处理文本
            elif 'txt' in key:
                text = content
        
        if not frames:
            return None

        # 2. 排序并选取连续的帧对 (current_frame, next_frame)
        sorted_indices = sorted(frames.keys())
        if not sorted_indices or len(sorted_indices) < 2:
            return None
            
        # 随机选择一个起始帧（除了最后一帧）
        idx_current = random.choice(sorted_indices[:-1])
        # 找到下一个帧的索引
        idx_next = sorted_indices[sorted_indices.index(idx_current) + 1]
        
        idx_start = idx_current  # 当前帧
        idx_mid = idx_next       # 下一帧
        idx_end = idx_current    # 保持一致，因为我们只需要当前帧预测下一帧
        
        # 3. 解码图像 (Bytes -> PIL -> Tensor)
        # 添加超时保护，防止损坏的图像导致卡住
        def decode_img(img_bytes):
            import signal
            from contextlib import contextmanager
            
            @contextmanager
            def timeout_context(seconds):
                """为图像解码添加超时保护"""
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Image decode timeout after {seconds}s")
                
                # 只在Unix系统上使用signal（Windows不支持SIGALRM）
                if hasattr(signal, 'SIGALRM'):
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    # Windows系统：直接执行，不设置超时（Windows不支持SIGALRM）
                    yield
            
            try:
                if hasattr(signal, 'SIGALRM'):
                    with timeout_context(5):  # 5秒超时
                        return img_processor(Image.open(io.BytesIO(img_bytes)))
                else:
                    # Windows系统：直接执行
                    return img_processor(Image.open(io.BytesIO(img_bytes)))
            except (TimeoutError, Exception) as e:
                logger.warning(f"Image decode failed: {type(e).__name__}: {e}")
                return None
            
        start_frame = decode_img(frames[idx_start])
        end_frame = decode_img(frames[idx_end])
        middle_frame = decode_img(frames[idx_mid])
        
        # 如果任何图像解码失败，跳过这个样本
        if start_frame is None or end_frame is None or middle_frame is None:
            return None
        
        # 4. 解码文本 (Bytes -> Str)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors='ignore').strip()
            
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'middle_frame': middle_frame,
            'text': text,
            '__key__': sample.get('__key__', '')
        }

    except Exception as e:
        sample_key = sample.get('__key__', 'UNKNOWN_KEY')
        logger.warning(f"Failed to process sample {sample_key}: {type(e).__name__}: {e}")
        return None


def get_video_dataset_and_collator(img_size, video_dir, seed, patch_size, train_batch_size=1, num_workers=0):
    # 准备图像处理器
    train_processor = image_transform(img_size, is_train=True)
    
    # -----------------------------------------------------------
    # 修复逻辑：手动展开 glob 通配符，获取真实文件列表
    # -----------------------------------------------------------
    urls = []
    if isinstance(video_dir, str):
        # 情况1：传入的是目录路径 (e.g., /path/to/data)
        if os.path.isdir(video_dir):
            pattern = os.path.join(video_dir, "*.tar")
            urls = sorted(glob.glob(pattern))
        # 情况2：传入的是带通配符的路径 (e.g., /path/to/data/*.tar)
        elif "*" in video_dir:
            urls = sorted(glob.glob(video_dir))
        # 情况3：传入的是单个文件或 URL
        else:
            urls = [video_dir]
            
        if not urls:
            raise FileNotFoundError(f"No .tar files found using path/pattern: {video_dir}")
            
        logger.info(f"Found {len(urls)} tar files for training.")
    else:
        # 情况4：传入的已经是列表
        urls = video_dir

    # -----------------------------------------------------------
    # 构建 WebDataset 管道
    # -----------------------------------------------------------
    # nodesplitter: 分布式训练时，将文件分配给不同节点
    # shardshuffle: 打乱 tar 文件的读取顺序
    # 关键修复：只在 num_workers > 0 时使用 split_by_worker
    # dataset = (
    #     wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=False, empty_check=False)
    #     .shuffle(1000, initial=1000)
    # )
    dataset = (
        wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=False, empty_check=False)
        .shuffle(1000, initial=1000)
        .repeat()
    )

    # # 只在有多个worker时才使用split_by_worker
    # # 当num_workers=0时，只有一个主进程，不需要split_by_worker
    if num_workers > 0:
        dataset = dataset.compose(wds.split_by_worker)
    
    dataset = (
        dataset
        .compose(group_by_directory)  
        .map(lambda s: process_wds_sample(s, train_processor))
        .select(lambda s: s is not None) 
    )   

    # dataset = (
    #     wds.ResampledShards(urls, deterministic=True)  # 替代 WebDataset(...)
    #     # 不要 split_by_worker，当 num_workers=0 时已去掉
    #     .compose(group_by_directory)
    #     .map(lambda s: process_wds_sample(s, train_processor))
    #     .select(lambda s: s is not None)
    # )

    data_collator = VideoFramesCollator(patch_size)

    return dataset, data_collator

@dataclass
class VideoFramesCollator:
    patch_size: int = 1
    
    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 过滤 None
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}

        start_frames = [sample["start_frame"] for sample in samples]
        end_frames = [sample["end_frame"] for sample in samples]
        middle_frames = [sample["middle_frame"] for sample in samples]
        texts = [sample["text"] for sample in samples]

        batch = {}
        
        def stack_frames(frames):
            if all(x is not None and x.shape == frames[0].shape for x in frames):
                return torch.stack(frames)
            return frames

        batch['start_frame'] = stack_frames(start_frames)
        batch['end_frame'] = stack_frames(end_frames)
        batch['middle_frame'] = stack_frames(middle_frames)
        batch['text'] = texts
        
        return batch


def loader(train_batch_size, num_workers, **args):
    """
    对外暴露的加载器接口
    """
    # 将num_workers传递给get_video_dataset_and_collator，以便正确配置split_by_worker
    dataset, collator = get_video_dataset_and_collator(train_batch_size=train_batch_size, num_workers=num_workers, **args)
    
    # 使用 DataLoader 包装 WebDataset
    # 注意：WebDataset 是 IterableDataset，不能使用 shuffle=True (已经在管道中处理了)
    # 在分布式训练中，减少num_workers可以避免死锁问题
    # persistent_workers可以避免重复创建worker，提高效率
    dataloader = DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        num_workers=num_workers if num_workers > 0 else 0, 
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,  # 只有在使用workers时才启用
        prefetch_factor=2 if num_workers > 0 else None,  # 预取因子，减少等待时间
    )
    
    return dataloader