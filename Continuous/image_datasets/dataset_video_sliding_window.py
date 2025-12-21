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

def group_by_video_from_full_key(data):
    """
    data: iterator of samples from tarfile_to_samples()
    Each sample has:
      sample["__key__"] like "2006.../frame_000000.jpg" (真实路径，不是抽象key)
      and binary under "jpg" or "txt"
    We group by the directory part before the last '/'.
    """
    current_vid = None
    buf = {}

    for sample in data:
        k = sample.get("__key__", "")
        if not k:
            continue

        # k 是真实路径： video_id/frame_000000.jpg 或 video_id/txt
        if "/" not in k:
            continue

        video_id, fname = k.split("/", 1)

        if current_vid is None:
            current_vid = video_id

        if video_id != current_vid:
            if buf:
                buf["__key__"] = current_vid
                yield buf
            buf = {}
            current_vid = video_id

        # 把内容塞进 buf，保持你 process_wds_sample 的格式
        for ext, content in sample.items():
            if ext.startswith("__"):
                continue
            # ext 在 tarfile_to_samples 下通常会是 'jpg' 或 'txt'
            new_key = f"{fname}.{ext}" if "." not in fname else fname  # fname 已经带 .jpg 也行
            buf[new_key] = content

    if current_vid is not None and buf:
        buf["__key__"] = current_vid
        yield buf



def process_wds_sample(sample, img_processor, return_mode: str = "triplet", max_frames: Optional[int] = None):
    """
    将聚合后的二进制数据解码为图像 Tensor 和文本。

    return_mode:
      - "triplet": 返回 start/middle/end (兼容你旧脚本)
      - "full_frames": 返回 frames: Tensor[T,C,H,W] (供滑动窗口)
    max_frames:
      - 如果 full_frames 很长，可限制最多取多少帧（按时间顺序取前 max_frames 帧）
    """
    frames = {}
    text = ""

    try:
        # 1) 分离图像帧和文本
        for key, content in sample.items():
            if key.startswith("__"):
                continue

            if "frame_" in key and any(ext in key for ext in ["jpg", "png", "jpeg", "webp"]):
                try:
                    frame_part = key.split(".")[0]  # frame_000123
                    idx = int(frame_part.split("_")[1])
                    frames[idx] = content
                except Exception:
                    pass

            elif "txt" in key:
                text = content

        if not frames:
            return None

        sorted_indices = sorted(frames.keys())
        if not sorted_indices:
            return None

        # 2) 解码工具：Bytes -> PIL -> Tensor，带超时
        def decode_img(img_bytes):
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout_context(seconds):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Image decode timeout after {seconds}s")

                if hasattr(signal, "SIGALRM"):
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    yield

            try:
                if hasattr(signal, "SIGALRM"):
                    with timeout_context(5):
                        return img_processor(Image.open(io.BytesIO(img_bytes)))
                else:
                    return img_processor(Image.open(io.BytesIO(img_bytes)))
            except (TimeoutError, Exception) as e:
                logger.warning(f"Image decode failed: {type(e).__name__}: {e}")
                return None

        # 3) 解码文本
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore").strip()

        # 4) 两种返回模式
        if return_mode == "triplet":
            idx_start = sorted_indices[0]
            idx_end = sorted_indices[-1]
            idx_mid = sorted_indices[len(sorted_indices) // 2]

            start_frame = decode_img(frames[idx_start])
            end_frame = decode_img(frames[idx_end])
            middle_frame = decode_img(frames[idx_mid])

            if start_frame is None or end_frame is None or middle_frame is None:
                return None

            return {
                "start_frame": start_frame,
                "middle_frame": middle_frame,
                "end_frame": end_frame,
                "text": text,
                "__key__": sample.get("__key__", ""),
            }

        elif return_mode == "full_frames":
            # 可选：限制帧数（取前 max_frames）
            if max_frames is not None and max_frames > 0:
                sorted_indices = sorted_indices[:max_frames]

            decoded = []
            kept_indices = []
            for idx in sorted_indices:
                im = decode_img(frames[idx])
                if im is None:
                    # 任意一帧坏了：策略你可以二选一
                    # 1) 跳过坏帧（更鲁棒）
                    continue
                    # 2) 直接丢弃整个视频（更干净）
                    # return None
                decoded.append(im)
                kept_indices.append(idx)

            if len(decoded) < 4:
                # 少于 4 帧，连 [t,t+1,t+2]->t+3 都做不了
                return None

            # decoded: list of [C,H,W] -> [T,C,H,W]
            frames_tensor = torch.stack(decoded, dim=0)

            return {
                "frames": frames_tensor,
                "frame_indices": torch.tensor(kept_indices, dtype=torch.long),
                "text": text,
                "__key__": sample.get("__key__", ""),
            }

        else:
            raise ValueError(f"Unknown return_mode: {return_mode}")

    except Exception as e:
        sample_key = sample.get("__key__", "UNKNOWN_KEY")
        logger.warning(f"Failed to process sample {sample_key}: {type(e).__name__}: {e}")
        return None



def get_video_dataset_and_collator(
    img_size, video_dir, seed, patch_size,
    train_batch_size=1, num_workers=0,
    return_mode: str = "triplet",
    max_frames_per_video: Optional[int] = None
):
    train_processor = image_transform(img_size, is_train=True)

    # 1) 展开 tar 列表
    if isinstance(video_dir, str):
        if os.path.isdir(video_dir):
            urls = sorted(glob.glob(os.path.join(video_dir, "*.tar")))
        elif "*" in video_dir:
            urls = sorted(glob.glob(video_dir))
        else:
            urls = [video_dir]
    else:
        urls = list(video_dir)

    if not urls:
        raise FileNotFoundError(f"No .tar files found using path/pattern: {video_dir}")
    logger.info(f"Found {len(urls)} tar files for training.")

    # 2) 一条 DataPipeline 走到底（不要再额外 compose 旧逻辑）
    pipeline = wds.DataPipeline(
        # 无限重复读 shard
        wds.ResampledShards(urls),

        wds.split_by_node,
        wds.split_by_worker if num_workers > 0 else (lambda x: x),

        # 先吐出 tar 内的单文件样本（每个是 frame/txt）
        wds.tarfile_to_samples(handler=wds.warn_and_continue),

        # 关键：先按 video_id 聚合成“一个视频一个样本”
        group_by_video_from_full_key,

        # 关键：再 shuffle（此时 shuffle 的单位是“视频样本”，不会把帧打碎）
        # toy 数据先用小一点，避免 initial 等太久
        wds.shuffle(50, initial=10),

        # 聚合样本 -> 解码为 frames tensor
        wds.map(lambda s: process_wds_sample(
            s, train_processor,
            return_mode=return_mode,
            max_frames=max_frames_per_video
        )),

        wds.select(lambda s: s is not None),
    )



    data_collator = VideoFramesCollator(patch_size)
    return pipeline, data_collator



@dataclass
class VideoFramesCollator:
    patch_size: int = 1

    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}

        batch = {}

        # -------- full_frames path --------
        if "frames" in samples[0]:
            # 每个 sample["frames"] : [T,C,H,W]，T 可变
            frames_list = [s["frames"] for s in samples]
            texts = [s.get("text", "") for s in samples]

            lengths = [f.shape[0] for f in frames_list]
            T_max = max(lengths)
            B = len(frames_list)
            C, H, W = frames_list[0].shape[1:]

            frames_pad = torch.zeros((B, T_max, C, H, W), dtype=frames_list[0].dtype)
            frame_mask = torch.zeros((B, T_max), dtype=torch.bool)

            for i, f in enumerate(frames_list):
                t = f.shape[0]
                frames_pad[i, :t] = f
                frame_mask[i, :t] = True

            batch["frames"] = frames_pad
            batch["frame_mask"] = frame_mask
            batch["text"] = texts

            if "frame_indices" in samples[0]:
                # 同样做 padding
                idx_list = [s["frame_indices"] for s in samples]
                idx_pad = torch.full((B, T_max), fill_value=-1, dtype=idx_list[0].dtype)
                for i, idx in enumerate(idx_list):
                    t = idx.shape[0]
                    idx_pad[i, :t] = idx
                batch["frame_indices"] = idx_pad

            return batch

        # -------- triplet path (旧逻辑) --------
        start_frames = [s["start_frame"] for s in samples]
        middle_frames = [s["middle_frame"] for s in samples]
        end_frames = [s["end_frame"] for s in samples]
        texts = [s.get("text", "") for s in samples]

        def stack_frames(frames):
            if all(x is not None and x.shape == frames[0].shape for x in frames):
                return torch.stack(frames)
            return frames

        batch["start_frame"] = stack_frames(start_frames)
        batch["middle_frame"] = stack_frames(middle_frames)
        batch["end_frame"] = stack_frames(end_frames)
        batch["text"] = texts
        return batch



def loader(train_batch_size, num_workers, return_mode="triplet", max_frames_per_video=None, **args):
    dataset, collator = get_video_dataset_and_collator(
        train_batch_size=train_batch_size,
        num_workers=num_workers,
        return_mode=return_mode,
        max_frames_per_video=max_frames_per_video,
        **args
    )
    
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