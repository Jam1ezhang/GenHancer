#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tarfile
import tempfile
from tqdm import tqdm
import argparse
from typing import List, Tuple, Optional

# Optional SSIM (fallback gracefully if not installed)
try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    ssim = None
    _HAS_SKIMAGE = False


# -----------------------------
# Video frame extraction
# -----------------------------
def extract_frames(
    video_path: str,
    sample_rate: int = 1,
    sample_mode: str = "fps",
    fps_target: float = 8.0,
    time_interval: int = 1000,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    从视频中提取帧，支持多种采样模式

    参数:
    - video_path: 视频文件路径
    - sample_rate: 固定帧间隔采样时的采样率（每sample_rate帧提取一帧）
    - sample_mode:
        'fixed': 固定帧间隔（默认），使用sample_rate参数
        'fps': 基于FPS的采样，提取指定FPS的帧，使用fps_target参数
        'fixed_time': 固定时间间隔采样，使用time_interval参数（毫秒）
    - fps_target: 当sample_mode='fps'时，目标FPS值
    - time_interval: 当sample_mode='fixed_time'时，时间间隔（毫秒）
    - max_frames: 最大提取帧数（None表示不限制；建议 toy 阶段限制一下防止超长视频）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    frame_count = 0
    last_extracted_time = -time_interval

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        should_extract = False

        if sample_mode == "fixed":
            should_extract = (frame_count % max(1, sample_rate) == 0)
        elif sample_mode == "fps":
            if video_fps and video_fps > 0 and fps_target > 0:
                frame_interval = max(1, int(round(video_fps / fps_target)))
                should_extract = (frame_count % frame_interval == 0)
            else:
                should_extract = (frame_count % max(1, sample_rate) == 0)
        elif sample_mode == "fixed_time":
            should_extract = (current_time - last_extracted_time >= time_interval)
        else:
            # unknown mode -> fallback
            should_extract = (frame_count % max(1, sample_rate) == 0)

        if should_extract:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(float(current_time))
            last_extracted_time = current_time

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames, timestamps


# -----------------------------
# Frame selection (triplet)
# -----------------------------
def select_frames_by_ssim(frames: List[np.ndarray], n_frames: int = 3) -> List[int]:
    """
    选择最具代表性的 n_frames 个帧（基于 SSIM 差异最大化）
    需要 skimage; 若不可用请不要调用该函数。
    """
    if len(frames) <= n_frames:
        return list(range(len(frames)))

    if not _HAS_SKIMAGE or ssim is None:
        raise RuntimeError("SSIM requires scikit-image. Please install scikit-image or use optical_flow method.")

    frame_count = len(frames)
    diff_matrix = np.zeros((frame_count, frame_count), dtype=np.float32)

    # precompute grayscale (and optionally resize later)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    for i in range(frame_count):
        for j in range(i + 1, frame_count):
            a = grays[i]
            b = grays[j]
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            ssim_val = float(ssim(a, b))
            diff = 1.0 - ssim_val
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff

    selected = [0]
    remaining = list(range(1, frame_count))

    while len(selected) < n_frames and remaining:
        best_idx = -1
        best_score = -1.0
        for idx in remaining:
            score = float(np.sum(diff_matrix[idx, selected]))
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    selected.sort()
    return selected


def select_frames_by_optical_flow(frames: List[np.ndarray], n_frames: int = 3) -> List[int]:
    """
    使用光流法选择 n_frames 个帧：倾向选择“运动变化显著且在时间上分布”的帧。
    """
    if len(frames) <= n_frames:
        return list(range(len(frames)))

    flow_intensities: List[float] = []
    prev_gray = None

    for frame in frames:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            flow_intensities.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_intensities.append(float(np.mean(magnitude)))
        prev_gray = curr_gray

    cumulative = np.cumsum(np.array(flow_intensities, dtype=np.float32))
    total = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    # Always include first and last
    selected = [0, len(frames) - 1]
    if n_frames <= 2:
        return sorted(list(set(selected)))

    # Choose intermediate frames near equally spaced flow-energy targets
    targets = [total * (i + 1) / (n_frames - 1) for i in range(n_frames - 2)]
    candidates = list(range(1, len(frames) - 1))

    for t in targets:
        if not candidates:
            break
        best = min(candidates, key=lambda x: abs(float(cumulative[x]) - t))
        selected.append(best)
        candidates.remove(best)

    selected = sorted(list(set(selected)))
    # if still less (rare), fill by uniform spacing
    while len(selected) < n_frames:
        idx = int(round((len(frames) - 1) * (len(selected) / (n_frames - 1))))
        selected.append(idx)
        selected = sorted(list(set(selected)))

    return selected[:n_frames]


# -----------------------------
# WebDataset writing helpers
# -----------------------------
def _safe_sample_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = base.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return base


def add_to_tar(tar_file: tarfile.TarFile, temp_dir: str, files_rel: List[str]) -> None:
    for rel_path in files_rel:
        abs_path = os.path.join(temp_dir, rel_path)
        tar_file.add(abs_path, arcname=rel_path)


def create_triplet_entry(
    temp_dir: str,
    frames: List[np.ndarray],
    selected_indices: List[int],
    video_path: str,
    sample_id: str,
) -> List[str]:
    """
    输出结构：
      sample_id/
        frame_0.jpg
        frame_1.jpg
        frame_2.jpg
        txt
    """
    folder = os.path.join(temp_dir, sample_id)
    os.makedirs(folder, exist_ok=True)

    rel_files: List[str] = []
    for i, fidx in enumerate(selected_indices):
        fn = f"frame_{i}.jpg"
        fp = os.path.join(folder, fn)
        cv2.imwrite(fp, cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR))
        rel_files.append(os.path.join(sample_id, fn))

    txt_path = os.path.join(folder, "txt")
    with open(txt_path, "w") as f:
        f.write(f"Video from {os.path.basename(video_path)}")
    rel_files.append(os.path.join(sample_id, "txt"))

    return rel_files


def create_full_frames_entry(
    temp_dir: str,
    frames: List[np.ndarray],
    video_path: str,
    sample_id: str,
) -> List[str]:
    """
    输出结构（兼容 dataset_video_nextpredic.py 读取 frame_*.jpg 并排序）：
      sample_id/
        frame_000000.jpg
        frame_000001.jpg
        ...
        txt
    """
    folder = os.path.join(temp_dir, sample_id)
    os.makedirs(folder, exist_ok=True)

    rel_files: List[str] = []
    for i, frame in enumerate(frames):
        fn = f"frame_{i:06d}.jpg"
        fp = os.path.join(folder, fn)
        cv2.imwrite(fp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        rel_files.append(os.path.join(sample_id, fn))

    txt_path = os.path.join(folder, "txt")
    with open(txt_path, "w") as f:
        f.write(f"Video from {os.path.basename(video_path)}")
    rel_files.append(os.path.join(sample_id, "txt"))

    return rel_files


def create_frame_pair_entry(
    temp_dir: str,
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    video_path: str,
    sample_id: str,
) -> List[str]:
    """
    输出结构（frame_pair 模式）：
      sample_id/
        frame_prev.jpg
        frame_next.jpg
        txt
    """
    folder = os.path.join(temp_dir, sample_id)
    os.makedirs(folder, exist_ok=True)

    prev_path = os.path.join(folder, "frame_prev.jpg")
    next_path = os.path.join(folder, "frame_next.jpg")
    cv2.imwrite(prev_path, cv2.cvtColor(frame_prev, cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_path, cv2.cvtColor(frame_next, cv2.COLOR_RGB2BGR))

    txt_path = os.path.join(folder, "txt")
    with open(txt_path, "w") as f:
        f.write(f"Frame pair from {os.path.basename(video_path)}")

    return [
        os.path.join(sample_id, "frame_prev.jpg"),
        os.path.join(sample_id, "frame_next.jpg"),
        os.path.join(sample_id, "txt"),
    ]


# -----------------------------
# Pair selection (frame_pair)
# -----------------------------
def select_frame_pairs_by_difference(
    frames: List[np.ndarray],
    n_pairs: Optional[int] = None,
    method: str = "optical_flow",
    top_percent: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    从连续帧对里选择变化最大的帧对（用于 frame_pair 模式的 top_difference）
    """
    if len(frames) < 2:
        return []

    diffs: List[Tuple[int, int, float]] = []
    for i in range(len(frames) - 1):
        a = frames[i]
        b = frames[i + 1]

        if method == "optical_flow":
            a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(a_g, b_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            score = float(np.mean(mag))
        elif method == "pixel_diff":
            score = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        elif method == "ssim":
            if not _HAS_SKIMAGE or ssim is None:
                # fallback to pixel_diff
                score = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
            else:
                a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
                if a_g.shape != b_g.shape:
                    b_g = cv2.resize(b_g, (a_g.shape[1], a_g.shape[0]))
                score = 1.0 - float(ssim(a_g, b_g))
        else:
            # fallback
            score = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

        diffs.append((i, i + 1, score))

    diffs.sort(key=lambda x: x[2], reverse=True)

    if n_pairs is None:
        n_pairs = max(1, int(len(diffs) * float(top_percent)))
    else:
        n_pairs = min(int(n_pairs), len(diffs))

    pairs = [(a, b) for a, b, _ in diffs[:n_pairs]]
    pairs.sort(key=lambda x: x[0])
    return pairs


# -----------------------------
# Main conversion logic
# -----------------------------
def list_video_files(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def convert_videos_to_webdataset(
    input_dir: str,
    output_dir: str,
    dataset_type: str = "triplet",  # triplet | frame_pair | full_frames
    sample_mode: str = "fps",
    sample_rate: int = 1,
    fps_target: float = 8.0,
    time_interval: int = 1000,
    shard_size: int = 1000,
    method: str = "optical_flow",  # for triplet selection: ssim|optical_flow
    max_frames: Optional[int] = None,
    # frame_pair extras
    pair_selection_method: str = "all",  # all | top_difference
    pair_difference_method: str = "optical_flow",  # optical_flow|pixel_diff|ssim
    n_pairs_per_video: Optional[int] = None,
    top_percent: float = 0.3,
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".webm", ".mkv", ".mov"),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    video_files = list_video_files(input_dir, video_exts)
    print(f"[convert] Found {len(video_files)} video files under: {input_dir}")

    shard_idx = 0
    sample_count = 0
    tar: Optional[tarfile.TarFile] = None

    def _open_new_shard():
        nonlocal tar, shard_idx
        if tar is not None:
            tar.close()
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.tar")
        tar = tarfile.open(shard_path, "w")
        shard_idx += 1

    try:
        for video_path in tqdm(video_files, desc="Converting"):
            try:
                frames, _ = extract_frames(
                    video_path,
                    sample_rate=sample_rate,
                    sample_mode=sample_mode,
                    fps_target=fps_target,
                    time_interval=time_interval,
                    max_frames=max_frames,
                )
                if not frames:
                    print(f"[warn] No frames extracted: {video_path}")
                    continue

                base_id = _safe_sample_id(video_path)

                if dataset_type == "triplet":
                    # select 3 representative frames
                    use_method = method
                    if use_method == "ssim" and (not _HAS_SKIMAGE or ssim is None):
                        print("[warn] scikit-image not available; fallback to optical_flow for triplet selection.")
                        use_method = "optical_flow"

                    if use_method == "ssim":
                        idxs = select_frames_by_ssim(frames, n_frames=3)
                    else:
                        idxs = select_frames_by_optical_flow(frames, n_frames=3)

                    if sample_count % shard_size == 0:
                        _open_new_shard()

                    with tempfile.TemporaryDirectory() as td:
                        rel_files = create_triplet_entry(td, frames, idxs, video_path, base_id)
                        assert tar is not None
                        add_to_tar(tar, td, rel_files)

                    sample_count += 1

                elif dataset_type == "full_frames":
                    # write many frame_*.jpg for this video into one sample folder
                    if sample_count % shard_size == 0:
                        _open_new_shard()

                    with tempfile.TemporaryDirectory() as td:
                        rel_files = create_full_frames_entry(td, frames, video_path, base_id)
                        assert tar is not None
                        add_to_tar(tar, td, rel_files)

                    sample_count += 1

                elif dataset_type == "frame_pair":
                    # create multiple samples per video
                    if len(frames) < 2:
                        continue

                    if pair_selection_method == "all":
                        pairs = [(i, i + 1) for i in range(len(frames) - 1)]
                    else:
                        pairs = select_frame_pairs_by_difference(
                            frames,
                            n_pairs=n_pairs_per_video,
                            method=pair_difference_method,
                            top_percent=top_percent,
                        )
                        if not pairs:
                            continue

                    for (a, b) in pairs:
                        if sample_count % shard_size == 0:
                            _open_new_shard()

                        pair_id = f"{base_id}_pair_{a:06d}_{b:06d}"
                        with tempfile.TemporaryDirectory() as td:
                            rel_files = create_frame_pair_entry(td, frames[a], frames[b], video_path, pair_id)
                            assert tar is not None
                            add_to_tar(tar, td, rel_files)

                        sample_count += 1

                else:
                    raise ValueError(f"Unknown dataset_type: {dataset_type}")

            except Exception as e:
                print(f"[error] Failed processing {video_path}: {e}")
                continue

    finally:
        if tar is not None:
            tar.close()

    print(f"[done] Created {sample_count} samples across {shard_idx} tar shards in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HMDB51 (or any folder of videos) into WebDataset tar shards."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing videos (e.g., hmdb51/walk)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WebDataset shards (*.tar)")

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="triplet",
        choices=["triplet", "frame_pair", "full_frames"],
        help="triplet: 3 representative frames per video; frame_pair: prev/next pairs; full_frames: many frame_*.jpg per video",
    )

    parser.add_argument("--sample_mode", type=str, default="fps", choices=["fixed", "fps", "fixed_time"])
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--fps_target", type=float, default=8.0)
    parser.add_argument("--time_interval", type=int, default=1000)
    parser.add_argument("--max_frames", type=int, default=None, help="Max extracted frames per video (toy speed-up). None = no limit")

    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--method", type=str, default="optical_flow", choices=["ssim", "optical_flow"],
                        help="Representative-frame selection method for triplet")

    # frame_pair specifics
    parser.add_argument("--pair_selection_method", type=str, default="all", choices=["all", "top_difference"])
    parser.add_argument("--pair_difference_method", type=str, default="optical_flow",
                        choices=["optical_flow", "pixel_diff", "ssim"])
    parser.add_argument("--n_pairs_per_video", type=int, default=None)
    parser.add_argument("--top_percent", type=float, default=0.3)

    args = parser.parse_args()

    convert_videos_to_webdataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        sample_mode=args.sample_mode,
        sample_rate=args.sample_rate,
        fps_target=args.fps_target,
        time_interval=args.time_interval,
        shard_size=args.shard_size,
        method=args.method,
        max_frames=args.max_frames if args.max_frames is not None else None,
        pair_selection_method=args.pair_selection_method,
        pair_difference_method=args.pair_difference_method,
        n_pairs_per_video=args.n_pairs_per_video,
        top_percent=args.top_percent,
    )


if __name__ == "__main__":
    main()
