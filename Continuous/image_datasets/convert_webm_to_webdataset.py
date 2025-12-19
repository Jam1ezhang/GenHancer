import os
import cv2
import numpy as np
import tarfile
import tempfile
# from skimage.metrics import structural_similarity as ssim
import uuid
import json
from tqdm import tqdm
import argparse


def extract_frames(video_path, sample_rate=1, sample_mode='fps', fps_target=1, time_interval=1000):
    """
    从视频中提取帧，支持多种采样模式
    
    参数:
    - video_path: 视频文件路径
    - sample_rate: 固定帧间隔采样时的采样率（每sample_rate帧提取一帧）
    - sample_mode: 采样模式，可选值：
        'fixed': 固定帧间隔（默认），使用sample_rate参数
        'fps': 基于FPS的采样，提取指定FPS的帧，使用fps_target参数
        'fixed_time': 固定时间间隔采样，使用time_interval参数（毫秒）
    - fps_target: 当sample_mode='fps'时，目标FPS值
    - time_interval: 当sample_mode='fixed_time'时，时间间隔（毫秒）
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_count = 0
    last_extracted_time = -time_interval  # 初始化上次提取时间
    
    # 获取视频的实际FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        should_extract = False
        
        if sample_mode == 'fixed':
            # 固定帧间隔采样
            should_extract = frame_count % sample_rate == 0
        elif sample_mode == 'fps':
            # 基于FPS的采样
            # 计算应该提取的帧间隔
            if video_fps > 0:
                frame_interval = max(1, int(video_fps / fps_target))
                should_extract = frame_count % frame_interval == 0
            else:
                # 如果无法获取视频FPS，回退到固定帧间隔
                should_extract = frame_count % sample_rate == 0
        elif sample_mode == 'fixed_time':
            # 固定时间间隔采样（毫秒）
            should_extract = current_time - last_extracted_time >= time_interval
        
        if should_extract:
            # 转换BGR为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(current_time)
            last_extracted_time = current_time
            
        frame_count += 1
    
    cap.release()
    return frames, timestamps


def select_frames_by_ssim(frames, n_frames=3):
    """
    使用SSIM方法选择最具代表性的n_frames个帧
    算法思路：
    1. 计算所有帧之间的SSIM差异矩阵
    2. 选择累积差异最大的帧，确保帧之间的视觉差异最大化
    """
    if len(frames) <= n_frames:
        return list(range(len(frames)))
    
    # 计算SSIM差异矩阵
    frame_count = len(frames)
    diff_matrix = np.zeros((frame_count, frame_count))
    
    for i in range(frame_count):
        for j in range(i+1, frame_count):
            # 将图像转换为灰度图以计算SSIM
            frame_i_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            frame_j_gray = cv2.cvtColor(frames[j], cv2.COLOR_RGB2GRAY)
            
            # 确保两个帧大小相同
            if frame_i_gray.shape != frame_j_gray.shape:
                frame_j_gray = cv2.resize(frame_j_gray, (frame_i_gray.shape[1], frame_i_gray.shape[0]))
            
            # 计算SSIM（结构相似性指数），值越接近1表示越相似
            ssim_val = ssim(frame_i_gray, frame_j_gray)
            # 转换为差异值（1-SSIM），值越大表示差异越大
            diff = 1 - ssim_val
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff
    
    # 选择累积差异最大的帧
    selected_indices = []
    remaining_indices = list(range(frame_count))
    
    # 首先选择第一个帧（通常是视频的开始）
    selected_indices.append(remaining_indices[0])
    remaining_indices.remove(0)
    
    # 然后选择与已选帧差异最大的帧
    while len(selected_indices) < n_frames and remaining_indices:
        max_total_diff = -1
        best_index = -1
        
        for idx in remaining_indices:
            total_diff = sum(diff_matrix[idx, sel_idx] for sel_idx in selected_indices)
            if total_diff > max_total_diff:
                max_total_diff = total_diff
                best_index = idx
        
        if best_index != -1:
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
    
    # 确保选择的帧按照时间顺序排序
    selected_indices.sort()
    
    return selected_indices

def select_frames_by_optical_flow(frames, n_frames=3):
    """
    使用光流法选择最具代表性的n_frames个帧
    算法思路：
    1. 计算连续帧之间的光流强度
    2. 选择光流变化最显著的帧，这些帧通常代表视频中的重要动态变化
    """
    if len(frames) <= n_frames:
        return list(range(len(frames)))
    
    # 计算相邻帧之间的光流强度
    flow_intensities = []
    prev_gray = None
    
    for frame in frames:
        # 转换为灰度图
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if prev_gray is not None:
            # 计算光流 (Farneback方法)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算光流强度（向量的大小）
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 计算平均光流强度作为帧间变化的度量
            mean_flow_intensity = np.mean(magnitude)
            flow_intensities.append(mean_flow_intensity)
        
        prev_gray = curr_gray
    
    # 在flow_intensities前面补0，使其长度与frames相同
    flow_intensities = [0] + flow_intensities
    
    # 计算累积光流能量
    cumulative_flow = np.zeros(len(flow_intensities))
    cumulative_flow[0] = flow_intensities[0]
    
    for i in range(1, len(flow_intensities)):
        cumulative_flow[i] = cumulative_flow[i-1] + flow_intensities[i]
    
    # 选择均匀分布但光流变化显著的帧
    selected_indices = [0]  # 总是选择第一帧
    remaining_indices = list(range(1, len(frames)-1))
    
    # 计算期望的累积流量间隔
    total_flow = cumulative_flow[-1]
    target_intervals = [total_flow * (i+1) / n_frames for i in range(n_frames-1)]
    
    # 找到最接近目标间隔的帧
    for target in target_intervals[:-1]:
        # 找到累积流量最接近目标的帧
        closest_idx = min(remaining_indices, key=lambda x: abs(cumulative_flow[x] - target))
        selected_indices.append(closest_idx)
        remaining_indices.remove(closest_idx)
    
    # 总是选择最后一帧
    selected_indices.append(len(frames)-1)
    
    # 确保选择的帧按照时间顺序排序
    selected_indices.sort()
    
    return selected_indices


def create_webdataset_entry(temp_dir, frames, selected_indices, video_path, sample_id):
    """
    在临时目录中创建webdataset条目的文件，使用样本文件夹结构
    """
    # 为每个样本创建唯一的键，作为文件夹名称
    sample_folder = sample_id
    
    # 在临时目录中创建样本文件夹
    sample_folder_path = os.path.join(temp_dir, sample_folder)
    os.makedirs(sample_folder_path, exist_ok=True)
    
    # 保存选定的帧，按照用户要求的格式命名
    frame_files = []
    for i, frame_idx in enumerate(selected_indices):
        frame_filename = f"frame_{i}.jpg"
        frame_path = os.path.join(sample_folder_path, frame_filename)
        # 将RGB帧转换回BGR以保存
        cv2.imwrite(frame_path, cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR))
        frame_files.append(frame_filename)
    
    # 创建文本描述文件，使用简单的'txt'作为文件名
    text_filename = "txt"
    text_path = os.path.join(sample_folder_path, text_filename)
    with open(text_path, 'w') as f:
        f.write(f"Video from {os.path.basename(video_path)}")
    
    # 记录所有文件的相对路径
    all_files = [os.path.join(sample_folder, f) for f in frame_files + [text_filename]]
    
    return sample_folder, all_files


def select_frame_pairs_by_difference(frames, n_pairs=None, method='optical_flow', top_percent=0.3):
    """
    选择帧间变化最大的帧对
    
    参数:
    - frames: 提取的所有帧列表
    - n_pairs: 要选择的帧对数量（如果为None，则使用top_percent）
    - method: 差异计算方法
        'optical_flow': 光流法，计算运动强度
        'pixel_diff': 像素差异法，计算像素级别的差异
        'ssim': SSIM法，计算结构相似性差异
    - top_percent: 当n_pairs为None时，选择差异最大的前百分比（0-1之间）
    
    返回:
    - selected_pairs: 选中的帧对索引列表，每个元素是(prev_idx, next_idx)元组
    """
    if len(frames) < 2:
        return []
    
    # 计算所有连续帧对的差异
    pair_differences = []
    
    for i in range(len(frames) - 1):
        frame_prev = frames[i]
        frame_next = frames[i + 1]
        
        if method == 'optical_flow':
            # 使用光流法计算帧间差异
            prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算光流强度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_flow = np.mean(magnitude)
            pair_differences.append((i, i + 1, mean_flow))
            
        elif method == 'pixel_diff':
            # 使用像素差异法（均方误差）
            diff = np.mean((frame_prev.astype(np.float32) - frame_next.astype(np.float32)) ** 2)
            pair_differences.append((i, i + 1, diff))
            
        elif method == 'ssim':
            # 使用SSIM差异法
            prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY)
            
            # 确保两个帧大小相同
            if prev_gray.shape != next_gray.shape:
                next_gray = cv2.resize(next_gray, (prev_gray.shape[1], prev_gray.shape[0]))
            
            # 计算SSIM，转换为差异值
            ssim_val = ssim(prev_gray, next_gray)
            diff = 1 - ssim_val
            pair_differences.append((i, i + 1, diff))
    
    # 按差异值降序排序
    pair_differences.sort(key=lambda x: x[2], reverse=True)
    
    # 确定要选择的帧对数量
    if n_pairs is None:
        n_pairs = max(1, int(len(pair_differences) * top_percent))
    else:
        n_pairs = min(n_pairs, len(pair_differences))
    
    # 选择差异最大的前n_pairs个帧对
    selected_pairs = [(prev_idx, next_idx) for prev_idx, next_idx, _ in pair_differences[:n_pairs]]
    
    # 按时间顺序排序
    selected_pairs.sort(key=lambda x: x[0])
    
    return selected_pairs


def create_frame_pair_entry(temp_dir, frame_prev, frame_next, video_path, sample_id):
    """
    在临时目录中创建帧对webdataset条目的文件
    用于通过上一帧预测下一帧的任务
    
    参数:
    - temp_dir: 临时目录路径
    - frame_prev: 前一帧（用于输入）
    - frame_next: 后一帧（用于预测目标）
    - video_path: 源视频路径
    - sample_id: 样本ID
    """
    # 为每个样本创建唯一的键，作为文件夹名称
    sample_folder = sample_id
    
    # 在临时目录中创建样本文件夹
    sample_folder_path = os.path.join(temp_dir, sample_folder)
    os.makedirs(sample_folder_path, exist_ok=True)
    
    # 保存前一帧（输入）
    prev_frame_filename = "frame_prev.jpg"
    prev_frame_path = os.path.join(sample_folder_path, prev_frame_filename)
    cv2.imwrite(prev_frame_path, cv2.cvtColor(frame_prev, cv2.COLOR_RGB2BGR))
    
    # 保存后一帧（目标）
    next_frame_filename = "frame_next.jpg"
    next_frame_path = os.path.join(sample_folder_path, next_frame_filename)
    cv2.imwrite(next_frame_path, cv2.cvtColor(frame_next, cv2.COLOR_RGB2BGR))
    
    # 创建文本描述文件
    text_filename = "txt"
    text_path = os.path.join(sample_folder_path, text_filename)
    with open(text_path, 'w') as f:
        f.write(f"Frame pair from {os.path.basename(video_path)}")
    
    # 记录所有文件的相对路径
    frame_files = [prev_frame_filename, next_frame_filename]
    all_files = [os.path.join(sample_folder, f) for f in frame_files + [text_filename]]
    
    return sample_folder, all_files


def add_to_tar(tar_file, temp_dir, files, sample_key):
    """
    将文件添加到tar归档中，保持样本文件夹结构
    """
    for file_rel_path in files:
        # 获取文件的绝对路径
        file_abs_path = os.path.join(temp_dir, file_rel_path)
        
        # 直接使用文件的相对路径作为tar归档中的路径，保持文件夹结构
        # 这里file_rel_path已经包含了样本文件夹，如'sample1/frame_0.jpg'
        tar_file.add(file_abs_path, arcname=file_rel_path)


def convert_webm_to_webdataset(input_dir, output_dir, sample_rate=1, sample_mode='fixed', 
                               fps_target=1, time_interval=1000, shard_size=1000, method='ssim'):
    """
    转换input_dir中的所有webm文件为webdataset格式（三元组模式）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有webm文件
    webm_files = [f for f in os.listdir(input_dir) if f.endswith('.webm')]
    print(f"找到 {len(webm_files)} 个webm文件")
    
    # 创建tar文件（shard）
    shard_idx = 0
    sample_count = 0
    tar = None
    
    try:
        for i, video_file in enumerate(tqdm(webm_files)):
            # 创建新的tar shard（如果需要）
            if sample_count % shard_size == 0:
                if tar is not None:
                    tar.close()
                
                shard_filename = f"shard_{shard_idx:05d}.tar"
                shard_path = os.path.join(output_dir, shard_filename)
                tar = tarfile.open(shard_path, "w")
                shard_idx += 1
            
            # 处理单个视频
            video_path = os.path.join(input_dir, video_file)
            try:
                # 提取帧
                frames, _ = extract_frames(video_path, sample_rate, sample_mode, fps_target, time_interval)
                
                if not frames:
                    print(f"警告：无法从 {video_file} 提取帧")
                    continue
                
                # 根据选择的方法选择代表性帧
                if method == 'ssim':
                    selected_indices = select_frames_by_ssim(frames, n_frames=3)
                elif method == 'optical_flow':
                    selected_indices = select_frames_by_optical_flow(frames, n_frames=3)
                
                # 使用视频文件名（不包括扩展名）作为样本ID
                # 移除扩展名并替换可能导致问题的字符
                sample_id = os.path.splitext(video_file)[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                # 使用临时目录创建webdataset条目
                with tempfile.TemporaryDirectory() as temp_dir:
                    sample_key, files = create_webdataset_entry(
                        temp_dir, frames, selected_indices, video_path, sample_id
                    )
                    
                    # 添加到tar文件
                    add_to_tar(tar, temp_dir, files, sample_key)
                
                sample_count += 1
                
            except Exception as e:
                print(f"处理 {video_file} 时出错: {e}")
                continue
    finally:
        if tar is not None:
            tar.close()
    
    print(f"处理完成！创建了 {sample_count} 个样本，分布在 {shard_idx} 个tar文件中。")


def convert_webm_to_frame_pairs(input_dir, output_dir, sample_rate=1, sample_mode='fixed', 
                                 fps_target=1, time_interval=1000, shard_size=1000,
                                 pair_selection_method='all', pair_difference_method='optical_flow',
                                 n_pairs_per_video=None, top_percent=0.3):
    """
    转换input_dir中的所有webm文件为帧对(frame pair)数据集格式
    每个样本包含两个连续的帧：前一帧用于输入，后一帧用于预测
    
    参数:
    - input_dir: 包含webm文件的输入目录
    - output_dir: WebDataset输出目录
    - sample_rate: 固定帧间隔采样时的采样率
    - sample_mode: 采样模式（'fixed', 'fps', 'fixed_time'）
    - fps_target: 目标FPS值（当sample_mode='fps'时）
    - time_interval: 时间间隔（当sample_mode='fixed_time'时，单位：毫秒）
    - shard_size: 每个tar文件中的样本数
    - pair_selection_method: 帧对选择方法
        'all': 使用所有连续帧对
        'top_difference': 选择变化最大的帧对
    - pair_difference_method: 帧对差异计算方法（当pair_selection_method='top_difference'时）
        'optical_flow': 光流法
        'pixel_diff': 像素差异法
        'ssim': SSIM法
    - n_pairs_per_video: 每个视频选择的帧对数量（None表示使用top_percent）
    - top_percent: 当n_pairs_per_video为None时，选择差异最大的前百分比（0-1之间）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有webm文件
    webm_files = [f for f in os.listdir(input_dir) if f.endswith('.webm')]
    print(f"找到 {len(webm_files)} 个webm文件")
    
    # 创建tar文件（shard）
    shard_idx = 0
    sample_count = 0
    tar = None
    
    try:
        for video_file in tqdm(webm_files, desc="处理视频"):
            video_path = os.path.join(input_dir, video_file)
            
            try:
                # 提取帧
                frames, _ = extract_frames(video_path, sample_rate, sample_mode, fps_target, time_interval)
                
                if len(frames) < 2:
                    print(f"警告：{video_file} 提取的帧数少于2，跳过")
                    continue
                
                # 根据选择方法确定要使用的帧对
                if pair_selection_method == 'all':
                    # 使用所有连续帧对
                    selected_pairs = [(i, i + 1) for i in range(len(frames) - 1)]
                elif pair_selection_method == 'top_difference':
                    # 选择变化最大的帧对
                    selected_pairs = select_frame_pairs_by_difference(
                        frames, 
                        n_pairs=n_pairs_per_video,
                        method=pair_difference_method,
                        top_percent=top_percent
                    )
                    if not selected_pairs:
                        print(f"警告：无法从 {video_file} 选择帧对，跳过")
                        continue
                
                # 为每个选中的帧对创建样本
                for pair_idx, (prev_idx, next_idx) in enumerate(selected_pairs):
                    # 创建新的tar shard（如果需要）
                    if sample_count % shard_size == 0:
                        if tar is not None:
                            tar.close()
                        
                        shard_filename = f"shard_{shard_idx:05d}.tar"
                        shard_path = os.path.join(output_dir, shard_filename)
                        tar = tarfile.open(shard_path, "w")
                        shard_idx += 1
                    
                    # 获取前一帧和后一帧
                    frame_prev = frames[prev_idx]
                    frame_next = frames[next_idx]
                    
                    # 创建样本ID：视频名称_帧索引
                    base_name = os.path.splitext(video_file)[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    sample_id = f"{base_name}_pair_{prev_idx:04d}_{next_idx:04d}"
                    
                    # 使用临时目录创建webdataset条目
                    with tempfile.TemporaryDirectory() as temp_dir:
                        sample_key, files = create_frame_pair_entry(
                            temp_dir, frame_prev, frame_next, video_path, sample_id
                        )
                        
                        # 添加到tar文件
                        add_to_tar(tar, temp_dir, files, sample_key)
                    
                    sample_count += 1
                
            except Exception as e:
                print(f"处理 {video_file} 时出错: {e}")
                continue
    
    finally:
        if tar is not None:
            tar.close()
    
    print(f"处理完成！创建了 {sample_count} 个帧对样本，分布在 {shard_idx} 个tar文件中。")


def main():
    parser = argparse.ArgumentParser(description='将webm视频转换为WebDataset格式')
    parser.add_argument('--input_dir', type=str, required=True, help='包含webm文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='WebDataset输出目录')
    parser.add_argument('--dataset_type', type=str, default='triplet', choices=['triplet', 'frame_pair'],
                        help='数据集类型：triplet（三元组，3个代表性帧）或 frame_pair（帧对，用于预测下一帧）')
    parser.add_argument('--sample_rate', type=int, default=1, help='固定帧间隔采样时的采样率')
    parser.add_argument('--sample_mode', type=str, default='fixed', choices=['fixed', 'fps', 'fixed_time'],
                        help='采样模式：fixed（固定帧间隔）、fps（基于FPS）、fixed_time（固定时间间隔）')
    parser.add_argument('--fps_target', type=float, default=1.0, help='当sample_mode=fps时的目标FPS值')
    parser.add_argument('--time_interval', type=int, default=1000, help='当sample_mode=fixed_time时的时间间隔（毫秒）')
    parser.add_argument('--shard_size', type=int, default=1000, help='每个tar文件中的样本数')
    parser.add_argument('--method', type=str, default='ssim', choices=['ssim', 'optical_flow'], 
                        help='帧选择方法（仅用于triplet模式）：ssim（结构相似性）或 optical_flow（光流法）')
    
    # frame_pair模式特有的参数
    parser.add_argument('--pair_selection_method', type=str, default='all', choices=['all', 'top_difference'],
                        help='帧对选择方法（仅用于frame_pair模式）：all（使用所有连续帧对）或 top_difference（选择变化最大的帧对）')
    parser.add_argument('--pair_difference_method', type=str, default='optical_flow', 
                        choices=['optical_flow', 'pixel_diff', 'ssim'],
                        help='帧对差异计算方法（仅用于frame_pair模式的top_difference选择）：optical_flow（光流法）、pixel_diff（像素差异）或 ssim（结构相似性）')
    parser.add_argument('--n_pairs_per_video', type=int, default=None,
                        help='每个视频选择的帧对数量（仅用于frame_pair模式的top_difference选择，None表示使用top_percent）')
    parser.add_argument('--top_percent', type=float, default=0.3,
                        help='选择差异最大的前百分比（仅用于frame_pair模式的top_difference选择，当n_pairs_per_video为None时使用，取值范围0-1）')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'triplet':
        print("使用三元组模式：从每个视频选择3个代表性帧")
        convert_webm_to_webdataset(
            args.input_dir, args.output_dir, 
            sample_rate=args.sample_rate, 
            sample_mode=args.sample_mode,
            fps_target=args.fps_target,
            time_interval=args.time_interval,
            shard_size=args.shard_size, 
            method=args.method
        )
    elif args.dataset_type == 'frame_pair':
        selection_info = f"使用帧对模式：创建连续的帧对用于预测下一帧"
        if args.pair_selection_method == 'top_difference':
            if args.n_pairs_per_video:
                selection_info += f" (每个视频选择变化最大的{args.n_pairs_per_video}个帧对，方法：{args.pair_difference_method})"
            else:
                selection_info += f" (选择变化最大的前{args.top_percent*100:.0f}%帧对，方法：{args.pair_difference_method})"
        else:
            selection_info += " (使用所有连续帧对)"
        print(selection_info)
        
        convert_webm_to_frame_pairs(
            args.input_dir, args.output_dir, 
            sample_rate=args.sample_rate, 
            sample_mode=args.sample_mode,
            fps_target=args.fps_target,
            time_interval=args.time_interval,
            shard_size=args.shard_size,
            pair_selection_method=args.pair_selection_method,
            pair_difference_method=args.pair_difference_method,
            n_pairs_per_video=args.n_pairs_per_video,
            top_percent=args.top_percent
        )


if __name__ == "__main__":
    main()