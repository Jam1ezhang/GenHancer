# 帧对(Frame Pair)数据集构建使用指南

## 概述

现在 `convert_webm_to_webdataset.py` 支持两种数据集模式：

1. **Triplet模式**：从每个视频选择3个代表性帧（原有功能）
2. **Frame Pair模式**：创建连续的帧对用于预测下一帧（新功能）

## Frame Pair模式特性

### 帧对选择方法

#### 1. `all` - 使用所有连续帧对（默认）
- 从视频中提取的所有帧创建连续的帧对
- 适用于需要大量训练数据的场景

#### 2. `top_difference` - 选择变化最大的帧对
- 自动筛选出帧间变化最显著的样本
- 提高数据质量，减少冗余
- 支持三种差异计算方法：
  - **optical_flow**：光流法，计算运动强度（默认，推荐）
  - **pixel_diff**：像素差异法，计算像素级别的MSE
  - **ssim**：结构相似性法，基于图像结构的差异

## 使用示例

### 示例 1：使用所有连续帧对

```bash
python convert_webm_to_webdataset.py \
    --input_dir /home/user/gptdata/zym/codespace_hallucination/video_data/Something-Something_V2/data/extracted_data/20bn-something-something-v2 \
    --output_dir /home/user/gptdata/zym/codespace_hallucination/video_data/webdataset/sth2sth \
    --dataset_type frame_pair \
    --sample_rate 30 \
    --sample_mode fixed \
    --pair_selection_method all \
    --shard_size 1000
```

### 示例 2：选择变化最大的前30%帧对（光流法）

```bash
python convert_webm_to_webdataset.py \
    --input_dir /home/user/gptdata/zym/codespace_hallucination/video_data/Something-Something_V2/data/extracted_data/20bn-something-something-v2 \
    --output_dir /home/user/gptdata/zym/codespace_hallucination/video_data/webdataset/sth2sth \
    --dataset_type frame_pair \
    --sample_rate 10 \
    --sample_mode fixed \
    --pair_selection_method top_difference \
    --pair_difference_method optical_flow \
    --top_percent 0.4 \
    --shard_size 1000
```

### 示例 3：每个视频选择固定数量的帧对（像素差异法）

```bash
python convert_webm_to_webdataset.py \
    --input_dir /path/to/webm/videos \
    --output_dir /path/to/output \
    --dataset_type frame_pair \
    --sample_rate 5 \
    --sample_mode fixed \
    --pair_selection_method top_difference \
    --pair_difference_method pixel_diff \
    --n_pairs_per_video 50 \
    --shard_size 1000
```

### 示例 4：基于FPS采样 + SSIM差异选择

```bash
python convert_webm_to_webdataset.py \
    --input_dir /path/to/webm/videos \
    --output_dir /path/to/output \
    --dataset_type frame_pair \
    --sample_mode fps \
    --fps_target 2.0 \
    --pair_selection_method top_difference \
    --pair_difference_method ssim \
    --top_percent 0.5 \
    --shard_size 500
```

## 参数说明

### 基础参数
- `--dataset_type`: 数据集类型（triplet / frame_pair）
- `--input_dir`: 输入视频目录
- `--output_dir`: 输出数据集目录
- `--shard_size`: 每个tar文件包含的样本数

### 采样参数
- `--sample_mode`: 采样模式（fixed / fps / fixed_time）
- `--sample_rate`: 固定帧间隔采样率
- `--fps_target`: 目标FPS（sample_mode=fps时）
- `--time_interval`: 时间间隔毫秒数（sample_mode=fixed_time时）

### Frame Pair专用参数
- `--pair_selection_method`: 帧对选择方法
  - `all`: 使用所有连续帧对
  - `top_difference`: 选择变化最大的帧对
- `--pair_difference_method`: 差异计算方法（top_difference时使用）
  - `optical_flow`: 光流法（推荐）
  - `pixel_diff`: 像素差异法
  - `ssim`: 结构相似性法
- `--n_pairs_per_video`: 每个视频选择的帧对数量（优先级高）
- `--top_percent`: 选择差异最大的前百分比（n_pairs_per_video为None时使用）

## 输出格式

每个帧对样本包含以下文件：

```
sample_folder/
├── frame_prev.jpg  # 输入帧（前一帧）
├── frame_next.jpg  # 目标帧（后一帧）
└── txt             # 文本描述
```

样本ID格式：`{视频名}_pair_{前帧索引}_{后帧索引}`

## 最佳实践建议

1. **数据量充足时**：使用 `--pair_selection_method all` 获取最多的训练样本

2. **需要高质量数据时**：使用 `--pair_selection_method top_difference` 
   - 推荐使用 `optical_flow` 方法
   - `--top_percent 0.2-0.3` 通常是较好的选择

3. **平衡质量和数量**：
   - 先用较高的 `--sample_rate` 提取更多帧
   - 再用 `top_difference` 筛选出最有价值的帧对

4. **视频内容静态时**：降低 `--top_percent` 或使用固定 `--n_pairs_per_video`

5. **视频内容动态时**：可以适当提高 `--top_percent` 获取更多有效样本

## 性能提示

- **光流法**(`optical_flow`)：速度适中，效果最好，推荐使用
- **像素差异法**(`pixel_diff`)：速度最快，但可能对光照变化敏感
- **SSIM法**(`ssim`)：效果好，但计算较慢，需要安装 scikit-image

## 依赖环境

```bash
pip install opencv-python numpy tqdm
pip install scikit-image  # 如果使用SSIM方法
```

