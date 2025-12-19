# GenHancer Video Mode Training Guide

This document provides a comprehensive guide for training GenHancer in video mode, which enhances CLIP visual representations using video data through a two-stage post-training strategy.

## 📋 Overview

Video mode training follows the same two-stage approach as image mode but is optimized for video data:

1. **Stage 1**: Initial training to adapt CLIP to video data
2. **Stage 2**: Fine-tuning with LoRA (Low-Rank Adaptation) for parameter-efficient enhancement

## 📁 Directory Structure

```
/Continuous/
├── clip_models/          # CLIP model implementations
├── image_datasets/       # Video dataset loading utilities
├── output_*              # Training output directories (auto-generated)
├── reconstruction/       # Reconstruction utilities
├── src/                  # Core model implementations
├── train_configs/        # Training configuration files
├── train_scripts/        # Training shell scripts
└── README.md             # This document
```

## 🚀 Quick Start

### Prerequisites

1. Download and prepare the video dataset (default: Something-Something V2)
2. Download the FLUX.1-dev autoencoder checkpoint
3. Set up the environment with required dependencies

### Stage 1 Training

```bash
# Navigate to the Continuous directory
cd /home/user/gptdata/zym/codespace_hallucination/GenHancer/Continuous

# Run Stage 1 training
bash train_scripts/scripts_train_OpenAICLIP_336_video_stage1.sh
```

### Stage 2 Training

```bash
# Run Stage 2 training (after Stage 1 completes)
bash train_scripts/scripts_train_OpenAICLIP_336_video_stage2_all.sh
```

## ⚙️ Configuration Details

### Dataset Configuration

The video mode currently supports the [Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something) dataset. To use a different dataset, modify the `video_dir` parameter in the configuration files.

### Stage 1 Configuration

**File**: `train_configs/test_OpenAICLIP_336_video_stage1.yaml`

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | Generative model name | `"flux-dev"` |
| `data_config.train_batch_size` | Training batch size | 32 |
| `data_config.num_workers` | Data loader workers | 0 |
| `data_config.img_size` | Input image size | 336 |
| `data_config.video_dir` | Video dataset directory | `/home/user/data/Sth2Sth` |
| `clip_config.clip_image_size` | CLIP input size | 336 |
| `output_dir` | Output directory | `output_OpenAICLIP_336_video_stage1_626/` |
| `max_train_steps` | Maximum training steps | 626 |
| `learning_rate` | Learning rate | 1e-4 |
| `checkpointing_steps` | Steps between checkpoints | 313 |

### Stage 2 Configuration

**File**: `train_configs/test_OpenAICLIP_336_video_stage2_all.yaml`

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `load_dir` | Stage 1 checkpoint directory | `output_OpenAICLIP_336_video_stage1_313/` |
| `load_step` | Stage 1 checkpoint step | 313 |
| `output_dir` | Output directory | `output_OpenAICLIP_336_video_stage2_all_load313/` |
| `max_train_steps` | Maximum training steps | 1000 |
| `learning_rate` | Learning rate (smaller for fine-tuning) | 1e-5 |
| `lora_config.r` | LoRA rank | 16 |
| `lora_config.lora_alpha` | LoRA alpha | 16 |
| `lora_config.lora_dropout` | LoRA dropout | 0.1 |

## 📊 Training Output

### Checkpoint Structure

Training outputs are saved in directories specified by `output_dir`:

```
/output_OpenAICLIP_336_video_stage1_626/
├── checkpoint-dit-313.bin       # DIT model checkpoint at step 313
├── checkpoint-project-clip-313.bin  # CLIP model checkpoint at step 313
├── checkpoint-visual-adapter-313.bin  # Visual adapter checkpoint at step 313
├── optimizer-state-313.bin      # Optimizer state at step 313
└── logs/                        # TensorBoard logs
```

```
/output_OpenAICLIP_336_video_stage2_all_load313/
├── clip-vit-large-patch14-336-100/  # Enhanced CLIP model at step 100
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── checkpoint-dit-100.bin       # DIT model checkpoint at step 100
├── checkpoint-project-clip-100.bin  # CLIP model checkpoint at step 100
├── checkpoint-visual-adapter-100.bin  # Visual adapter checkpoint at step 100
├── optimizer-state-100.bin      # Optimizer state at step 100
└── logs/                        # TensorBoard logs
```

### Logging

Training logs are saved in the `logs/` directory and can be viewed with TensorBoard:

```bash
tensorboard --logdir=output_OpenAICLIP_336_video_stage1_626/logs/
```

## 🔧 Environment Setup

### Autoencoder Configuration

The training scripts require a FLUX.1-dev autoencoder checkpoint. Set the path in the training scripts:

```bash
export AE="/path/to/flux.1-dev/ae.safetensors"
```

### NCCL Configuration

For distributed training, the scripts include optional NCCL environment variables to resolve GPU communication issues. Uncomment and modify as needed:

```bash
# export NCCL_TIMEOUT=7200  # 2-hour timeout
export NCCL_P2P_DISABLE=1  # Disable P2P communication if needed
export NCCL_DEBUG=WARN      # Set to INFO for detailed debugging
```

## 📏 Evaluation

After training, evaluate the enhanced CLIP model on the MMVP-VLM benchmark:

```bash
# Navigate to the project root
cd /home/user/gptdata/zym/codespace_hallucination/GenHancer

# Run evaluation
python evaluation/evaluate_mmvp_OpenAICLIP_336.py --benchmark_dir 'YOUR_MMVP_VLM_PATH' --vision_tower_name 'output_OpenAICLIP_336_video_stage2_all_load313/clip-vit-large-patch14-336-1000'
```

## 🎯 Key Features of Video Mode

1. **Video Dataset Support**: Optimized for action recognition datasets like Something-Something V2
2. **Two-Stage Training**: Initial adaptation followed by parameter-efficient fine-tuning
3. **LoRA Fine-Tuning**: Reduces computational cost while maintaining performance
4. **Flexible Configuration**: Easily adjustable parameters for different video datasets
5. **Checkpoint Management**: Regular checkpoints with configurable intervals

## 📈 Expected Results

Video mode training typically enhances CLIP's visual representations for action recognition tasks. The exact improvement depends on the dataset and training configuration.

## 🤔 Troubleshooting

### Common Issues

1. **NCCL Communication Errors**:
   - Uncomment and adjust NCCL environment variables in the training scripts
   - Try reducing `num_workers` in the configuration files

2. **Out of Memory Errors**:
   - Reduce `train_batch_size` in the configuration files
   - Increase `gradient_accumulation_steps` to maintain effective batch size

3. **Dataset Loading Issues**:
   - Ensure the video dataset path is correct
   - Verify the dataset format matches the expected structure

## 📜 License

The video mode training code follows the same [Apache 2 License](https://github.com/mashijie1028/Gen4Rep/blob/main/LICENSE) as the main GenHancer repository.

## 📚 Citation

If you use GenHancer's video mode in your research, please cite our paper:

```bibtex
@article{ma2025genhancer,
    title={GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers},
    author={Ma, Shijie and Ge, Yuying and Wang, Teng and Guo, Yuxin and Ge, Yixiao and Shan, Ying},
    journal={arXiv preprint arXiv:2503.19480},
    year={2025}
}
```

## 📧 Contact

For questions or issues related to video mode training, please contact:
- mashijie9817@gmail.com
