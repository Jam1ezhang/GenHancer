export AE="/home/user/gptdata/zym/codespace_hallucination/ckpts/FLUX.1-dev/ae.safetensors"

accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_stage1.py --config "train_configs/test_OpenAICLIP_336_stage1.yaml"
