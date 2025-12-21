export AE="/data/zby/GenHancer/Continuous/src/flux/FLUX.1-dev/ae.safetensors"

# accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_nextpredic_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1.yaml"
# accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_sliding_windows_nextpredic_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1_sliding_window.yaml"
accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_indefinite_length_sliding_windows_nextpredic_stage1.py --config "train_configs/test_OpenAICLIP_224_stage1_indefinite_length_sliding_window.yaml"