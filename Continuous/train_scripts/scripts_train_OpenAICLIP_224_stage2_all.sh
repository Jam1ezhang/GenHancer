export AE="/data/zby/GenHancer/Continuous/src/flux/FLUX.1-dev/ae.safetensors"

# accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_use2frames_nextpredic_stage2_all.py --config "train_configs/test_OpenAICLIP_224_stage2_all.yaml"
accelerate launch --config_file "train_configs/accelerate_config.yaml" train_OpenAICLIP_sliding_windows_nextpredic_stage2_all.py --config "train_configs/test_OpenAICLIP_224_stage2_all_sliding_window.yaml"
