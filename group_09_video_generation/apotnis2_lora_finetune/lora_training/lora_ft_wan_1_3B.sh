# conda activate diffsynth
module load cuda/12.6

accelerate launch DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --dataset_base_path uiuc_south_quad \
  --dataset_metadata_path uiuc_south_quad/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 8 \
  --model_paths '[
    [
        "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "models/Wan-AI/Wan2.2-TI2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.2-TI2V-1.3B/Wan2.2_VAE.pth"
]' \
  --learning_rate 1e-5 \
  --num_epochs 15 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "models/train/Wan2.2-TI2V-1.3B_lora" \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 
#  --lora_checkpoint "models/train/Wan2.2-TI2V-1.3B_lora/epoch-<epoch_number>.safetensors" 

