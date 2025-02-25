#!/bin/bash

accelerate launch --multi_gpu --config_file training/scripts/multi_gpu.yaml training/train_accelerate.py \
  --pretrained_model_name_or_path "/root/autodl-tmp/ckpt_base/stable-diffusion-2" \
  --modality "seg" \
  --noise_type "gaussian" \
  --max_train_steps 20000 \
  --checkpointing_steps 1000 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "/root/autodl-tmp/diff_based_seg_lotus" \
  --enable_xformers_memory_efficient_attention \
  --resume_from_checkpoint "latest" \
  "$@" 2>&1 | tee output.log