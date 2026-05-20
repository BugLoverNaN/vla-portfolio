#!/bin/bash
# ============================================================
# M3: Full fine-tuning - SmolVLA on SO-101 PickPlace
# Steps: 20000, ~80 min on RTX 4090 D
# Expected: loss 0.5 -> 0.05-0.10
# ============================================================
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

lerobot-train \
  --policy.path=/root/autodl-tmp/vla/models/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.root=/root/autodl-tmp/vla/datasets/svla_so101_pickplace \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir=/root/autodl-tmp/vla/outputs/smolvla_so101_m3 \
  --job_name=smolvla_so101_m3 \
  --wandb.enable=false \
  2>&1 | tee /root/autodl-tmp/vla/outputs/smolvla_so101_m3_train.log
