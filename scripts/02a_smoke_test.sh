#!/bin/bash
# ============================================================
# Smoke Test: 500 steps to validate the entire training pipeline
# Fixes:
#   - rename_map: dataset 'up'/'side' -> SmolVLA 'camera1'/'camera2'
#   - empty_cameras=1: pad missing camera3 with zeros
#   - video_backend=pyav: avoid torchcodec/PyTorch 2.8 ABI mismatch
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
  --steps=500 \
  --save_freq=500 \
  --log_freq=50 \
  --output_dir=/root/autodl-tmp/vla/outputs/smoke_test \
  --job_name=smoke_test \
  --wandb.enable=false \
  2>&1 | tee /root/autodl-tmp/vla/outputs/smoke_test.log
