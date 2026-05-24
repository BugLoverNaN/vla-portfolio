#!/bin/bash
# M5 Phase 2a: Minimal smoke test (50 steps) - schema compatibility check
# Goal: detect tensor shape mismatch BEFORE committing to 20K steps
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

lerobot-train \
  --policy.path=/root/autodl-tmp/vla/models/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-tmp/vla/datasets/libero/libero \
  --dataset.video_backend=pyav \
  --dataset.episodes='[1261, 1262, 1263, 1264, 1265]' \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=4 \
  --steps=50 \
  --save_freq=50 \
  --log_freq=10 \
  --output_dir=/root/autodl-tmp/vla/outputs/libero_smoke_50 \
  --job_name=libero_smoke_50 \
  --wandb.enable=false \
  2>&1 | tee /root/autodl-tmp/vla/outputs/libero_smoke_50.log
