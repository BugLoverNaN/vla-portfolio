#!/bin/bash
# M5 Phase 2b: Full smoke test (500 steps) - confirm stable learning
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Generate full episode list (1261-1692, 432 episodes)
EPISODES=$(python -c "import json; print(json.dumps(list(range(1261, 1693))))")

lerobot-train \
  --policy.path=/root/autodl-tmp/vla/models/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-tmp/vla/datasets/libero/libero \
  --dataset.video_backend=pyav \
  --dataset.episodes="$EPISODES" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size=8 \
  --steps=500 \
  --save_freq=500 \
  --log_freq=50 \
  --output_dir=/root/autodl-tmp/vla/outputs/libero_spatial_smoke_500 \
  --job_name=libero_spatial_smoke_500 \
  --wandb.enable=false \
  2>&1 | tee /root/autodl-tmp/vla/outputs/libero_spatial_smoke_500.log
