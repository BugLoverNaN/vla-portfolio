#!/bin/bash
# M5b Full Training: 40K steps with batch=16 num_workers=8
# Config validated via A/B/C smoke test (data_s 0.36s, step/s 1.67)
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=true

EPISODES=$(python -c "import json; print(json.dumps(list(range(0, 1693))))")

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
  --batch_size=16 \
  --num_workers=8 \
  --steps=40000 \
  --save_freq=5000 \
  --log_freq=200 \
  --output_dir=/root/autodl-tmp/vla/outputs/libero_4suites_m5b \
  --job_name=libero_4suites_m5b \
  --wandb.enable=false \
  2>&1 | tee /root/autodl-tmp/vla/outputs/libero_4suites_train.log
