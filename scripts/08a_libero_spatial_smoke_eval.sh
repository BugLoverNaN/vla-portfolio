#!/bin/bash
# M5 Phase 4a: Smoke evaluation (3 episodes) with rename_map
set -e

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/hf_cache

lerobot-eval \
  --policy.path=/root/autodl-tmp/vla/outputs/libero_spatial_m5/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=3 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir=/root/autodl-tmp/vla/eval_results/libero_spatial_smoke \
  2>&1 | tee /root/autodl-tmp/vla/outputs/libero_spatial_smoke_eval.log
