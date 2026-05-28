#!/bin/bash
# M5b Final Pipeline: train -> eval 4 suites -> auto-shutdown
# SKIPS smoke test (already validated manually)
# Estimated: train ~6.6h + eval ~6.6h = ~13-14h total
set -e

PIPELINE_LOG=/root/autodl-tmp/vla/outputs/m5b_pipeline_final.log
exec > >(tee -a $PIPELINE_LOG) 2>&1

echo ""
echo "=================================================="
echo "M5b Final Pipeline Started at $(date)"
echo "Config: batch=16 num_workers=8 (smoke-validated)"
echo "Expected: train ~6.6h + eval ~6.6h = ~13-14h"
echo "=================================================="

# === Phase 1: Full training (~6.6h) ===
echo ""
echo ">>> Phase 1/2: Training starting at $(date)"
bash /root/autodl-tmp/vla/scripts/11_libero_4suites_train.sh
echo ""
echo ">>> Training finished at $(date)"

# Verify checkpoint exists
CKPT=/root/autodl-tmp/vla/outputs/libero_4suites_m5b/checkpoints/last/pretrained_model
if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "ERROR: Checkpoint not found at $CKPT, aborting eval"
  exit 1
fi
echo ">>> Checkpoint OK: $CKPT"

# === Phase 2: Evaluation (~6.6h, 4 suites) ===
echo ""
echo ">>> Phase 2/2: Evaluation starting at $(date)"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/hf_cache

# Eval each suite (|| true so one failure doesn't kill the whole pipeline)
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo ""
  echo ">>> Evaluating $SUITE at $(date)"
  
  lerobot-eval \
    --policy.path=$CKPT \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=$SUITE \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --output_dir=/root/autodl-tmp/vla/eval_results/m5b_$SUITE \
    || echo "WARNING: $SUITE eval failed but continuing"
  
  echo ">>> $SUITE done at $(date)"
done

# === Final summary ===
echo ""
echo "=================================================="
echo "M5b Pipeline Finished at $(date)"
echo "=================================================="
echo ""
echo "Results summary:"
TOTAL_RATE=0
COUNT=0
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  EVAL_JSON=/root/autodl-tmp/vla/eval_results/m5b_$SUITE/eval_info.json
  if [ -f "$EVAL_JSON" ]; then
    RATE=$(python -c "import json; data=json.load(open('$EVAL_JSON')); print(f\"{data['overall']['pc_success']:.1f}\")" 2>/dev/null || echo "N/A")
    echo ">>> $SUITE: $RATE%"
  else
    echo ">>> $SUITE: MISSING"
  fi
done

# === Auto-shutdown after 5 min grace ===
echo ""
echo ">>> Auto-shutdown in 5 minutes at $(date)"
echo ">>> If you want to cancel: ssh in and run 'sudo shutdown -c'"
sleep 300
/usr/bin/shutdown -h now
