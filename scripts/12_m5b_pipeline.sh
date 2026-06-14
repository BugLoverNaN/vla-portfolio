#!/bin/bash
# M5b Full Pipeline: smoke -> train (40K) -> eval (4 suites) -> auto-shutdown
# Estimated: smoke 5min + train ~14h + eval ~6-7h = ~20-21h total
set -e

PIPELINE_LOG=/root/autodl-tmp/vla/outputs/m5b_pipeline.log
exec > >(tee -a $PIPELINE_LOG) 2>&1

echo ""
echo "=================================================="
echo "M5b Pipeline Started at $(date)"
echo "=================================================="

# === Phase 1: Smoke test (5 min) ===
echo ""
echo ">>> Phase 1/3: Smoke test starting at $(date)"
bash /root/autodl-tmp/vla/scripts/11a_libero_4suites_smoke.sh

# Sanity check
SMOKE_LOG=/root/autodl-tmp/vla/outputs/libero_4suites_smoke.log
SMOKE_FINAL_LOSS=$(grep "step:300" $SMOKE_LOG | tail -1 | grep -oP 'loss:\K[\d.]+' | head -1)
echo ""
echo ">>> Smoke final loss: $SMOKE_FINAL_LOSS"

if [ -z "$SMOKE_FINAL_LOSS" ]; then
  echo "ERROR: Could not parse smoke loss, aborting"
  exit 1
fi

LOSS_OK=$(python -c "print(1 if $SMOKE_FINAL_LOSS < 1.5 else 0)" 2>/dev/null || echo 0)
if [ "$LOSS_OK" != "1" ]; then
  echo "ERROR: Smoke loss too high ($SMOKE_FINAL_LOSS), aborting"
  exit 1
fi
echo ">>> Smoke OK, proceeding to training"

# === Phase 2: Full training (~14h) ===
echo ""
echo ">>> Phase 2/3: Full training starting at $(date)"
bash /root/autodl-tmp/vla/scripts/11_libero_4suites_train.sh
echo ">>> Training finished at $(date)"

# === Phase 3: Evaluation (~6-7h, 4 suites) ===
echo ""
echo ">>> Phase 3/3: Evaluation starting at $(date)"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/hf_cache

CKPT=/root/autodl-tmp/vla/outputs/libero_4suites_m5b/checkpoints/last/pretrained_model

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

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  EVAL_JSON=/root/autodl-tmp/vla/eval_results/m5b_$SUITE/eval_info.json
  if [ -f "$EVAL_JSON" ]; then
    RATE=$(python -c "import json; data=json.load(open('$EVAL_JSON')); print(f\"{data['overall']['pc_success']:.1f}%\")" 2>/dev/null || echo "N/A")
    echo ">>> $SUITE: $RATE"
  else
    echo ">>> $SUITE: MISSING"
  fi
done

# === Auto-shutdown ===
echo ""
echo ">>> Auto-shutdown in 5 minutes at $(date)"
sleep 300
/usr/bin/shutdown -h now
