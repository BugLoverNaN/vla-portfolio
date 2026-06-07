#!/bin/bash
# M6 Data Efficiency Pipeline
# - Train 5 SmolVLA models on different episode counts (10/30/50/100/200)
# - Evaluate each on LIBERO-Spatial (n=500)
# - Auto-shutdown after completion
# - 432 episodes uses existing M5a results
#
# Estimated total: ~21-22 hours (12h train + 8-9h eval)
# Config: batch=16, num_workers=8 (validated by M5b)

set -e

PIPELINE_LOG=/root/autodl-tmp/vla/outputs/m6_pipeline.log
exec > >(tee -a $PIPELINE_LOG) 2>&1

echo ""
echo "=================================================="
echo "M6 Data Efficiency Pipeline Started at $(date)"
echo "Episodes to test: 10 / 30 / 50 / 100 / 200"
echo "(432 already done in M5a, reusing 60.0%)"
echo "Config: batch=16 num_workers=8"
echo "Expected: ~12h train + ~8-9h eval = ~21-22h total"
echo "=================================================="

# Environment
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=true

# === Define experiments ===
# Format: "n_episodes:steps"
declare -a EXPERIMENTS=(
    "10:5000"
    "30:8000"
    "50:10000"
    "100:15000"
    "200:20000"
)

# === Function: train one variant ===
train_variant() {
    local N_EP=$1
    local STEPS=$2
    
    echo ""
    echo "------------------------------------------"
    echo ">>> Training data_eff_${N_EP}ep (steps=${STEPS}) at $(date)"
    echo "------------------------------------------"
    
    # Generate episode range: starts at 1261 (LIBERO-Spatial start)
    local START=1261
    local END=$((START + N_EP))
    local EPISODES=$(python -c "import json; print(json.dumps(list(range(${START}, ${END}))))")
    
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
        --steps=$STEPS \
        --save_freq=$STEPS \
        --log_freq=100 \
        --output_dir=/root/autodl-tmp/vla/outputs/m6_data_eff_${N_EP}ep \
        --job_name=m6_data_eff_${N_EP}ep \
        --wandb.enable=false
    
    echo ">>> data_eff_${N_EP}ep training done at $(date)"
}

# === Function: evaluate one variant ===
eval_variant() {
    local N_EP=$1
    local CKPT=/root/autodl-tmp/vla/outputs/m6_data_eff_${N_EP}ep/checkpoints/last/pretrained_model
    
    echo ""
    echo "------------------------------------------"
    echo ">>> Evaluating data_eff_${N_EP}ep at $(date)"
    echo "------------------------------------------"
    
    if [ ! -f "$CKPT/model.safetensors" ]; then
        echo "WARNING: Checkpoint missing for ${N_EP}ep, skipping eval"
        return
    fi
    
    lerobot-eval \
        --policy.path=$CKPT \
        --policy.device=cuda \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.n_episodes=50 \
        --eval.batch_size=1 \
        --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
        --output_dir=/root/autodl-tmp/vla/eval_results/m6_data_eff/m6_${N_EP}ep \
        || echo "WARNING: ${N_EP}ep eval failed but continuing"
    
    echo ">>> data_eff_${N_EP}ep eval done at $(date)"
}

# === Phase 1: Train all 5 variants ===
echo ""
echo "========================================"
echo "PHASE 1/2: Training (5 variants)"
echo "========================================"

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r N_EP STEPS <<< "$EXP"
    train_variant $N_EP $STEPS
done

echo ""
echo ">>> All training done at $(date)"

# === Phase 2: Evaluate all 5 variants ===
echo ""
echo "========================================"
echo "PHASE 2/2: Evaluation (5 variants)"
echo "========================================"

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r N_EP STEPS <<< "$EXP"
    eval_variant $N_EP
done

# === Final summary ===
echo ""
echo "=================================================="
echo "M6 Pipeline Finished at $(date)"
echo "=================================================="
echo ""
echo "Results summary:"
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r N_EP STEPS <<< "$EXP"
    EVAL_JSON=/root/autodl-tmp/vla/eval_results/m6_data_eff/m6_${N_EP}ep/eval_info.json
    if [ -f "$EVAL_JSON" ]; then
        RATE=$(python -c "import json; data=json.load(open('$EVAL_JSON')); print(f\"{data['overall']['pc_success']:.1f}\")" 2>/dev/null || echo "N/A")
        echo ">>> ${N_EP} episodes: $RATE%"
    else
        echo ">>> ${N_EP} episodes: MISSING"
    fi
done
echo ">>> 432 episodes (M5a baseline): 60.0%"

# === Auto-shutdown ===
echo ""
echo ">>> Auto-shutdown in 5 minutes at $(date)"
echo ">>> If you want to cancel: ssh in and run 'sudo shutdown -c'"
sleep 300
/usr/bin/shutdown -h now
