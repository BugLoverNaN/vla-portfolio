# M3: SmolVLA Fine-tuning — Engineering Reflections

> Date: 2026-05-19
> Run: 20,000 steps, 1h 11min on RTX 4090 D
> Result: loss 0.5xx → 0.086 (see results/03_m3_training_summary.md)

This document captures non-obvious engineering decisions and observations from the full M3 fine-tuning run. The smoke test (M2.5) cleared all setup issues; this note focuses on what happened during the full run and what I'd do differently.

## Key Engineering Decisions

### 1. Smoke-test-first workflow
Spent 2 minutes validating with 500 steps before committing to 71 minutes of full training. The smoke test caught all 7 LeRobot v0.5 compatibility issues (see `03_finetuning_setup.md`) before they became expensive failures. This is standard discipline borrowed from ML production practice.

### 2. Auto-shutdown via background watchdog
Set up `nohup bash -c 'while pgrep -f "lerobot-train"... shutdown'` to free the GPU automatically when training completed, preventing ~6 hours of idle GPU billing while I slept.

**Caveat discovered**: The watchdog used `pgrep -f "lerobot-train"` which matched its own command line containing that string (self-match bug). The shutdown never triggered. Next time I'll use `kill -0 $PID` against a captured PID — PID-based liveness checks don't have this string-matching pitfall.

### 3. log-scale loss visualization
Plotted loss on log y-axis. Linear-scale plots hide late-stage convergence (loss <0.2 looks flat) while log-scale reveals that loss continued improving from 0.2 to 0.08 over the second half of training.

## Observations on SmolVLA Behavior

1. **Pretraining transfer is strong**: Loss started at ~0.5, not the ~2+ I'd expect for random initialization. SmolVLA-base's pretraining on 487 community datasets (heavily SO-100 family) already encodes most of what's needed for SO-101 PickPlace. Fine-tuning here is closer to "task specialization" than "embodiment learning."

2. **Most learning happens early**: ~80% of total loss reduction occurs in the first 5,000 steps. The remaining 15,000 steps reduce loss only marginally (0.15 → 0.08). For applied workloads, 5-10K steps would deliver most of the benefit at half the compute cost.

3. **No need for fancy tricks**: Default LeRobot hyperparameters (lr=1e-4, AdamW, cosine decay, batch=8) just worked. No grad clipping interventions, no NaN events, no learning rate retuning. SmolVLA is operationally well-behaved.

4. **Memory efficient**: 4090 D's 24GB stayed under 50% utilization even with batch=8. There's headroom to push batch=16 or 32 for higher throughput in future runs.

## Open Questions Worth Investigating

- **Overfitting check**: Does loss on a held-out validation split keep dropping past step 5000, or does train/val gap widen? (Current run uses train-only loss.)
- **LoRA vs full fine-tune**: How much loss/quality is sacrificed by training only LoRA adapters (~5M params) instead of all 100M trainable params? Useful for cheap multi-task fine-tuning.
- **Batch scaling**: Does batch=32 with linear LR scaling reach same loss in fewer wall-clock minutes? 4090 has the memory.

These are the experiments I'd run next if this were a production project rather than a portfolio demo.

## Reproducibility Notes

To reproduce this exact run:
- Hardware: any 16GB+ NVIDIA GPU (RTX 3090/4090 class)
- Run script: `scripts/02_train_smolvla_so101.sh`
- Dataset: `lerobot/svla_so101_pickplace` (auto-downloaded from HF Hub via hf-mirror)
- Base model: `lerobot/smolvla_base`
- Expected wall time: 70-90 min depending on GPU

Configuration is fully captured in `outputs/smolvla_so101_m3/checkpoints/last/pretrained_model/train_config.json`, which LeRobot also writes automatically for `--resume=true` support.
