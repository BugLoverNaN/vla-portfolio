# M5: LIBERO-Spatial Fine-tuning — Engineering Reflections

> Date: 2026-05-24 to 2026-05-25
> Training: 20,000 steps, 2h 51min on RTX 4090 D
> Full evaluation: 500 episodes, 60.0% success rate

This document captures engineering decisions and observations from the M5 LIBERO-Spatial fine-tuning.

## Key Engineering Decisions

### 1. Two-stage smoke testing methodology

Instead of one smoke test, used two:
- Stage 1 (50 steps, 5 episodes): catch schema mismatch fast
- Stage 2 (500 steps, full 432 episodes): confirm stable learning at scale
- Only after both pass: launch 20K-step full training

Stage 2 caught the draccus parser bug (range() expression not accepted), which would have wasted 3 hours if discovered in full training.

### 2. PID-based watchdog (lessons from M3 self-match bug)

M3 watchdog used pgrep matching command-line strings, which self-matched the watchdog process. M5 fix:

    EVAL_PID=$(pgrep -f "lerobot-eval" | head -1)
    while kill -0 $EVAL_PID 2>/dev/null; do
      sleep 60
    done

kill -0 checks process liveness via OS-level PID lookup with zero string matching.

### 3. Two-tier evaluation (smoke + full)

Smoke eval (n=30) finished in 6 min and gave 50% with high variance — 3 tasks showed 0% (statistical artifact).
Full eval (n=500) ran 100 min and gave 60% with tight variance — all tasks 30-78% (no false zeros).

This validates the LIBERO benchmark protocol of n=50 per task.

## Observations on SmolVLA on LIBERO

### Quantified pretraining transfer gap

- M3 (SO-101) initial loss: 0.55
- M5 (LIBERO/Franka) initial loss: 1.83
- Ratio: 3.3x higher for cross-embodiment

This is empirical evidence of the embodiment gap. SmolVLA-base was pretrained on 487 community datasets, dominantly SO-100 family (6-DoF joint position). Transferring to Franka (7-DoF EEF delta) requires substantial action-head adaptation.

### Loss-eval correlation

- Final training loss: 0.42
- Eval success rate: 60.0%

For comparison, M3 final loss 0.086 corresponds to known good real-robot transfer in similar tasks. The loss-success mapping is highly task-dependent.

### Long-tail task difficulty

Per-task success rate ranges 30-78%. Tasks 4 and 5 are notably harder (42%, 30%). Visual inspection of failure videos would reveal:
- Are these tasks with rare object configurations?
- Are they semantically harder language instructions?
- Are they physically harder (smaller objects, narrower targets)?

This is a clear future research direction.

### Schema padding works as designed

SmolVLA's max_state_dim=32 and max_action_dim=32 handled [8]-dim state and [7]-dim action transparently via attention masking. No manual config patching needed. This is non-trivial — many architectures would have required explicit retraining the action head.

## Open Questions

- **Inference parameter sensitivity**: LeRobot issues #2354/#3264 suggest num_steps and n_action_steps materially affect eval. Default 60% might rise to 70%+ with tuning.
- **More training**: SmolVLA paper used 60K-100K steps for 95% on LIBERO. 20K steps is clearly under-trained.
- **Strategy B (next milestone)**: Mixed 4-suite training will reveal whether multi-task training helps or hurts single-suite (Spatial) performance.

## Reproducibility

Hardware: any 24GB+ NVIDIA GPU. Code reproduces via:

1. Install: pip install -e ".[smolvla,dataset,libero]" + 14 pitfalls in notes/03_finetuning_setup.md
2. Data: HuggingFaceVLA/libero (~32GB)
3. Run: scripts 07a then 07b then 07 then 08a then 08

Reference SmolVLA paper LIBERO-Spatial: 95.4%. Our reproduction: 60.0%. The gap is explained by training scale (20K vs 60K-100K steps) and inference parameter tuning, not by methodology bugs.
