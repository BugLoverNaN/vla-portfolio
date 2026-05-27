# M5: LIBERO-Spatial Fine-tuning Results

## Training Setup

- **Base model**: SmolVLA-450M (lerobot/smolvla_base)
- **Dataset**: HuggingFaceVLA/libero — LIBERO-Spatial suite (432 episodes, 52,970 frames)
- **Robot**: Franka Panda 7-DoF (EEF delta action)
- **Hardware**: NVIDIA RTX 4090 D (24GB)
- **Training steps**: 20,000
- **Batch size**: 8

## Training Results

- **Initial loss**: ~1.83
- **Final loss**: ~0.42
- **Loss reduction**: ~77%
- **Wall-clock time**: ~2h 51min

### Comparison with M3 (SO-101)

| Metric | M3 (SO-101) | M5 (LIBERO-Spatial) |
|--------|-------------|---------------------|
| Robot | 6-DoF joint position | 7-DoF EEF delta |
| Dataset frames | 11,939 | 52,970 |
| Initial loss | 0.55 | 1.83 |
| Final loss | 0.086 | 0.42 |
| Train time | 1h 11min | 2h 51min |

**Observation**: M5 initial loss is 3.3× higher than M3, quantifying the pretraining transfer gap for cross-embodiment fine-tuning. SmolVLA-base was pretrained heavily on SO-100 family (6-DoF joint pos), so transferring to Franka 7-DoF EEF delta requires substantial action-head re-learning.

## Evaluation Results (n=500, full eval)

**Overall success rate: 60.0% (300/500)**

### Per-task breakdown

| Task ID | Successes | Rate |
|---------|-----------|------|
| Task 0 | 34/50 | 68% |
| Task 1 | 35/50 | 70% |
| Task 2 | 28/50 | 56% |
| Task 3 | 39/50 | 78% |
| Task 4 | 21/50 | 42% |
| Task 5 | 15/50 | 30% |
| Task 6 | 35/50 | 70% |
| Task 7 | 32/50 | 64% |
| Task 8 | 34/50 | 68% |
| Task 9 | 27/50 | 54% |

### Task difficulty distribution

- Strong (>=68%): Tasks 0, 1, 3, 6, 8 (5 tasks)
- Medium (54-67%): Tasks 2, 7, 9 (3 tasks)
- Weak (30-42%): Tasks 4, 5 (2 tasks)

All 10 tasks have non-zero success rate, indicating the model learned partial capability for every task. No catastrophic failures (the 0% rate observed in smoke eval n=30 was statistical artifact).

## Reference Comparisons

| System | LIBERO-Spatial | Notes |
|--------|----------------|-------|
| **This work (20K steps)** | **60.0%** | Our reproduction |
| Octo baseline | ~60% | Comparable |
| Diffusion Policy | ~78% | Paper |
| OpenVLA-OFT | ~88% | 7B params |
| SmolVLA paper | 95.4% | 60K-100K steps |

The gap to paper number reflects: (a) 20K vs 60K+ training steps, (b) batch 8 vs 32-64, (c) 3 epoch vs 10+ epoch coverage, (d) default vs tuned inference params. LeRobot issues #2354/#3264 document that even the official SmolVLA checkpoint cannot reproduce paper numbers in community evaluations.

## Smoke vs Full Eval Comparison

The two evaluation rounds illustrate sampling variance:

| Eval | n | Overall | Range across tasks |
|------|---|---------|---------------------|
| Smoke | 30 | 50.0% | 0% to 100% |
| Full | 500 | 60.0% | 30% to 78% |

Smoke eval had 3 tasks at 0% success, all of which turned out to be 30-54% in full eval. This is a classic example of why n>=50 per task is the LIBERO benchmark standard.

## Engineering Notes

Successfully resolved schema mismatch between SmolVLA-base (camera1/2/3 names, [6]-dim state, [6]-dim action) and LIBERO data (image/image2, [8]-dim state, [7]-dim action). SmolVLA max_dim=32 padding mechanism handles dimension differences transparently; --rename_map handles camera renaming.

## Reproducing

    # Smoke test (5 min validation)
    bash scripts/07b_libero_spatial_smoke_500.sh
    
    # Full training (3 hours, 20K steps)
    bash scripts/07_libero_spatial_train.sh
    
    # Full evaluation (100 min, 500 episodes)
    bash scripts/08_libero_spatial_full_eval.sh
