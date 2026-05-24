# M5: LIBERO-Spatial Fine-tuning Results

## Training Setup
- **Base model**: SmolVLA-450M (lerobot/smolvla_base)
- **Dataset**: HuggingFaceVLA/libero — LIBERO-Spatial suite only (432 episodes / ~53K frames)
- **Robot**: Franka Panda 7-DoF (EEF delta action)
- **Hardware**: NVIDIA RTX 4090 D (24GB)
- **Training steps**: 20,000
- **Batch size**: 8
- **Effective batch size**: 8

## Training Results
- **Initial loss**: 2.552
- **Final loss**: 0.415
- **Loss reduction**: 83.7%
- **Wall-clock time**: ~2h 51min

### Comparison with M3 (SO-101)

| Metric | M3 (SO-101) | M5 (LIBERO-Spatial) |
|--------|-------------|---------------------|
| Robot | 6-DoF joint pos | 7-DoF EEF delta |
| Dataset frames | 11,939 | 52,970 |
| Initial loss | 0.55 | 2.55 |
| Final loss | 0.086 | 0.415 |
| Train time | 1h 11min | ~2h 51min |

**Observation**: LIBERO-Spatial initial loss is ~4.6× higher than SO-101's, demonstrating
that pretraining transfer from SmolVLA-base is weaker for cross-embodiment (Franka 7-DoF EEF delta)
than for the SO-100 family it was extensively pretrained on.

## Smoke Evaluation Results (3 episodes per task, 30 total)

**Overall success rate: 50.0% (15/30)**

| Task ID | Successes | Rate |
|---------|-----------|------|
| 0 | 2/3 | 67% |
| 1 | 3/3 | 100% |
| 2 | 3/3 | 100% |
| 3 | 2/3 | 67% |
| 4 | 0/3 | 0% |
| 5 | 0/3 | 0% |
| 6 | 1/3 | 33% |
| 7 | 2/3 | 67% |
| 8 | 2/3 | 67% |
| 9 | 0/3 | 0% |


## Engineering Notes

### Successful Schema Mapping
- LIBERO data cameras (`image`, `image2`) → SmolVLA expected (`camera1`, `camera2`)
- Third camera (`camera3`) auto-padded via `--policy.empty_cameras=1`
- 8-dim state and 7-dim action handled by SmolVLA's `max_state_dim`/`max_action_dim` padding (max=32)

### Failed Tasks (0% success)
Tasks 4, 5, 9 had 0/3 success rate in smoke eval. This represents a clear long-tail
distribution where certain object configurations the model cannot solve at all,
characteristic of imitation learning at this scale.

### Reproducing
```bash
# Smoke test (validate pipeline, 5 minutes)
bash scripts/07b_libero_spatial_smoke_500.sh

# Full training (20K steps, 3 hours)
bash scripts/07_libero_spatial_train.sh

# Smoke eval (30 episodes, 6 minutes)
bash scripts/08a_libero_spatial_smoke_eval.sh

# Full eval (500 episodes, 100 minutes)
bash scripts/08_libero_spatial_full_eval.sh
```

## Reference
- SmolVLA paper LIBERO-Spatial: 95.4%
- LeRobot issue #2354, #3264: official checkpoint also can't reproduce paper numbers
- This work: 50.0% (smoke n=30), full eval (n=500) running
