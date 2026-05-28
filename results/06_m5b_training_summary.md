# M5b: LIBERO 4-Suite Mixed Training Results

## Training Setup

- **Base model**: SmolVLA-450M (lerobot/smolvla_base)
- **Dataset**: HuggingFaceVLA/libero — ALL 4 suites mixed (1693 episodes, 273K frames)
- **Robot**: Franka Panda 7-DoF (EEF delta action)
- **Hardware**: NVIDIA RTX 4090 D (24GB)
- **Training steps**: 40,000
- **Batch size**: 16
- **num_workers**: 8 (A/B/C tested for throughput)

## Training Results

- **Final loss**: 0.367
- **Wall-clock time**: 5h 41min
- **Throughput**: ~1.95 step/s effective (batch=16)

### Throughput optimization (A/B/C test)

| num_workers | data_s | step/s | GPU util |
|-------------|--------|--------|----------|
| 4 (default) | 0.74s | 1.0 | 24% |
| 6 | 0.44s | 1.48 | 34% |
| 8 (chosen) | 0.36s | 1.67 | 39% |

Increasing num_workers from LeRobot default 4 to 8 improved training throughput by 67%. Did not push to 16 because that caused dataset cache lock-file deadlock (documented in pitfall #16).

## Evaluation Results (n=500 per suite, 2000 total trials)

| Suite | Success Rate | Paper (450M) |
|-------|--------------|--------------|
| LIBERO-Spatial | 53.8% | 95.4% |
| LIBERO-Object | 50.6% | 96.6% |
| LIBERO-Goal | 66.0% | 93.4% |
| LIBERO-Long | 33.0% | 80.0% |
| **Average** | **50.9%** | 91.4% |

### Per-suite best/worst tasks

- Goal best: Task 7 at 92%
- Long worst: Task 4 at 4% (long-horizon manipulation)

## Key Finding: Multi-Task Interference

Comparing Strategy A (single-suite) vs Strategy B (4-suite mix) on LIBERO-Spatial:

| Strategy | Spatial Success | Training |
|----------|-----------------|----------|
| A (single-suite, 20K steps) | 60.0% | Spatial only |
| B (4-suite mix, 40K steps) | 53.8% | All 4 suites |

Despite 2x more training steps, the 4-suite mixed model scored 6.2pp LOWER on Spatial. This demonstrates multi-task interference: sharing model capacity across 4 task distributions dilutes single-suite specialization. This is a well-known trade-off between generalist (multi-task) and specialist (single-task) policies.

## Reference Context

The gap to SmolVLA paper numbers (avg 91.4% vs our 50.9%) reflects training budget (40K vs the paper's much larger budget) and inference parameter tuning. LeRobot issues #2354/#3264/#3287 document that even official checkpoints fall short of paper numbers in community reproduction, with effective batch size being a suspected key factor.

## Reproducing

    # A/B/C smoke test (validate config)
    bash scripts/11b_smoke_workers6.sh
    bash scripts/11c_smoke_workers8.sh
    
    # Full pipeline: train 40K + eval 4 suites + auto-shutdown
    bash scripts/12_m5b_pipeline_final.sh
