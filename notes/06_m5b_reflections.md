# M5b: LIBERO 4-Suite Mixed Training — Engineering Reflections

> Date: 2026-05-27 to 2026-05-28
> Training: 40,000 steps, 5h 41min on RTX 4090 D
> Evaluation: 4 suites x 500 episodes = 2000 trials
> 4-suite average: 50.9%

This document captures the engineering and research insights from M5b, the 4-suite mixed training experiment, building on M5a (single-suite Spatial).

## Key Research Finding: Multi-Task Interference

The headline result is a comparison between two training strategies:

| Strategy | LIBERO-Spatial | Steps | Data |
|----------|----------------|-------|------|
| A (specialist) | 60.0% | 20K | Spatial only (432 ep) |
| B (generalist) | 53.8% | 40K | All 4 suites (1693 ep) |

Strategy B used 2x more steps and 4x more data, yet scored 6.2pp LOWER on Spatial. This is multi-task interference: when one model learns 4 task distributions simultaneously, capacity is shared and single-suite specialization degrades. This mirrors findings in the multi-task RL literature (negative transfer between dissimilar tasks).

Implication for VLA deployment: if the target application is a narrow task family, a specialist model may outperform a generalist trained on broader data, at equal model size.

## Throughput Optimization (A/B/C Test)

Investigated GPU underutilization (initially ~50% with batch=8). Root cause via log analysis: data_s (dataloader wait) >> updt_s (GPU compute), meaning GPU was starved waiting for CPU data prep.

A/B/C results at batch_size=16:

| num_workers | data_s | step/s | GPU util |
|-------------|--------|--------|----------|
| 4 | 0.74s | 1.0 | 24% |
| 6 | 0.44s | 1.48 | 34% |
| 8 | 0.36s | 1.67 | 39% |

Chose num_workers=8: 67% throughput gain over default. Stopped there because 16 workers triggered dataset cache lock-file deadlock + OOM-like behavior. Engineering principle: stability > marginal optimization.

## Train vs Eval Optimization Philosophy

A deliberate decision: optimized training throughput (num_workers) but kept evaluation parameters at defaults (batch_size=1, n_action_steps=50, num_steps=10). Rationale:

- Training optimization target: throughput (faster = cheaper, doesn't change result)
- Eval optimization target: reproducibility (must keep Strategy A and B numbers comparable)

Changing eval batch_size could introduce numerical drift via batched inference, making A vs B comparison invalid. Kept eval identical to Strategy A for scientific validity.

## Suite Difficulty Analysis

| Suite | Success | Difficulty signal |
|-------|---------|-------------------|
| Goal | 66.0% | Easiest — goal-conditioned, clear targets |
| Spatial | 53.8% | Medium — spatial reasoning |
| Object | 50.6% | Medium — diverse objects |
| Long | 33.0% | Hardest — long-horizon multi-step |

Long-horizon (LIBERO-Long) is dramatically harder. Task 4 scored only 4% (2/50). Long tasks compound errors across many sub-steps; a single mid-trajectory failure dooms the episode. This matches the paper (Long is the lowest suite even for the official model at 80% vs 90%+ for others).

## Full Automated Pipeline

Built a single bash pipeline (12_m5b_pipeline_final.sh) that runs unattended:

1. Training (40K steps, ~5.7h)
2. Checkpoint verification (abort eval if missing)
3. Evaluation of 4 suites sequentially (each suite isolated with || true)
4. Results summary printing
5. Auto-shutdown after 5-min grace

set -e ensures a training failure aborts before wasting eval time. Per-suite || true ensures one suite failure doesn't kill the others. The pipeline completed fully autonomously overnight and triggered auto-shutdown, saving GPU cost.

## Open Questions

- Would per-suite fine-tuning from the mixed checkpoint recover the specialist performance?
- Does the multi-task interference shrink with a larger model (capacity bottleneck hypothesis)?
- Long-horizon: would action chunk size tuning or longer episode_length help?

## Reproducibility

Hardware: 24GB+ NVIDIA GPU + high-RAM host (the cache build benefits from large RAM). Pipeline:
1. A/B/C smoke (scripts 11b, 11c)
2. Full pipeline (script 12_m5b_pipeline_final.sh)

Result: 4-suite avg 50.9%. Strategy A single-suite Spatial: 60.0%. The specialist-generalist gap (60.0 vs 53.8 on Spatial) is the key takeaway.
