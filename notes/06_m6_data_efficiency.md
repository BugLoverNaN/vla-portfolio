# M6 — Data Efficiency on LIBERO-Spatial

## Goal
How many demonstrations does SmolVLA need to reach stable success on LIBERO-Spatial? Map the data-efficiency curve.

## Setup
- Backbone: `smolvla_base`, fine-tuned per variant
- Suite: LIBERO-Spatial (10 tasks)
- Episode sampling: sequential from index 1261 (10/30/50/100/200); the 432-ep endpoint reuses the M5a full-Spatial run
- Adaptive steps: 10→5k, 30→8k, 50→10k, 100→15k, 200→20k (avoid overfitting small sets)
- Config: batch=16, num_workers=8 (validated in M5b)
- Eval: n=500 (50 episodes x 10 tasks) per variant
- Fully automated train -> eval -> auto-shutdown pipeline (~22h unattended)

## Results
| Episodes | Steps | Success (n=500) |
|---|---|---|
| 10  | 5k  | 13.8% |
| 30  | 8k  | 50.6% |
| 50  | 10k | 54.6% |
| 100 | 15k | 64.0% |
| 200 | 20k | 64.2% |
| 432 (M5a) | 20k | 60.0% |

## Findings
1. Sharp threshold between 10 and 30 demos (+36.8 pts): the policy collapses at ~10 demos and clears 50% by 30.
2. Fast saturation: 50 demos reach ~85% of peak; performance plateaus at ~64% from 100 demos onward (100->200 = +0.2).
3. Strong data efficiency from the pretrained VLA backbone: ~100 in-domain demonstrations suffice for near-peak LIBERO-Spatial performance.

## Caveats
- Sequential (not stratified) sampling: small subsets may under-cover the 10 Spatial tasks, so the curve partly conflates demo quantity with task coverage. A stratified resample (k demos/task) would decouple them.
- The 100/200-ep points (64.0/64.2%) sit slightly above the 432-ep baseline (60.0%); this is within eval noise (~±3-4% at n=500) and across separate runs, NOT evidence that more data hurts.

## Pitfalls
- `--dataset.episodes` needs a JSON list, not a Python `range(...)` expression (draccus does not eval Python) -> generated via `json.dumps(list(range(...)))`. (Same root cause as pitfall #13.)
- LeRobot auto-scales the LR scheduler (warmup/decay shrink with the step budget); other frameworks would need manual linear scaling.
