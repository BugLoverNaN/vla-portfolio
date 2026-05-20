# M3 Results — SmolVLA Fine-tuning on SO-101 PickPlace

**Date**: 2026-05-19
**Hardware**: NVIDIA RTX 4090 D (24GB)
**Total training time**: 1h 11m
**Status**: ✅ COMPLETED (20,000 / 20,000 steps)

## Configuration

| Parameter | Value |
|---|---|
| Base model | `lerobot/smolvla_base` (450M params, 100M trainable) |
| Dataset | `lerobot/svla_so101_pickplace` (50 ep, 11,939 frames) |
| Steps | 20,000 |
| Batch size | 8 |
| Optimizer | AdamW (peak lr=1e-4, cosine decay) |
| Video backend | pyav |
| Camera mapping | up→camera1, side→camera2, camera3=empty |

## Loss Curve

![Loss Curve](03_loss_curve.png)

Raw data: [03_loss_data.csv](03_loss_data.csv)
Condensed log: [03_m3_training_log.txt](03_m3_training_log.txt)

## Training Metrics

| Stage | Loss | Grad Norm | LR |
|---|---|---|---|
| Step 100 (start) | 0.674 | 8.414 | 7.7e-06 |
| Step 20,000 (final) | **0.086** | 1.391 | 2.5e-06 |
| Smoke test (500 steps, M2.5) | 0.196 | 2.74 | - |

## Observations

1. **Strong convergence**: Loss decreased from 0.674 → 0.086, a 87% reduction over 20K steps.

2. **Pretraining transfer dominant**: The model started at loss ~0.5 (not ~2-3 as random-init would), confirming that SmolVLA-base's community-data pretraining already covers SO-100/SO-101 embodiment semantics. Fine-tuning is task specialization on top of strong priors.

3. **Stable gradients**: Grad norm trended from initial spikes (~8) to stable ~1.4 by end of training. No exploding/vanishing gradient issues; no manual intervention needed.

4. **Marginal returns after step ~5,000**: Loss curve flattens significantly after the first quarter of training. For production workloads on similar data, 5-10K steps may suffice and save ~50% compute.

5. **Throughput**: ~4-5 steps/sec on RTX 4090 D. data_s ≪ updt_s confirms pyav decoder is not the bottleneck.

## Checkpoint Validation

Inference pipeline tested on M3 checkpoint:
- ✅ `SmolVLAPolicy.from_pretrained()` loads successfully (906MB model.safetensors)
- ✅ Pre/post-processors load from checkpoint dir
- ✅ End-to-end forward pass returns 6-DoF action tensor

See [04_finetuned_inference_output.txt](04_finetuned_inference_output.txt).

## Next Steps (M4)

Set up LIBERO simulation environment and run quantitative evaluation
(task success rate across LIBERO-Spatial / Object / Goal / Long suites).
