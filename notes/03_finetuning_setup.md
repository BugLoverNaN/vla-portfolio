# M3 Fine-tuning Setup — Engineering Notes

> Date: 2026-05-18
> Outcome: Smoke test passed (500 steps, loss 0.55 → 0.20 on RTX 4090 D)
> Time spent debugging: ~2 hours (7 distinct issues resolved)

This document captures the **non-obvious engineering issues** encountered while setting up SmolVLA fine-tuning with LeRobot v0.5+. Every issue here was a real blocker resolved through reading source code, error tracebacks, and HF issue trackers.

---

## TL;DR — Working Configuration

    lerobot-train \
      --policy.path=<local_smolvla_base_path> \
      --policy.push_to_hub=false \
      --policy.device=cuda \
      --policy.empty_cameras=1 \
      --dataset.repo_id=lerobot/svla_so101_pickplace \
      --dataset.root=<local_dataset_path> \
      --dataset.video_backend=pyav \
      --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
      --batch_size=8 \
      --steps=20000 \
      --wandb.enable=false

---

## Issue Log (7 distinct blockers)

### #1 — LeRobot v0.5 renamed the training entry point

**Symptom**:

    ModuleNotFoundError: No module named 'lerobot.scripts.train'

**Root cause**: v0.5 moved from `python -m lerobot.scripts.train` to a console script `lerobot-train` registered via `pyproject.toml` entry points.

**Fix**: Use `lerobot-train` directly.

---

### #2 — `datasets` extras no longer pulled by default

**Symptom**:

    ImportError: 'datasets' is required but not installed.
    Install it with: pip install 'lerobot[dataset]'

**Root cause**: LeRobot split optional dependencies more granularly. The `smolvla` extras no longer transitively include `datasets`.

**Fix**:

    pip install -e ".[dataset,smolvla]"

---

### #3 — `--policy.dtype` removed from SmolVLAConfig

**Symptom**:

    error: unrecognized arguments: --policy.dtype=bfloat16

**Root cause**: SmolVLA's policy config class in v0.5 doesn't expose `dtype`. Mixed precision is now controlled via `--policy.use_amp` at the trainer level.

**Note**: Many older tutorials/docs still reference `--policy.dtype`. Trust the actual `lerobot-train --help` output.

**Fix**: Omit the parameter entirely; SmolVLA uses fp32 weights by default with AMP autocast available.

---

### #4 — `policy.repo_id` required by default

**Symptom**:

    ValueError: 'repo_id' argument missing.
    Please specify it to push the model to the hub.

**Root cause**: v0.5 changed the default behavior: training now assumes you'll upload to HF Hub at the end. Without an explicit `repo_id`, validation fails.

**Fix**: For local-only training, explicitly disable:

    --policy.push_to_hub=false

---

### #5 — Dataset / model camera schema mismatch

**Symptom**:

    ValueError: Feature mismatch between dataset/environment and policy config.
    - Missing features: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
    - Extra features: ['observation.images.side', 'observation.images.up']

**Root cause**: SmolVLA-base was pretrained on a standardized 3-camera schema (camera1=top, camera2=wrist, camera3=side). But `svla_so101_pickplace` only recorded 2 cameras named `up` and `side`.

**Fix**:

    --rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'
    --policy.empty_cameras=1

`rename_map` translates dataset keys → policy keys. `empty_cameras=1` tells the policy "one expected camera is missing; pad with zeros."

**Engineering insight**: This is the #1 schema mismatch pattern when fine-tuning VLA models across community datasets. The "standardized 3-camera schema" is convention from the SmolVLA paper (OBS_IMAGE_1/2/3).

---

### #6 — torchcodec ABI break with PyTorch 2.8

**Symptom**:

    OSError: /root/miniconda3/lib/python3.12/site-packages/torchcodec/libtorchcodec_core7.so:
    undefined symbol: torch_dtype_float4_e2m1fn_x2

**Root cause**: torchcodec (LeRobot's default video decoder) ships compiled C++ extensions linked against specific PyTorch C++ ABI. PyTorch 2.8 introduced `float4_e2m1fn_x2` dtype symbols; pre-2.8 torchcodec doesn't have these.

**Fix**: Switch to the pure-Python pyav backend, which has no native binding to PyTorch internals:

    pip install av
    # Then in training command:
    --dataset.video_backend=pyav

**Performance impact**: Negligible. pyav decodes at ~0.08s/batch which is well below the ~0.15s GPU compute time per step.

---

### #7 — Driver / PyTorch CUDA compatibility (resolved earlier)

**Symptom**: PyTorch 2.8 cu130 binaries fail on instances with NVIDIA driver < 560.

**Fix**: Rent AutoDL instance with CUDA 12.8 image + Driver ≥ 580, then install:

    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128

---

## Validated Configuration (snapshot)

| Component | Version |
|---|---|
| NVIDIA Driver | 580.76.05 |
| CUDA (driver-reported) | 12.8 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| LeRobot | main branch (≥ 0.5.2) |
| av (pyav) | latest from PyPI |
| GPU | RTX 4090 D (24GB) |

## Future-proofing

To avoid these issues on a fresh setup:

1. Always read `lerobot-train --help` first to confirm current argument names.
2. Pin `torch`, `torchvision`, `torchaudio` in `constraints.txt` to prevent silent upgrades.
3. Prefer pyav over torchcodec for video decoding unless explicit performance benchmarks demand otherwise.
4. Test with a 500-step smoke run before committing to a full multi-hour training job.
