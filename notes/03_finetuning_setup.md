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

---

## M4 LIBERO Integration Pitfalls (2026-05-22)

The following issues were encountered while preparing the LIBERO simulation environment and dataset. All resolved.

### #8 — egl_probe build fails on CMake 4.x

**Context**: Running `pip install -e ".[libero]"` to install LIBERO extras.

**Symptom**:

    CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
      Compatibility with CMake < 3.5 has been removed from CMake.
      Update the VERSION argument <min> value.
      Or, add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.

**Root cause**: `egl_probe` (a transitive dependency of LIBERO for EGL device probing) ships a `CMakeLists.txt` using pre-3.5 syntax. AutoDL images ship CMake 4.x, which dropped compatibility with that range.

**Fix**:

    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    pip install -e ".[libero]"

The env var tells CMake 4.x to apply pre-3.5 policy defaults instead of erroring. This is the official escape hatch documented in CMake's release notes.

**Engineering insight**: C++ extensions in ML packages often lag behind toolchain releases by years. When facing native build failures, the first diagnostic question is "is this a version-compatibility issue between the native toolchain (CMake / gcc / setuptools) and the Python wrapper?"

---

### #9 — LIBERO assets distribution failures (HF mirror 429 / rate-limiting)

**Context**: First `python scripts/06_test_libero_env.py` run triggers LIBERO to download ~70MB of scene assets (XML + meshes) from `lerobot/libero-assets` on HuggingFace.

**Symptom**: Repeated `HTTP 429 Too Many Requests` from hf-mirror, then SSL handshake timeout. Local fallback also fails (assets missing in package). Final error:

    FileNotFoundError: libero/libero/assets/scenes/libero_tabletop_base_style.xml

**Root cause**: LIBERO assets consist of hundreds of small files (per-scene XML/OBJ/STL/MSH). Concurrent fetches from hf-mirror trigger per-IP rate limiting. AutoDL's IP range hits the limit fast.

**Fix**: Bypass the mirror entirely by cloning the LIBERO source repo from GitHub via AutoDL's academic accelerator:

    source /etc/network_turbo
    cd /tmp && git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
    LIBERO_ASSETS_DIR=$(python -c "import libero, os; print(os.path.dirname(libero.__file__) + '/libero/assets')")
    rm -rf $LIBERO_ASSETS_DIR
    cp -r /tmp/LIBERO/libero/libero/assets $LIBERO_ASSETS_DIR

**Engineering insight**: When a mirror service rate-limits you, try the original source via accelerated egress instead of fighting the mirror. The original LIBERO repo on GitHub contains complete assets and is reachable via academic accelerator in <1 minute.

---

### #10 — HuggingFaceVLA/libero dataset: xethub S3 503 + download fragility

**Context**: Downloading the 32GB `HuggingFaceVLA/libero` dataset (377 parquet files).

**Symptom**: Persistent `503 Service Unavailable` errors from `cas-bridge.xethub.hf.co` (HuggingFace's xet protocol S3 backend). aria2 retries indefinitely; ~10 files fail to download even after multiple retry rounds.

**Root cause**: HuggingFace splits large dataset files via the xet protocol, which stores actual bytes on S3 (`cas-bridge.xethub.hf.co`). The S3 endpoint enforces aggressive rate limiting for clients without authentication tokens.

**Fix** (combined approach):
1. **Multi-threaded retry with hfd**: 8-way parallel via `~/hfd.sh <repo> --dataset --tool aria2c -x 8`
2. **Mirror + accelerator fallback**: Switch between `HF_ENDPOINT=https://hf-mirror.com` and `source /etc/network_turbo + unset HF_ENDPOINT`
3. **Resume support**: hfd uses aria2 session files, so re-running picks up where it left off

For the final 6 stubborn files, fell back to direct `wget --continue` via academic accelerator.

**Engineering insight**: HF Hub downloads are not a single API but a multi-stage redirect chain (HF API → xethub presigned URL → S3 object). Each stage has its own rate limits. When stuck, try cycling between endpoints and tools rather than hammering the same stuck request.

---

### #11 — Nested directory pollution from hfd resume

**Context**: After multiple hfd retries from different working directories, the dataset ended up in two nested locations.

**Symptom**:

    /root/autodl-tmp/vla/datasets/libero/data/        ← partially downloaded
    /root/autodl-tmp/vla/datasets/libero/libero/data/ ← also partially downloaded
    /root/autodl-tmp/vla/datasets/libero/libero/libero/data/ ← yet another partial!

Each level had a different subset of files. No single directory was complete.

**Root cause**: `hfd` infers the target subdirectory from `repo_id` when started inside a directory that already matches the pattern. Re-running it from different `cwd`s created nested duplicates.

**Fix**: Merge with `mv -n` (no-clobber, keeps existing files):

    INNER=/root/autodl-tmp/vla/datasets/libero/libero/data/chunk-000
    OUTER=/root/autodl-tmp/vla/datasets/libero/data/chunk-000
    mv -n $INNER/*.parquet $OUTER/
    rm -rf /root/autodl-tmp/vla/datasets/libero/libero/

**Engineering insight**: For download/sync tools that resume by directory state, always run them from the same working directory. `mv -n` is the safe merge pattern: it preserves existing files and only moves what's missing.

---

### #12 — Disk full masquerading as 503 / SSL timeout / "stuck process"

**Context**: Multiple seemingly-unrelated failures over several hours: aria2 503 errors, wget "stuck", Python `LeRobotDataset()` silently hanging with no output.

**Symptom diversity**:
- `aria2: errorCode=16 ... cause: No space left on device` (clear)
- `wget: Cannot write to 'file-340.parquet' (Success)` (misleading "Success")
- `Python: LeRobotDataset(...)` hangs indefinitely after printing "Loading..." (no error)
- `OSError: [Errno 28] No space left on device` deep in pyarrow / fsspec traceback (only when error surfaces)

**Root cause**: AutoDL default data disk is 50GB. The LIBERO dataset (32GB parquet) + HuggingFace `datasets` library Arrow cache (~100GB) + M3 checkpoints (13GB) far exceed it. Symptoms manifested differently depending on which layer hit the disk-full condition first.

**Fix** (two-part):
1. **Redirect HF cache to data disk** (otherwise it goes to `~/.cache/`):

       echo 'export HF_HOME=/root/autodl-tmp/hf_cache' >> ~/.bashrc
       echo 'export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets' >> ~/.bashrc
       echo 'export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub' >> ~/.bashrc
       source ~/.bashrc
       mkdir -p $HF_DATASETS_CACHE $HUGGINGFACE_HUB_CACHE

2. **Resize data disk** via AutoDL console: 50GB → 145GB (sufficient for ~100GB Arrow cache + room for training checkpoints)

**Engineering insight**: This is the single most important takeaway from M4. When facing inexplicable "network errors" or "stuck processes" in ML pipelines, **always check disk first** via `df -h`. ML data pipelines write large temporary files in non-obvious places (fsspec local cache, datasets Arrow cache, pyarrow scratch). A full disk surfaces as 5+ different error messages across the stack. PNG bytes embedded in parquet means the LeRobot dataset doesn't need a `videos/` directory — the `info.json` `video_path` template is just an unused schema field.


---

## M5 Training Pitfalls (2026-05-24)

### #13 — LeRobot draccus parser rejects Python expressions

**Context**: Trying to pass dataset.episodes='range(1261, 1693)' to lerobot-train.

**Symptom**:

    draccus.utils.DecodingError: dataset.episodes: Could not decode the value into any of the given types:
        list[int]: The given value='range(1261, 1693)' is not of a valid input for a list type

**Root cause**: LeRobot v0.5 uses draccus for CLI config parsing. It expects JSON literals, not Python expressions. range(...) is a Python builtin, not a JSON form.

**Fix**: Materialize the range to JSON list at shell level:

    EPISODES=$(python -c "import json; print(json.dumps(list(range(1261, 1693))))")
    lerobot-train --dataset.episodes="$EPISODES" ...

**Engineering insight**: When a CLI tool rejects "obviously valid" syntax, check what parser it uses. Different parsers (argparse, click, draccus, hydra, fire) have very different opinions on what's valid input.

---

### #14 — Eval requires same rename_map as training

**Context**: Running lerobot-eval after training, hit feature mismatch error.

**Symptom**:

    ValueError: Feature mismatch between dataset/environment and policy config.
    - Missing features: ['observation.images.camera1', ...]
    - Extra features: ['observation.images.image', 'observation.images.image2']

**Root cause**: LIBERO simulation outputs cameras with dataset-original names (image, image2), but the policy was trained with renamed features (camera1, camera2). The rename mapping must be applied at eval time too.

**Fix**: Pass identical --rename_map argument to lerobot-eval:

    lerobot-eval ... \
      --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

**Engineering insight**: Schema transformations during training need to be replayed at inference/eval time. The LeRobot error message is exemplary — it lists exact missing/extra features and shows the fix.

---

## M5 Training Pitfalls (2026-05-24)

### #13 — LeRobot draccus parser rejects Python expressions

**Context**: Passing dataset.episodes='range(1261, 1693)' to lerobot-train.

**Symptom**:

    draccus.utils.DecodingError: dataset.episodes: Could not decode the value into any of the given types:
        list[int]: The given value='range(1261, 1693)' is not of a valid input for a list type

**Root cause**: LeRobot v0.5 uses draccus for CLI parsing. It expects JSON literals, not Python expressions.

**Fix**: Materialize the range to JSON at shell level:

    EPISODES=$(python -c "import json; print(json.dumps(list(range(1261, 1693))))")
    lerobot-train --dataset.episodes="$EPISODES" ...

**Engineering insight**: When a CLI rejects "obviously valid" syntax, check which parser library it uses. Different parsers (argparse, click, draccus, hydra, fire) have very different opinions on valid input.

---

### #14 — Eval requires same rename_map as training

**Context**: Running lerobot-eval after training, hit feature mismatch error.

**Symptom**:

    ValueError: Feature mismatch between dataset/environment and policy config.
    - Missing features: ['observation.images.camera1', 'observation.images.camera2', ...]
    - Extra features: ['observation.images.image', 'observation.images.image2']

**Root cause**: LIBERO simulation outputs cameras with dataset-original names (image, image2), but the policy was trained with renamed features (camera1, camera2). The rename mapping must be applied at eval time too.

**Fix**: Pass identical --rename_map argument to lerobot-eval:

    lerobot-eval ... \
      --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

**Engineering insight**: Schema transformations during training must be replayed at inference/eval. The LeRobot error message is exemplary — lists exact missing/extra fields with fix command.
