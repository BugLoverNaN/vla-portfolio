# M5: LIBERO-Spatial Fine-tuning — Engineering Reflections

> Date: 2026-05-24
> Run: 20,000 steps, ~2h 51min on RTX 4090 D
> Smoke eval result: 50% success rate (n=30)
> Full eval (n=500): in progress

This document captures non-obvious engineering decisions and observations from the M5 fine-tuning on LIBERO-Spatial. The methodology mirrors M3 (smoke-test-first, atomic commits) with adaptations for the new embodiment.

## Key Engineering Decisions

### 1. Two-stage smoke testing

- Stage 1 (50 steps, 5 episodes): catch schema mismatch fast
- Stage 2 (500 steps, full 432 episodes): confirm stable learning at scale
- Only after both pass: launch 20K-step full training

This caught the dataset.episodes range() decoding bug (LeRobot's draccus parser doesn't accept Python expressions, only JSON literals) in stage 2, before wasting 3 hours.

### 2. PID-based watchdog (lessons from M3 self-match bug)

M3 used pgrep -f "lerobot-train" which self-matched the watchdog process. This time:

    EVAL_PID=$(pgrep -f "lerobot-eval" | head -1)
    while kill -0 $EVAL_PID 2>/dev/null; do
      sleep 60
    done

kill -0 checks process liveness without sending a signal — zero string matching, zero self-match risk.

### 3. Episode-index based dataset subset

LIBERO has 4 suites (Spatial/Object/Goal/Long) interleaved in episode indices, not partitioned by suite. Identified LIBERO-Spatial as continuous range [1261, 1693) via task description matching ("pick up the black bowl ... place it on the plate").

## Observations on SmolVLA Behavior on LIBERO

1. **Pretraining transfer is weaker than for SO-101**: Initial loss ~1.83 (vs 0.55 for M3). This is empirical evidence of the embodiment gap — SmolVLA-base was pretrained heavily on SO-100 family (6-DoF joint pos), but LIBERO uses Franka Panda (7-DoF EEF delta). The action space difference forces significant action-head re-learning.

2. **Convergence pattern similar to M3**: ~80% of loss reduction happens in first 5K steps. By step 10K, marginal returns sharply diminish (loss 0.5 to 0.42 over the last 10K steps).

3. **Long-tail task difficulty**: Smoke eval shows clear bimodal distribution — 4 tasks at 67-100% success, 3 tasks at 0%. This is typical of imitation learning at this scale: certain object configurations the model just hasn't learned to handle.

4. **Schema mapping just works**: Despite SmolVLA-base config declaring state.shape=[6] and the data having state.shape=[8], SmolVLA's max_state_dim=32 padding mechanism transparently handles the mismatch. No manual config patching needed.

## Open Questions Worth Investigating

- **Why tasks 4, 5, 9 fail completely**: Inspect the failed videos — is it object reachability, language grounding, or specific spatial configurations?
- **Validation loss curve**: Currently train-loss only. A val split would show whether the 0.42 plateau is convergence or overfitting.
- **Inference parameter sensitivity**: LeRobot issues #2354/#3264 suggest policy.num_steps and policy.n_action_steps materially affect eval results. Could push from 50% to 70%+ with tuning.
- **More training**: SmolVLA paper used 60K-100K steps. 20K likely under-trained.

## Reproducibility Notes

Hardware: any 24GB+ NVIDIA GPU (RTX 4090 / A6000 / H100). Code reproduces via:

1. pip install -e ".[smolvla,dataset,libero]" + 14 pitfalls from notes/03_finetuning_setup.md
2. Data: HuggingFaceVLA/libero (8-12GB, see #10 in pitfalls doc for download tips)
3. Run: scripts 07a then 07b then 07 then 08a then 08

Reference SmolVLA paper number for LIBERO-Spatial: 95.4%. Reproduction in this work: 50% (smoke, n=30). The gap reflects (a) 20K vs 60K+ training steps, (b) default vs tuned inference params, and (c) LeRobot issues #2354/#3264 where even official checkpoints can't reproduce paper numbers.
