"""
04_test_finetuned_inference.py
Load the M3-finetuned SmolVLA checkpoint and verify inference.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

CHECKPOINT = "/root/autodl-tmp/vla/outputs/smolvla_so101_m3/checkpoints/last/pretrained_model"
device = "cuda"

print(f"Loading M3 finetuned model from:\n  {CHECKPOINT}\n")
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT).to(device).eval()
print(f"Model loaded ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params)")

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    CHECKPOINT,
    preprocessor_overrides={"device_processor": {"device": device}},
)

batch = {}
for key, feat in policy.config.input_features.items():
    if "image" in key:
        batch[key] = torch.rand(1, *feat.shape)
    elif "state" in key:
        batch[key] = torch.randn(1, *feat.shape)
batch["task"] = ["pick up the red lego brick"]

print("\nRunning inference 3 times (flow matching introduces noise):")
for i in range(3):
    processed = preprocess(batch)
    with torch.no_grad():
        action = policy.select_action(processed)
    action_final = postprocess(action)
    print(f"  Run {i+1}: {action_final.flatten().cpu().numpy().round(3)}")

print("\nM3 finetuned SmolVLA inference SUCCESS")
