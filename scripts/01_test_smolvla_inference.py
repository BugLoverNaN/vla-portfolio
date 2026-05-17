"""
SmolVLA Hello World - 推理验证(官方 API 版)
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

print("Loading model...")
model_path = "/root/autodl-tmp/vla/models/smolvla_base"
device = "cuda"

# === 1. 加载 policy ===
policy = SmolVLAPolicy.from_pretrained(model_path)
policy = policy.to(device)
policy.eval()
print(f"✅ Model on {next(policy.parameters()).device}")
print(f"   Params: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

# === 2. 创建预处理器(关键!) ===
# preprocess 会自动:
#   - tokenize 语言指令 -> observation.language.tokens
#   - normalize 状态和动作
#   - 把张量移到指定 device
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_path,  # 也可以是 HF repo id "lerobot/smolvla_base"
    preprocessor_overrides={"device_processor": {"device": device}},
)
print("✅ Pre/post processors built")

# === 3. 看模型需要什么输入 ===
print("\n=== Input features ===")
for k, v in policy.config.input_features.items():
    print(f"  {k}: shape={v.shape}")
print("\n=== Output features ===")
for k, v in policy.config.output_features.items():
    print(f"  {k}: shape={v.shape}")

# === 4. 构造原始输入(任意 device,preprocess 会搬到 cuda) ===
raw_batch = {}
for key, feat in policy.config.input_features.items():
    if "image" in key:
        # 图像通常需要 [0,1] 范围,用 rand 而非 randn
        raw_batch[key] = torch.rand(1, *feat.shape)
    elif "state" in key:
        raw_batch[key] = torch.randn(1, *feat.shape)

# 语言指令(原始字符串,preprocess 会 tokenize)
raw_batch["task"] = ["pick up the red block"]

print("\n=== Raw batch ===")
for k, v in raw_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} {v.dtype}")
    else:
        print(f"  {k}: {v}")

# === 5. preprocess + 推理 + postprocess ===
processed = preprocess(raw_batch)

print("\n=== Processed batch keys ===")
for k in processed.keys():
    v = processed[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} {v.dtype} on {v.device}")
    else:
        print(f"  {k}: {type(v).__name__}")

with torch.no_grad():
    action = policy.select_action(processed)

# postprocess 把动作反归一化到真实数值范围
action_final = postprocess(action)

print(f"\n✅ Raw action shape: {action.shape}")
print(f"✅ Postprocessed action: {action_final}")
print("\n🎉 VLA inference SUCCESS")