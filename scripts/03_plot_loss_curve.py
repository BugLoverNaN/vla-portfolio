"""
03_plot_loss_curve.py
Extract loss/grad_norm/lr from training log and plot the curve.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = "/root/autodl-tmp/vla/outputs/smolvla_so101_m3_train.log"
OUT_PNG = "/root/autodl-tmp/vla/results/03_loss_curve.png"
OUT_CSV = "/root/autodl-tmp/vla/results/03_loss_data.csv"

pattern = re.compile(
    r"step:(\d+\.?\d*[Kk]?).*?loss:([\d.]+).*?grdn:([\d.]+).*?lr:([\d.eE+-]+)"
)

def parse_step(s):
    s = s.strip()
    if s.lower().endswith('k'):
        return int(float(s[:-1]) * 1000)
    return int(s)

steps, losses, grads, lrs = [], [], [], []
with open(LOG_PATH) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(parse_step(m.group(1)))
            losses.append(float(m.group(2)))
            grads.append(float(m.group(3)))
            lrs.append(float(m.group(4)))

print(f"Parsed {len(steps)} training log entries.")
print(f"Step range: {steps[0]} - {steps[-1]}")
print(f"Loss range: {max(losses):.3f} -> {min(losses):.3f}")
print(f"Final loss: {losses[-1]:.3f}")

with open(OUT_CSV, "w") as f:
    f.write("step,loss,grad_norm,lr\n")
    for s, l, g, lr in zip(steps, losses, grads, lrs):
        f.write(f"{s},{l},{g},{lr}\n")
print(f"CSV saved: {OUT_CSV}")

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(steps, losses, color="tab:blue", linewidth=1.5)
axes[0].set_ylabel("Loss")
axes[0].set_title(f"SmolVLA Fine-tuning on SO-101 PickPlace (final loss = {losses[-1]:.3f})")
axes[0].grid(alpha=0.3)
axes[0].set_yscale("log")

axes[1].plot(steps, grads, color="tab:orange", linewidth=1.5)
axes[1].set_ylabel("Grad Norm")
axes[1].grid(alpha=0.3)

axes[2].plot(steps, lrs, color="tab:green", linewidth=1.5)
axes[2].set_ylabel("Learning Rate")
axes[2].set_xlabel("Step")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
print(f"Plot saved: {OUT_PNG}")
