"""Plot M5 LIBERO-Spatial training loss."""
import re
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_file = '/root/autodl-tmp/vla/outputs/libero_spatial_train.log'
out_png = '/root/autodl-tmp/vla/results/05_m5_loss_curve.png'
out_csv = '/root/autodl-tmp/vla/results/05_m5_loss_data.csv'

pattern = re.compile(r'step:(\d+\w?)\s+smpl:(\S+)\s+ep:(\S+)\s+epch:([\d.]+)\s+loss:([\d.]+)\s+grdn:([\d.]+)')

steps, losses, grads = [], [], []
with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            step_str = m.group(1)
            if step_str.endswith('K'):
                step = int(float(step_str[:-1]) * 1000)
            else:
                step = int(step_str)
            steps.append(step)
            losses.append(float(m.group(5)))
            grads.append(float(m.group(6)))

print(f"Parsed {len(steps)} entries")
print(f"Step range: {steps[0]} to {steps[-1]}")
print(f"Loss: {losses[0]:.3f} to {losses[-1]:.3f}")

with open(out_csv, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'loss', 'grad_norm'])
    for s, l, g in zip(steps, losses, grads):
        writer.writerow([s, l, g])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, losses, 'b-', linewidth=1.5)
ax.set_xlabel('Training step')
ax.set_ylabel('Loss (log scale)')
ax.set_yscale('log')
ax.set_title(f'M5: SmolVLA on LIBERO-Spatial | Final loss: {losses[-1]:.4f} | Eval: 60.0% (n=500)')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, dpi=120, bbox_inches='tight')
print(f"Saved {out_png}")
