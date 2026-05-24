"""
Plot LIBERO-Spatial training loss curve from M5 training log.
"""
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless

log_file = '/root/autodl-tmp/vla/outputs/libero_spatial_train.log'
out_png = '/root/autodl-tmp/vla/results/05_m5_loss_curve.png'
out_csv = '/root/autodl-tmp/vla/results/05_m5_loss_data.csv'

# Parse log lines like: "step:100 smpl:800 ep:7 epch:0.02 loss:1.086 grdn:5.672 ..."
pattern = re.compile(r'step:(\d+\w?)\s+smpl:(\S+)\s+ep:(\S+)\s+epch:([\d.]+)\s+loss:([\d.]+)\s+grdn:([\d.]+)')

steps, losses, grads = [], [], []
with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            step_str = m.group(1)
            # Handle "20K" -> 20000 style
            if step_str.endswith('K'):
                step = int(float(step_str[:-1]) * 1000)
            else:
                step = int(step_str)
            loss = float(m.group(5))
            grad = float(m.group(6))
            steps.append(step)
            losses.append(loss)
            grads.append(grad)

print(f"Parsed {len(steps)} log entries")
print(f"Step range: {steps[0]} - {steps[-1]}")
print(f"Loss range: {min(losses):.3f} - {max(losses):.3f}")
print(f"Final loss: {losses[-1]:.4f}")

# Save CSV
import csv
with open(out_csv, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'loss', 'grad_norm'])
    for s, l, g in zip(steps, losses, grads):
        writer.writerow([s, l, g])
print(f"✅ CSV saved: {out_csv}")

# Plot loss with log scale
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, losses, 'b-', linewidth=1.5, label='Loss')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss (log scale)')
ax.set_yscale('log')
ax.set_title(f'M5: SmolVLA fine-tuning on LIBERO-Spatial\n'
             f'Final loss: {losses[-1]:.4f} | Total steps: {steps[-1]:,}')
ax.grid(True, which='both', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(out_png, dpi=120, bbox_inches='tight')
print(f"✅ Plot saved: {out_png}")
