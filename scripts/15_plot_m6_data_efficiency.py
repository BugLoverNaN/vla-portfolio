"""Plot M6 data-efficiency curve from eval_info.json files."""
import json, os, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATHS = {n: f'/root/autodl-tmp/vla/eval_results/m6_data_eff/m6_{n}ep/eval_info.json'
         for n in [10, 30, 50, 100, 200]}
M5A_432 = 60.0  # M5a full-Spatial baseline (separate run)

eps, rates = [], []
print("M6 Data Efficiency Results\n" + "=" * 40)
for n, p in PATHS.items():
    if os.path.exists(p):
        r = json.load(open(p))['overall']['pc_success']
        print(f"{n:4d} ep: {r:5.1f}%"); eps.append(n); rates.append(r)
    else:
        print(f"{n:4d} ep: MISSING (skipped)")
print(f" 432 ep: {M5A_432:5.1f}% (M5a baseline)")

os.makedirs('/root/autodl-tmp/vla/results', exist_ok=True)
with open('/root/autodl-tmp/vla/results/06_m6_data_efficiency_data.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['n_episodes', 'success_rate'])
    for e, r in zip(eps, rates): w.writerow([e, r])
    w.writerow([432, M5A_432])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(eps, rates, 'o-', lw=2.5, ms=11, color='#2E86AB',
        label='M6: SmolVLA fine-tuned (n_eval=500)')
for e, r in zip(eps, rates):
    ax.annotate(f'{r:.1f}%', (e, r), xytext=(0, 11), textcoords='offset points',
                ha='center', fontsize=9.5, fontweight='bold')
ax.plot(432, M5A_432, '*', ms=20, color='#E1A100', label='M5a baseline: 432 ep (separate run)')
ax.annotate(f'{M5A_432:.1f}%', (432, M5A_432), xytext=(0, -20),
            textcoords='offset points', ha='center', fontsize=9.5)
ax.set_xscale('log'); ax.set_xticks(eps + [432])
ax.set_xticklabels([str(e) for e in eps] + ['432'])
ax.set_xlabel('Training episodes (log scale)'); ax.set_ylabel('LIBERO-Spatial success rate (%)')
ax.set_title('M6 — SmolVLA data efficiency on LIBERO-Spatial')
ax.set_ylim(0, 75); ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('/root/autodl-tmp/vla/results/06_m6_data_efficiency_curve.png', dpi=150, bbox_inches='tight')
print("\n✅ saved results/06_m6_data_efficiency_curve.png + .csv")
