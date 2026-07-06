"""
Regenera figuras de anomaly scores con titulos en español y KDE.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
import json

OUTPUT_DIR = Path('results/maskgit_cls/eval')

GROUP_NAMES = ['SNIa', 'SNIbc', 'SNII', 'SNIIn', 'SLSN', 'PISN']
COLORS = {
    'SNIa': '#1f77b4', 'SNIbc': '#ff7f0e', 'SNII': '#2ca02c',
    'SNIIn': '#d62728', 'SLSN': '#9467bd', 'PISN': '#8c564b'
}

all_scores = np.load(OUTPUT_DIR / 'anomaly_scores_all.npy')
groups = np.load(OUTPUT_DIR / 'groups_all.npy', allow_pickle=True)
with open(OUTPUT_DIR / 'anomaly_scores.json') as f:
    results = json.load(f)

valid_mask = np.isfinite(all_scores)

# KDE plot
fig, ax = plt.subplots(figsize=(10, 6))
x_grid = np.linspace(1.0, 3.7, 500)

for g in GROUP_NAMES:
    mask = (groups == g) & valid_mask
    g_scores = all_scores[mask]
    n = mask.sum()
    kde = gaussian_kde(g_scores, bw_method=0.15)
    density = kde(x_grid)
    ax.plot(x_grid, density, color=COLORS[g], linewidth=2,
            label=f"{g} (n={n:,})")
    ax.fill_between(x_grid, density, alpha=0.15, color=COLORS[g])
    ax.axvline(results[g]['mean'], color=COLORS[g], linestyle='--',
               alpha=0.5, linewidth=1)

ax.set_xlabel('Score de anomalía $(-\log\, p(\mathbf{s}))$', fontsize=12)
ax.set_ylabel('Densidad', fontsize=12)
ax.set_title(u'Distribuci\u00f3n del score de anomal\u00eda por grupo de supernova',
             fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'histogram_scores.png', dpi=150, bbox_inches='tight')
print("Histograma KDE guardado", flush=True)
plt.close()

# Boxplot
fig, ax = plt.subplots(figsize=(8, 6))
order = [g for g, _ in sorted(results.items(), key=lambda x: x[1]['mean'])]
data_boxes = [all_scores[(groups == g) & valid_mask] for g in order]
box_colors = [COLORS[g] for g in order]

bp = ax.boxplot(data_boxes, tick_labels=order, patch_artist=True,
                showfliers=False, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

means = [results[g]['mean'] for g in order]
ax.scatter(range(1, len(order)+1), means, color='black', marker='D',
           s=40, zorder=5, label='Media')

ax.set_ylabel('Score de anomalía $(-\log\, p(\mathbf{s}))$', fontsize=12)
ax.set_title(u'Score de anomal\u00eda por grupo de supernova', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_scores.png', dpi=150, bbox_inches='tight')
print("Boxplot guardado", flush=True)
plt.close()

print("Listo!", flush=True)
