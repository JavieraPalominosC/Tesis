"""Rehace las figuras del eval cola MAD desde summary.json (sin GPU)."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path('results/synthetic_eval/mad_tail/summary.json')
OUT = Path(os.environ.get('MAD_TAIL_FIG_OUT', str(SRC.parent)))
TAGS = [f'df{k}_unfrozen' for k in range(10, 101, 10)]


def pretty(tag):
    if tag == 'control':
        return 'control'
    if tag.startswith('df') and tag.endswith('_unfrozen'):
        return f'{tag[2:-9]} MAD'
    return tag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(SRC))
    tau = d['tau_max']
    rows = d['per_image']
    by = {}
    for r in rows:
        by.setdefault(r['snid'], {})[r['tag']] = r
    ctrl = [by[s]['control']['max_nll'] for s in by if 'control' in by[s]]
    tags = [t for t in TAGS if any(t in by[s] for s in by)]
    labels = [pretty(t) for t in tags]
    pt = d['per_tag']
    f1s = [pt[t]['max_detector']['f1'] for t in tags]
    recs = [pt[t]['max_detector']['recall'] for t in tags]
    x = np.arange(len(tags))

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(x, f1s, color='steelblue')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    for xi, v in zip(x, f1s):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('perturbacion (veces MAD)')
    ax.set_ylabel('F1')
    ax.set_title(f'Deteccion cola MAD, escala libre\ntau max P95={tau:.2f}')
    fig.tight_layout()
    fig.savefig(OUT / 'f1_vs_tag.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(x, recs, color='darkorange')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    for xi, v in zip(x, recs):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('perturbacion (veces MAD)')
    ax.set_ylabel('recall')
    ax.set_title('Recall del maximo NLL')
    fig.tight_layout()
    fig.savefig(OUT / 'recall_vs_tag.png', dpi=130)
    plt.close(fig)

    data, labs = [ctrl], ['control']
    for t in tags:
        data.append([by[s][t]['max_nll'] for s in by if t in by[s]])
        labs.append(pretty(t))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.boxplot(data, tick_labels=labs, showfliers=False)
    ax.axhline(tau, color='crimson', ls='--', lw=1, label=f'tau={tau:.2f}')
    ax.set_ylabel('max_ij  -log p')
    ax.set_xlabel('perturbacion (veces MAD)')
    ax.set_title('Distribucion del maximo NLL')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'dist_max.png', dpi=130)
    plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'Figuras en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
