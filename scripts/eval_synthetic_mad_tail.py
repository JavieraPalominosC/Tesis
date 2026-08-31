"""
F1 del maximo NLL sobre MAD en la cola, escala SIN congelar.

Uso: python scripts/eval_synthetic_mad_tail.py
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json
import collections
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT = 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall_global/checkpoints/maskgit-epoch=20-val/loss=3.6256.ckpt'
MANIFEST = 'data/synthetic_mad_tail_manifest.json'
OUT = Path('results/synthetic_eval/mad_tail')
OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
N_ROUNDS, MASK_RATIO = 10, 0.5
CTRL_Q = 0.95
TAGS_ORDER = [f'df{k}_unfrozen' for k in range(10, 101, 10)]


def pretty_label(tag):
    if tag == 'control':
        return 'control'
    if tag.startswith('df') and tag.endswith('_unfrozen'):
        return f"{tag[2:-9]} MAD"
    return tag


@torch.no_grad()
def nll_map(model, tokens, n_rounds=10, mask_ratio=0.5):
    B, T = tokens.shape
    total = torch.zeros(B, T, device=tokens.device)
    counts = torch.zeros(B, T, device=tokens.device)
    for _ in range(n_rounds):
        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        masked = tokens.clone()
        masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    return total / counts.clamp(min=1.0)


def f1_from_scores(neg, pos, tau):
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    fp = int((neg > tau).sum())
    tp = int((pos > tau).sum())
    fn = int((pos <= tau).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pos=int(len(pos)))


def main():
    with open(MANIFEST) as f:
        man = json.load(f)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev)
    vq.eval()
    vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval()
    mg.to(dev)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    print(f'modelos cargados  n_img={len(man)}', flush=True)

    rows = []
    maps_ok = 0
    for rec in man:
        p = Path(rec['path'])
        if not p.exists():
            continue
        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x)
            _, _, idx, *_ = vq.codebook(z)
        tok = idx.view(1, -1).long().to(dev)
        nll = nll_map(mg, tok, N_ROUNDS, MASK_RATIO)[0].cpu().numpy().reshape(GRID, GRID)
        out = dict(
            snid=rec['snid'], tag=rec['tag'], df=rec.get('df'),
            frozen=rec.get('frozen'), n_pts=rec.get('n_pts'),
            max_nll=float(nll.max()), mean_nll=float(nll.mean()),
        )
        rows.append(out)
        if rec['tag'] in ('control', 'df20_unfrozen', 'df50_unfrozen', 'df100_unfrozen') and maps_ok < 8:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(nll, cmap='inferno')
            ax.set_title(f'{rec["snid"]} {rec["tag"]}\nmax={nll.max():.2f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(OUT / f'map_{rec["snid"]}_{rec["tag"]}.png', dpi=110)
            plt.close(fig)
            maps_ok += 1
        print(f'  {rec["snid"]} {rec["tag"]}: max={out["max_nll"]:.3f}  '
              f'mean={out["mean_nll"]:.3f}', flush=True)

    by = collections.defaultdict(dict)
    for r in rows:
        by[r['snid']][r['tag']] = r

    ctrl = [by[s]['control']['max_nll'] for s in by if 'control' in by[s]]
    ctrl_mean = [by[s]['control']['mean_nll'] for s in by if 'control' in by[s]]
    tau = float(np.quantile(ctrl, CTRL_Q))
    tau_mean = float(np.quantile(ctrl_mean, CTRL_Q))
    print(f'\ncontroles n={len(ctrl)}  tau_max={tau:.4f}  tau_mean={tau_mean:.4f}',
          flush=True)

    tags = [t for t in TAGS_ORDER if any(t in by[s] for s in by)]
    per_tag = {}
    for tag in tags:
        pos = [by[s][tag]['max_nll'] for s in by if 'control' in by[s] and tag in by[s]]
        pos_m = [by[s][tag]['mean_nll'] for s in by if 'control' in by[s] and tag in by[s]]
        dmax = [by[s][tag]['max_nll'] - by[s]['control']['max_nll']
                for s in by if 'control' in by[s] and tag in by[s]]
        dmean = [by[s][tag]['mean_nll'] - by[s]['control']['mean_nll']
                 for s in by if 'control' in by[s] and tag in by[s]]
        met = f1_from_scores(ctrl, pos, tau)
        met_m = f1_from_scores(ctrl_mean, pos_m, tau_mean)
        per_tag[tag] = dict(
            mean_max=float(np.mean(pos)),
            mean_dmax=float(np.mean(dmax)),
            mean_dmean=float(np.mean(dmean)),
            max_detector=met,
            mean_detector=met_m,
        )
        print(f'  {tag}: F1_max={met["f1"]:.3f}  rec={met["recall"]:.3f}  '
              f'F1_mean={met_m["f1"]:.3f}  Δmax={np.mean(dmax):+.3f}  '
              f'Δmean={np.mean(dmean):+.3f}', flush=True)

    labels = [pretty_label(t) for t in tags]
    f1s = [per_tag[t]['max_detector']['f1'] for t in tags]
    recs = [per_tag[t]['max_detector']['recall'] for t in tags]
    f1m = [per_tag[t]['mean_detector']['f1'] for t in tags]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(x, f1s, color='steelblue')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    for xi, v in zip(x, f1s):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('perturbacion (veces MAD)')
    ax.set_ylabel('F1')
    ax.set_title(f'Deteccion cola MAD, escala libre\n'
                 f'tau max P{int(CTRL_Q*100)}={tau:.2f}')
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
    for tag in tags:
        data.append([by[s][tag]['max_nll']
                     for s in by if 'control' in by[s] and tag in by[s]])
        labs.append(pretty_label(tag))
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

    with open(OUT / 'summary.json', 'w') as f:
        json.dump({
            'ctrl_quantile': CTRL_Q, 'tau_max': tau, 'tau_mean': tau_mean,
            'n_control': len(ctrl), 'per_tag': per_tag, 'per_image': rows,
        }, f, indent=2)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
