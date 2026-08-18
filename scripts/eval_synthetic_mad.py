"""
Hit rate vs amplitud en MAD, una linea por banda.

Hit = (score_perturbado - score_control) > TAU
TAU por defecto 0.3 (dispersion intra-grupo de waterfall).

Tambien guarda Δscore crudo para re-umbralizar sin re-correr GPU.

Uso: python scripts/eval_synthetic_mad.py [TAU]
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

VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
IMG_DIR      = 'data/images/elasticc_1/synthetic_mad'
MANIFEST     = 'data/synthetic_mad_manifest.json'
OUT          = Path('results/synthetic_eval/mad')
OUT.mkdir(parents=True, exist_ok=True)

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
TAUS = (0.15, 0.3, 0.5)
N_ROUNDS, MASK_RATIO = 10, 0.5
BAND_NAMES = ['u', 'g', 'r', 'i', 'z', 'Y']
KS = list(range(1, 11))


@torch.no_grad()
def score_tokens(model, tokens, n_rounds=10, mask_ratio=0.5):
    B, T = tokens.shape
    total = torch.zeros(B, device=tokens.device)
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
        nll = -(tlp * mask.float()).sum(1) / mask.float().sum(1)
        total += nll
    return total / n_rounds


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
    print('modelos cargados', flush=True)

    scores = {}
    for rec in man:
        p = Path(rec['path'])
        if not p.exists():
            continue
        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x)
            _, _, idx, *_ = vq.codebook(z)
        tok = idx.view(1, -1).long().to(dev)
        sc = float(score_tokens(mg, tok, N_ROUNDS, MASK_RATIO)[0].cpu())
        key = (rec['snid'], rec['tag'])
        scores[key] = sc
        rec['score'] = sc

    by_snid = collections.defaultdict(list)
    for rec in man:
        by_snid[rec['snid']].append(rec)

    dscore = {b: {k: [] for k in KS} for b in BAND_NAMES}
    n_ctrl = 0
    for snid, recs in by_snid.items():
        ctrl = next((r for r in recs if r['tag'] == 'control'), None)
        if ctrl is None or (snid, 'control') not in scores:
            continue
        sc0 = scores[(snid, 'control')]
        n_ctrl += 1
        for r in recs:
            if r['band'] is None:
                continue
            tag = r['tag']
            if (snid, tag) not in scores:
                continue
            ds = scores[(snid, tag)] - sc0
            r['dscore'] = ds
            dscore[r['band']][r['k']].append(ds)

    print(f'\nObjetos con control: {n_ctrl}', flush=True)
    print(f'TAU primario = {TAU}  (hit si Δscore > TAU)', flush=True)

    hit = {b: [] for b in BAND_NAMES}
    mean_ds = {b: [] for b in BAND_NAMES}
    print('\n' + '=' * 78)
    print('HIT RATE por banda  (Δscore > {:.2f})'.format(TAU))
    print('{:<6}'.format('k') + ''.join(f'{b:>8}' for b in BAND_NAMES))
    for k in KS:
        row = f'{k:<6}'
        for b in BAND_NAMES:
            vals = np.array(dscore[b][k], dtype=float)
            if len(vals) == 0:
                hr, md = np.nan, np.nan
            else:
                hr = float((vals > TAU).mean())
                md = float(vals.mean())
            hit[b].append(hr)
            mean_ds[b].append(md)
            row += f'{hr:8.2f}'
        print(row)
    print('=' * 78)

    curves = {}
    for tau in TAUS:
        curves[str(tau)] = {}
        for b in BAND_NAMES:
            curves[str(tau)][b] = [
                float((np.array(dscore[b][k]) > tau).mean()) if dscore[b][k] else None
                for k in KS
            ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for b in BAND_NAMES:
        ax.plot(KS, hit[b], marker='o', lw=1.8, label=b)
    ax.set_xlabel('tamaño de la perturbación (unidades de MAD)')
    ax.set_ylabel(f'hit rate  (Δscore > {TAU})')
    ax.set_xticks(KS)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title='banda', ncol=2, frameon=False)
    ax.set_title('Detección de un punto perturbado vs amplitud en MAD')
    fig.tight_layout()
    fig.savefig(OUT / 'hitrate_by_band.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for b in BAND_NAMES:
        ax.plot(KS, mean_ds[b], marker='o', lw=1.8, label=b)
    ax.axhline(TAU, color='k', ls='--', lw=1, label=f'TAU={TAU}')
    ax.set_xlabel('tamaño de la perturbación (unidades de MAD)')
    ax.set_ylabel('Δscore medio')
    ax.set_xticks(KS)
    ax.grid(True, alpha=0.3)
    ax.legend(title='banda', ncol=2, frameon=False)
    ax.set_title('Δscore medio vs amplitud en MAD')
    fig.tight_layout()
    fig.savefig(OUT / 'dscore_by_band.png', dpi=140)
    plt.close(fig)

    summary = dict(
        tau=TAU, n_objects=n_ctrl, ks=KS, bands=BAND_NAMES,
        hit_rate=hit, mean_dscore=mean_ds, hit_rate_by_tau=curves,
        n_per_cell={b: {str(k): len(dscore[b][k]) for k in KS} for b in BAND_NAMES},
        protocol={
            'mad': 'median(|f - median(f)|) por banda',
            'perturbation': 'un punto existente al azar, f += k * MAD',
            'normalization': 'percentiles del control, congelados',
            'representation': 'waterfall',
        },
    )
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(OUT / 'per_image.json', 'w') as f:
        json.dump(man, f, indent=2)
    print(f'\nFiguras: {OUT}/hitrate_by_band.png  {OUT}/dscore_by_band.png', flush=True)


if __name__ == '__main__':
    main()
