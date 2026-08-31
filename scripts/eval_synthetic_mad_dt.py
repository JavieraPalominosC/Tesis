"""
Detector del profe: maximo global del mapa NLL por token.

  score_ij = -log p(s_ij | resto)   (32x32)
  m        = max_ij score_ij

Se compara la distribucion de m en controles vs cada (df, dt).
Umbral tau = percentil 95 de los controles (FPR ~ 5%).
F1 de deteccion: control = negativo, perturbado = positivo.

Tambien guarda el score medio (detector viejo) para contrastar.

Uso: python scripts/eval_synthetic_mad_dt.py [REP]
     REP in {waterfall, waterfall_global}
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

REP = sys.argv[1] if len(sys.argv) > 1 else 'waterfall_global'
assert REP in ('waterfall', 'waterfall_global')

VQVAE_CKPTS = {
    'waterfall':        'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt',
    'waterfall_global': 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt',
}
MASKGIT_CKPTS = {
    'waterfall':        'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt',
    'waterfall_global': 'results/maskgit_full/waterfall_global/checkpoints/maskgit-epoch=20-val/loss=3.6256.ckpt',
}
PATHS = {
    'waterfall': dict(
        img='data/images/elasticc_1/synthetic_mad_dt',
        man='data/synthetic_mad_dt_manifest.json',
        out='results/synthetic_eval/mad_dt',
    ),
    'waterfall_global': dict(
        img='data/images/elasticc_1/synthetic_mad_dt_wfg',
        man='data/synthetic_mad_dt_wfg_manifest.json',
        out='results/synthetic_eval/mad_dt_wfg',
    ),
}

VQVAE_CKPT = VQVAE_CKPTS[REP]
MASKGIT_CKPT = MASKGIT_CKPTS[REP]
IMG_DIR = PATHS[REP]['img']
MANIFEST = PATHS[REP]['man']
OUT = Path(PATHS[REP]['out'])
OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
N_ROUNDS, MASK_RATIO = 10, 0.5
CTRL_Q = 0.95
N_MAP_SHOW = 6


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
    tn = int((neg <= tau).sum())
    tp = int((pos > tau).sum())
    fn = int((pos <= tau).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1)


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
    print(f'[{REP}] modelos cargados  n_img={len(man)}', flush=True)

    rows = []
    maps_saved = 0
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
        rec_out = dict(
            snid=rec['snid'], tag=rec['tag'], df=rec.get('df'), dt=rec.get('dt'),
            n_pts=rec.get('n_pts'), mad=rec.get('mad'), mjd_c=rec.get('mjd_c'),
            max_nll=float(nll.max()), mean_nll=float(nll.mean()),
        )
        rows.append(rec_out)

        want_map = rec['tag'] in ('control', 'df10_dt7', 'df100_dt1', 'df200_dt1')
        if want_map and maps_saved < N_MAP_SHOW * 2:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(nll, cmap='inferno')
            ax.set_title(f'{rec["snid"]} {rec["tag"]}\nmax={nll.max():.2f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(OUT / f'map_{rec["snid"]}_{rec["tag"]}.png', dpi=110)
            plt.close(fig)
            maps_saved += 1
        print(f'  {rec["snid"]} {rec["tag"]}: max={rec_out["max_nll"]:.3f}  '
              f'mean={rec_out["mean_nll"]:.3f}', flush=True)

    by_snid = collections.defaultdict(dict)
    for r in rows:
        by_snid[r['snid']][r['tag']] = r

    ctrl = [by_snid[s]['control']['max_nll']
            for s in by_snid if 'control' in by_snid[s]]
    ctrl_mean = [by_snid[s]['control']['mean_nll']
                 for s in by_snid if 'control' in by_snid[s]]
    tau = float(np.quantile(ctrl, CTRL_Q))
    tau_mean = float(np.quantile(ctrl_mean, CTRL_Q))
    print(f'\ncontroles n={len(ctrl)}  tau_max(q{CTRL_Q})={tau:.4f}  '
          f'tau_mean(q{CTRL_Q})={tau_mean:.4f}', flush=True)

    dfs = sorted({r['df'] for r in rows if r['df'] is not None})
    dts = sorted({r['dt'] for r in rows if r['dt'] is not None})

    grid_f1 = np.zeros((len(dts), len(dfs)))
    grid_rec = np.zeros_like(grid_f1)
    grid_dmax = np.zeros_like(grid_f1)
    grid_dmean = np.zeros_like(grid_f1)
    per_cell = {}
    for i, dt in enumerate(dts):
        for j, df_k in enumerate(dfs):
            tag = f'df{df_k}_dt{dt}'
            pos, dmax, dmean = [], [], []
            for s, d in by_snid.items():
                if 'control' not in d or tag not in d:
                    continue
                pos.append(d[tag]['max_nll'])
                dmax.append(d[tag]['max_nll'] - d['control']['max_nll'])
                dmean.append(d[tag]['mean_nll'] - d['control']['mean_nll'])
            met = f1_from_scores(ctrl, pos, tau)
            met_mean = f1_from_scores(ctrl_mean, [
                by_snid[s][tag]['mean_nll']
                for s in by_snid if 'control' in by_snid[s] and tag in by_snid[s]
            ], tau_mean)
            grid_f1[i, j] = met['f1']
            grid_rec[i, j] = met['recall']
            grid_dmax[i, j] = float(np.mean(dmax)) if dmax else 0.0
            grid_dmean[i, j] = float(np.mean(dmean)) if dmean else 0.0
            per_cell[tag] = dict(
                n=len(pos),
                mean_max=float(np.mean(pos)) if pos else None,
                mean_dmax=grid_dmax[i, j],
                mean_dmean=grid_dmean[i, j],
                max_detector=met,
                mean_detector=met_mean,
            )
            print(f'  {tag}: n_pts medio ver manifest  F1_max={met["f1"]:.3f}  '
                  f'recall={met["recall"]:.3f}  F1_mean={met_mean["f1"]:.3f}  '
                  f'Δmax={grid_dmax[i, j]:+.3f}  Δmean={grid_dmean[i, j]:+.3f}',
                  flush=True)

    # --- figuras ---
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(grid_f1, origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(dfs)), [str(x) for x in dfs])
    ax.set_yticks(range(len(dts)), [str(x) for x in dts])
    ax.set_xlabel('df (veces MAD)')
    ax.set_ylabel('dt (dias MJD)')
    ax.set_title(f'F1 deteccion  max NLL  [{REP}]\n'
                 f'tau = P{int(CTRL_Q*100)} controles = {tau:.2f}')
    for i in range(len(dts)):
        for j in range(len(dfs)):
            ax.text(j, i, f'{grid_f1[i, j]:.2f}', ha='center', va='center',
                    color='white' if grid_f1[i, j] < 0.55 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / 'f1_max_nll.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(grid_rec, origin='lower', vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(dfs)), [str(x) for x in dfs])
    ax.set_yticks(range(len(dts)), [str(x) for x in dts])
    ax.set_xlabel('df (veces MAD)')
    ax.set_ylabel('dt (dias MJD)')
    ax.set_title(f'Recall  max NLL  [{REP}]')
    for i in range(len(dts)):
        for j in range(len(dfs)):
            ax.text(j, i, f'{grid_rec[i, j]:.2f}', ha='center', va='center',
                    color='white' if grid_rec[i, j] < 0.55 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / 'recall_max_nll.png', dpi=130)
    plt.close(fig)

    # distribuciones: control vs df a dt fijo (el mas largo y el de 1 dia)
    for dt_show in (dts[0], dts[-1]):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        data = [ctrl]
        labels = ['control']
        for df_k in dfs:
            tag = f'df{df_k}_dt{dt_show}'
            data.append([by_snid[s][tag]['max_nll']
                         for s in by_snid
                         if 'control' in by_snid[s] and tag in by_snid[s]])
            labels.append(f'df={df_k}')
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(tau, color='crimson', ls='--', lw=1,
                   label=f'tau P{int(CTRL_Q*100)}={tau:.2f}')
        ax.set_ylabel('max_ij  -log p(s_ij)')
        ax.set_title(f'Distribucion del maximo NLL  dt={dt_show} d  [{REP}]')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f'dist_max_dt{dt_show}.png', dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, dt in enumerate(dts):
        ax.plot(dfs, grid_dmax[i], 'o-', label=f'dt={dt} d')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('df (veces MAD)')
    ax.set_ylabel('Δ max NLL  (pert - control)')
    ax.set_title(f'Subida del maximo vs amplitud  [{REP}]')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'dmax_vs_df.png', dpi=130)
    plt.close(fig)

    summary = dict(
        rep=REP, ctrl_quantile=CTRL_Q, tau_max=tau, tau_mean=tau_mean,
        n_control=len(ctrl),
        ctrl_max_mean=float(np.mean(ctrl)),
        ctrl_max_std=float(np.std(ctrl)),
        dfs=dfs, dts=dts,
        f1_max=grid_f1.tolist(),
        recall_max=grid_rec.tolist(),
        dmax=grid_dmax.tolist(),
        dmean=grid_dmean.tolist(),
        per_cell=per_cell,
        per_image=rows,
    )
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
