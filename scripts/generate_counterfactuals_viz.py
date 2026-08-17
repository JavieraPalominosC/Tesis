"""
Contrafactuales con visualizacion mejorada:
 - panel de superposicion recon vs CF por banda (curvas encima, no lado a lado)
 - heatmap con contraste realzado (percentiles)
 - quantile configurable para hacer el CF mas agresivo

Uso: python scripts/generate_counterfactuals_viz.py <REP> [QUANTILE]
     ej: python scripts/generate_counterfactuals_viz.py waterfall 0.70
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

REP = sys.argv[1]
ANOMALY_QUANTILE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85

VQVAE_CKPTS = {
    'overlay_lines':    'results/vqvae_overlay_full/checkpoints/vqvae-epoch=17-train/loss=0.1378.ckpt',
    'points':           'results/vqvae_points_full/checkpoints/vqvae-epoch=52-train/loss=0.0658.ckpt',
    'waterfall':        'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt',
    'waterfall_global': 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt',
}
MASKGIT_CKPTS = {
    'overlay_lines':    'results/maskgit_full/overlay_lines/checkpoints/maskgit-epoch=14-val/loss=2.8207.ckpt',
    'points':           'results/maskgit_full/points/checkpoints/maskgit-epoch=13-val/loss=2.9244.ckpt',
    'waterfall':        'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt',
}
LABELS_PATHS = {
    'overlay_lines':    'data/labels_overlay_full.json',
    'points':           'data/labels_points_full.json',
    'waterfall':        'data/labels_waterfall_full.json',
    'waterfall_global': 'data/labels_waterfall_global_full.json',
}

MASKGIT_CKPT = MASKGIT_CKPTS[REP]
VQVAE_CKPT   = VQVAE_CKPTS[REP]
TOKENS_PATH  = f'data/tokens/tokens_{REP}_full.npy'
PATHS_FILE   = f'data/tokens/paths_{REP}_full.txt'
LABELS_FILE  = LABELS_PATHS[REP]
OUTPUT_DIR   = Path(f'results/maskgit_full/{REP}/counterfactuals_viz_q{int(ANOMALY_QUANTILE*100)}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOP = 6
N_STEPS = 12
GRID_H, GRID_W = 32, 32
ANOMALY_LABELS = [14, 15, 16]
N_BANDS = 6
BAND_NAMES = ['u','g','r','i','z','Y']

CLASS_NAMES = {14:'SLSN-I+host',15:'SLSN-I_no_host',16:'PISN'}


@torch.no_grad()
def nll_per_token(model, tokens, n_rounds=5, mask_ratio=0.5):
    B, T = tokens.shape
    total = torch.zeros(B, T, device=tokens.device)
    counts = torch.zeros(B, T, device=tokens.device)
    for _ in range(n_rounds):
        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        masked = tokens.clone(); masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float(); counts += mask.float()
    return total / counts.clamp(min=1.0)


def trace_curve(img_rgb, row0, row1):
    """Extrae un trazo aproximado de la banda: por cada columna, la fila media
    de los pixeles no-blancos dentro del carril [row0,row1)."""
    H, W, _ = img_rgb.shape
    sub = img_rgb[row0:row1, :, :]
    nonwhite = sub.mean(axis=2) < 0.9
    ys = np.full(W, np.nan)
    for c in range(W):
        idx = np.where(nonwhite[:, c])[0]
        if len(idx):
            ys[c] = row0 + idx.mean()
    return ys


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{REP}] quantile={ANOMALY_QUANTILE}  MaskGIT: {MASKGIT_CKPT}", flush=True)

    maskgit = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=device, strict=False)
    maskgit.eval(); maskgit.to(device)
    vqvae = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=device)
    vqvae.eval(); vqvae.to(device)
    print("Modelos cargados", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    with open(PATHS_FILE) as f: paths = [l.strip() for l in f]
    with open(LABELS_FILE) as f: labels_map = json.load(f)
    snids = [os.path.basename(p).replace('.png','') for p in paths]
    labels = np.array([labels_map.get(s,-1) for s in snids])
    sel_idx = np.where(np.isin(labels, ANOMALY_LABELS))[0][:N_TOP]

    for rank, idx in enumerate(tqdm(sel_idx), 1):
        snid = snids[idx]; cls = CLASS_NAMES.get(labels[idx],'?')
        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        local_nll = nll_per_token(maskgit, tokens, n_rounds=5)
        thr = torch.quantile(local_nll, ANOMALY_QUANTILE)
        amask = local_nll > thr
        n_mod = int(amask.sum().item())

        cf_tokens = maskgit.generate_counterfactual(tokens, amask, n_steps=N_STEPS)

        with torch.no_grad():
            x_recon = vqvae.decode_from_indices(tokens.view(1,GRID_H,GRID_W))
            x_cf    = vqvae.decode_from_indices(cf_tokens.view(1,GRID_H,GRID_W))
        recon = x_recon[0].cpu().permute(1,2,0).numpy().clip(0,1)
        cf    = x_cf[0].cpu().permute(1,2,0).numpy().clip(0,1)

        diff = np.abs(recon-cf).mean(axis=2)
        vmax = np.percentile(diff, 99.5)    # contraste realzado
        vmin = np.percentile(diff, 60)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        fig.suptitle(f"[{REP} q={ANOMALY_QUANTILE}] {snid} | {cls} | tokens mod={n_mod}/1024", fontsize=12)

        axes[0].imshow(recon); axes[0].set_title('Reconstruccion'); axes[0].axis('off')

        # Superposicion por banda: recon (solido) vs CF (punteado)
        H = recon.shape[0]
        rows_per_band = H // N_BANDS
        for b in range(N_BANDS):
            r0, r1 = b*rows_per_band, (b+1)*rows_per_band
            y_rec = trace_curve(recon, r0, r1)
            y_cf  = trace_curve(cf,    r0, r1)
            band_label = BAND_NAMES[(N_BANDS-1)-b]   # arriba = indice mayor
            axes[1].plot(y_rec, lw=1.6, label=f'{band_label} recon' if b==0 else None, color=f'C{b}')
            axes[1].plot(y_cf, lw=1.4, ls='--', color=f'C{b}')
        axes[1].invert_yaxis()
        axes[1].set_title('Recon (solido) vs Contrafactual (punteado)')
        axes[1].set_xlabel('Tiempo (columna de pixel)')
        axes[1].set_ylabel('Posicion vertical (banda)')
        axes[1].grid(alpha=0.25)

        im = axes[2].imshow(diff, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[2].set_title('|Recon-CF| (contraste realzado)'); axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f'cfviz_{snid}_{cls}.png', dpi=120, bbox_inches='tight')
        plt.close()

    print(f"[{REP}] {len(sel_idx)} figuras en {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
