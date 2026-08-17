"""
Interpretabilidad de anomalias para waterfall: atribucion por banda + perfil temporal.
Mapea cada token del grid 32x32 a su banda por POSICION (carriles horizontales),
que es mas robusto que detectar por color.

Uso: python scripts/interpret_anomaly_waterfall.py
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from scripts.train_maskgit_cls import MaskGITWithViz

REP = 'waterfall'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
TOKENS_PATH  = 'data/tokens/tokens_waterfall_full.npy'
PATHS_FILE   = 'data/tokens/paths_waterfall_full.txt'
LABELS_FILE  = 'data/labels_waterfall_full.json'
OUTPUT_DIR   = Path('results/maskgit_full/waterfall/interpret')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 32
N_EXAMPLES = 6
ANOMALY_LABELS = [14, 15, 16]

BAND_NAMES  = {0:'u',1:'g',2:'r',3:'i',4:'z',5:'Y'}
BAND_COLORS = {0:'#7F00FF',1:'#007FFF',2:'#FF7F00',3:'#FF007F',4:'#7FFF00',5:'#00FF7F'}
CLASS_NAMES = {14:'SLSN-I+host',15:'SLSN-I_no_host',16:'PISN'}


def band_map_by_position(grid=GRID, n_bands=6):
    """Mapea cada fila del grid a su banda segun el carril que ocupa.
    En la imagen waterfall la banda de indice mayor esta ARRIBA (fila 0),
    por eso se invierte el indice."""
    bm = np.zeros((grid, grid), dtype=int)
    rows_per_band = grid / n_bands
    for gi in range(grid):
        lane = int(gi // rows_per_band)          # 0 = arriba
        band = (n_bands - 1) - lane              # arriba = banda de indice mayor
        bm[gi, :] = np.clip(band, 0, n_bands - 1)
    return bm


@torch.no_grad()
def nll_per_token(model, tokens, n_rounds=10, mask_ratio=0.5):
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
        logits = model(masked)
        lp = F.log_softmax(logits, dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    return total / counts.clamp(min=1.0)


def content_mask_from_image(img, grid=GRID):
    """True donde el parche tiene contenido (no es fondo blanco)."""
    patch = img.shape[0] // grid
    cm = np.zeros((grid, grid), dtype=bool)
    for gi in range(grid):
        for gj in range(grid):
            p = img[gi*patch:(gi+1)*patch, gj*patch:(gj+1)*patch, :]
            cm[gi, gj] = (p.reshape(-1,3).mean(axis=1) < 0.9).sum() > 0
    return cm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=device, strict=False)
    model.eval(); model.to(device)
    print("Modelo cargado", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    with open(PATHS_FILE) as f:
        paths = [l.strip() for l in f]
    with open(LABELS_FILE) as f:
        labels_map = json.load(f)
    snids = [os.path.basename(p).replace('.png','') for p in paths]
    labels = np.array([labels_map.get(s,-1) for s in snids])
    sel_idx = np.where(np.isin(labels, ANOMALY_LABELS))[0][:N_EXAMPLES]

    band_map = band_map_by_position()

    for rank, idx in enumerate(sel_idx, 1):
        snid = snids[idx]; cls = CLASS_NAMES.get(labels[idx],'?')
        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        nll = nll_per_token(model, tokens, n_rounds=10)[0].cpu().numpy().reshape(GRID, GRID)
        img = np.asarray(Image.open(paths[idx]).convert('RGB').resize((256,256)), dtype=np.float32)/255.0
        content = content_mask_from_image(img)

        # Atribucion por banda: percentil 90 del NLL entre tokens CON CONTENIDO de esa banda
        band_nll = np.zeros(6)
        for b in range(6):
            mb = (band_map == b) & content
            if mb.sum() > 0:
                band_nll[b] = np.percentile(nll[mb], 90)
        band_frac = band_nll / band_nll.sum() if band_nll.sum() > 0 else band_nll

        # Perfil temporal: NLL medio por columna, solo tokens con contenido
        temporal = np.zeros(GRID)
        for c in range(GRID):
            col = content[:, c]
            if col.sum() > 0:
                temporal[c] = nll[col, c].mean()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'[waterfall] {snid} | {cls}', fontsize=13)

        axes[0].imshow(img); axes[0].set_title('Waterfall original'); axes[0].axis('off')

        im = axes[1].imshow(nll, cmap='hot')
        axes[1].set_title('NLL por token'); axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        colors_list = [BAND_COLORS[b] for b in range(6)]
        axes[2].barh(range(6), band_frac, color=colors_list, edgecolor='black')
        axes[2].set_yticks(range(6))
        axes[2].set_yticklabels([BAND_NAMES[b] for b in range(6)])
        axes[2].set_xlabel('Contribucion relativa al score')
        axes[2].set_title('Atribucion por banda')

        axes[3].plot(np.linspace(0, 1, GRID), temporal, color='crimson', lw=2)
        axes[3].set_xlabel('Tiempo normalizado')
        axes[3].set_ylabel('NLL medio')
        axes[3].set_title('Perfil temporal de anomalia')
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f'interpret_{snid}_{cls}.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  {rank}/{len(sel_idx)}  {snid} {cls}  banda dominante: {BAND_NAMES[int(band_frac.argmax())]}", flush=True)

    print(f"[waterfall] Listo en {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
