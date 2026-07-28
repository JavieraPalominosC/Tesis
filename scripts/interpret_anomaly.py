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
from glob import glob

from scripts.train_maskgit_cls import MaskGITWithViz

REP = 'overlay'
MASKGIT_CKPT = glob(f'results/maskgit_exp/{REP}/checkpoints/maskgit-epoch=29-val/*.ckpt')[0]
TOKENS_PATH = f'data/tokens/tokens_{REP}_exp.npy'
PATHS_FILE = f'data/tokens/paths_{REP}_exp.txt'
LABELS_FILE = 'data/images_exp/labels_subset.json'
OUTPUT_DIR = Path(f'results/maskgit_exp/{REP}/interpret')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 32
PATCH = 8
N_EXAMPLES = 6
ANOMALY_LABELS = [14, 15, 16]

BAND_COLORS = {0:'#00FF7F',1:'#7FFF00',2:'#FF007F',3:'#FF7F00',4:'#007FFF',5:'#7F00FF'}
BAND_NAMES = {0:'u',1:'g',2:'r',3:'i',4:'z',5:'Y'}
CLASS_NAMES = {14:'SLSN-I+host',15:'SLSN-I_no_host',16:'PISN'}

def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i+2],16) for i in (0,2,4)], dtype=np.float32)/255.0
BAND_RGB = {k: hex_to_rgb(v) for k,v in BAND_COLORS.items()}


def dominant_band_per_patch(img):
    band_map = np.full((GRID,GRID), -1, dtype=int)
    for gi in range(GRID):
        for gj in range(GRID):
            patch = img[gi*PATCH:(gi+1)*PATCH, gj*PATCH:(gj+1)*PATCH, :]
            pixels = patch.reshape(-1,3)
            non_white = pixels[pixels.mean(axis=1) < 0.9]
            if len(non_white) == 0:
                continue
            presence = np.zeros(6)
            for band_id, rgb in BAND_RGB.items():
                presence[band_id] = (np.linalg.norm(non_white - rgb, axis=1) < 0.3).sum()
            if presence.sum() > 0:
                band_map[gi,gj] = presence.argmax()
    return band_map


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

    for rank, idx in enumerate(sel_idx, 1):
        snid = snids[idx]; cls = CLASS_NAMES.get(labels[idx],'?')
        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        # NLL por token -> grid 32x32
        nll = nll_per_token(model, tokens, n_rounds=10)[0].cpu().numpy().reshape(GRID, GRID)

        # band_map de la imagen
        img = np.asarray(Image.open(paths[idx]).convert('RGB').resize((256,256)), dtype=np.float32)/255.0
        band_map = dominant_band_per_patch(img)

        # ── #2: atribucion por banda ──
        # Sumar NLL de los tokens que pertenecen a cada banda
        band_nll = np.zeros(6)
        band_count = np.zeros(6)
        for b in range(6):
            mask_b = band_map == b
            if mask_b.sum() > 0:
                band_nll[b] = np.percentile(nll[mask_b], 90)
                band_count[b] = mask_b.sum()
        # Normalizar a fraccion (contribucion relativa a la anomalia)
        band_frac = band_nll / band_nll.sum() if band_nll.sum() > 0 else band_nll

        # ── #5: perfil temporal ──
        # NLL promedio por columna, solo sobre tokens con contenido
        content = band_map >= 0
        temporal = np.zeros(GRID)
        for c in range(GRID):
            col_content = content[:, c]
            if col_content.sum() > 0:
                temporal[c] = nll[col_content, c].mean()

        # ── Figura ──
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'[interpretabilidad] {snid} | {cls}', fontsize=13)

        axes[0].imshow(img)
        axes[0].set_title('Overlay original')
        axes[0].axis('off')

        im = axes[1].imshow(nll, cmap='hot')
        axes[1].set_title('NLL por token')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Atribucion por banda (barra horizontal)
        colors_list = [BAND_COLORS[b] for b in range(6)]
        axes[2].barh(range(6), band_frac, color=colors_list, edgecolor='black')
        axes[2].set_yticks(range(6))
        axes[2].set_yticklabels([BAND_NAMES[b] for b in range(6)])
        axes[2].set_xlabel('Fracción de anomalía')
        axes[2].set_title('¿Qué banda es anómala?')
        axes[2].invert_yaxis()

        # Perfil temporal
        axes[3].plot(range(GRID), temporal, color='crimson', linewidth=2)
        axes[3].fill_between(range(GRID), temporal, alpha=0.3, color='crimson')
        axes[3].set_xlabel('Tiempo (columna del grid)')
        axes[3].set_ylabel('NLL media')
        axes[3].set_title('¿Cuándo es anómala?')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f'interp_rank{rank:02d}_{snid}_{cls}.png', dpi=120, bbox_inches='tight')
        plt.close()

        # Imprimir resumen textual
        top_band = BAND_NAMES[band_frac.argmax()]
        peak_time = temporal.argmax()
        print(f"#{rank} {snid} ({cls}): banda mas anomala={top_band} "
              f"({band_frac.max():.1%}), pico temporal en col {peak_time}", flush=True)

    print(f"\nInterpretaciones en {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
