"""
Diagnostico: visualiza donde el MaskGIT asigna NLL alta (anomalia local).
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

from src.models.maskgit.maskgit import MaskGIT

MASKGIT_CKPT = 'results/maskgit_cls/checkpoints/maskgit-epoch=41-val/loss=3.0498.ckpt'
TOKENS_PATH = 'data/tokens/tokens_cls.npy'
PATHS_FILE = 'data/tokens/paths_cls.txt'
SCORES_FILE = 'results/maskgit_cls/eval/anomaly_scores_all.npy'
OUTPUT_DIR = Path('results/maskgit_cls/diagnose')

N_DIAG = 5
GRID_H, GRID_W = 32, 32


@torch.no_grad()
def nll_per_token(model, tokens, n_rounds=10, mask_ratio=0.5):
    B, T = tokens.shape
    total_nll = torch.zeros(B, T, device=tokens.device)
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
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total_nll += (-token_lp) * mask.float()
        counts += mask.float()
    counts = counts.clamp(min=1.0)
    return total_nll / counts


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    maskgit = MaskGIT.load_from_checkpoint(MASKGIT_CKPT, map_location=device,
                                           strict=False)
    maskgit.eval()
    maskgit.to(device)
    print("MaskGIT cargado OK", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    scores_all = np.load(SCORES_FILE)
    with open(PATHS_FILE) as f:
        paths = [line.strip() for line in f]
    snids = [os.path.basename(p).replace('.png', '') for p in paths]

    top_idx = np.argsort(scores_all)[::-1][:N_DIAG]

    for rank, idx in enumerate(top_idx, 1):
        snid = snids[idx]
        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        local_nll = nll_per_token(maskgit, tokens, n_rounds=10)
        nll_grid = local_nll[0].cpu().numpy().reshape(GRID_H, GRID_W)

        orig = np.asarray(
            Image.open(paths[idx]).convert('RGB').resize((256, 256)),
            dtype=np.float32
        ) / 255.0

        gray = orig.mean(axis=2)
        content_mask_full = gray < 0.9
        content_32 = content_mask_full.reshape(32, 8, 32, 8).mean(axis=(1, 3))
        has_content = content_32 > 0.05

        nll_flat = nll_grid.flatten()
        content_flat = has_content.flatten()
        top_nll_idx = np.argsort(nll_flat)[::-1][:256]
        overlap = content_flat[top_nll_idx].mean()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"#{rank} SNID={snid} score={scores_all[idx]:.3f}  |  "
                     f"top-25pct NLL sobre contenido: {overlap:.1%}", fontsize=12)

        axes[0].imshow(orig)
        axes[0].set_title('Original')
        axes[0].axis('off')

        im1 = axes[1].imshow(nll_grid, cmap='hot')
        axes[1].set_title('NLL por token (32x32)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        axes[2].imshow(orig)
        nll_upscaled = np.kron(nll_grid, np.ones((8, 8)))
        axes[2].imshow(nll_upscaled, cmap='hot', alpha=0.5)
        axes[2].set_title('NLL superpuesta')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'diag_rank{rank:02d}_{snid}.png',
                    dpi=120, bbox_inches='tight')
        plt.close()

        print(f"#{rank} {snid}: top-25pct NLL sobre contenido = {overlap:.1%}",
              flush=True)

    print(f"\nDiagnostico guardado en {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
