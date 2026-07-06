"""
Generacion de contrafactuales para supernovas anomalas.
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
from tqdm import tqdm

from src.models.maskgit.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

MASKGIT_CKPT = 'results/maskgit_cls/checkpoints/maskgit-epoch=41-val/loss=3.0498.ckpt'
VQVAE_CKPT = 'results/vqvae/checkpoints_cls01/vqvae-epoch=07-train/loss=0.1188.ckpt'
TOKENS_PATH = 'data/tokens/tokens_cls.npy'
PATHS_FILE = 'data/tokens/paths_cls.txt'
LABELS_FILE = 'data/folds/labels.json'
SCORES_FILE = 'results/maskgit_cls/eval/anomaly_scores_all.npy'
OUTPUT_DIR = Path('results/maskgit_cls/counterfactuals')

N_TOP = 20
ANOMALY_QUANTILE = 0.75
N_STEPS = 10
GRID_H, GRID_W = 32, 32

CLASS_NAMES = {
    0: 'SNIa-SALT2', 1: 'SNIa-91bg', 2: 'SNIax',
    3: 'SNII-Templates', 4: 'SNII-NMF', 5: 'SNII+HostXT',
    6: 'SNIb-Templates', 7: 'SNIb+HostXT', 8: 'SNIc-Templates',
    9: 'SNIc+HostXT', 10: 'SNIcBL+HostXT', 11: 'SNIIb+HostXT',
    12: 'SNIIn-MOSFIT', 13: 'SNIIn+HostXT',
    14: 'SLSN-I+host', 15: 'SLSN-I_no_host', 16: 'PISN',
}


@torch.no_grad()
def nll_per_token(model, tokens, n_rounds=5, mask_ratio=0.5):
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


def load_image(path, size=256):
    img = Image.open(path).convert('RGB').resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)

    maskgit = MaskGIT.load_from_checkpoint(MASKGIT_CKPT, map_location=device,
                                           strict=False)
    maskgit.eval()
    maskgit.to(device)
    print("MaskGIT cargado OK", flush=True)

    vqvae = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=device)
    vqvae.eval()
    vqvae.to(device)
    print("VQ-VAE cargado OK", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    scores_all = np.load(SCORES_FILE)

    with open(PATHS_FILE) as f:
        paths = [line.strip() for line in f]

    with open(LABELS_FILE) as f:
        labels_map = json.load(f)

    snids = [os.path.basename(p).replace('.png', '') for p in paths]
    labels = np.array([labels_map.get(s, -1) for s in snids])

    top_idx = np.argsort(scores_all)[::-1][:N_TOP]
    print(f"\nTop {N_TOP} anomalias seleccionadas:", flush=True)
    for rank, idx in enumerate(top_idx, 1):
        cls = CLASS_NAMES.get(labels[idx], 'Unknown')
        print(f"  {rank}. SNID={snids[idx]}  score={scores_all[idx]:.4f}  "
              f"clase={cls}", flush=True)

    print(f"\nGenerando contrafactuales...", flush=True)
    summary = []

    for rank, idx in enumerate(tqdm(top_idx, desc="Contrafactuales"), 1):
        snid = snids[idx]
        cls = CLASS_NAMES.get(labels[idx], 'Unknown')
        score = float(scores_all[idx])

        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        local_nll = nll_per_token(maskgit, tokens, n_rounds=5)
        threshold = torch.quantile(local_nll, ANOMALY_QUANTILE)
        anomaly_mask = local_nll > threshold
        n_anomalous = int(anomaly_mask.sum().item())

        cf_tokens = maskgit.generate_counterfactual(
            tokens, anomaly_mask, n_steps=N_STEPS
        )

        orig_grid = tokens.view(1, GRID_H, GRID_W)
        cf_grid = cf_tokens.view(1, GRID_H, GRID_W)

        with torch.no_grad():
            x_orig_recon = vqvae.decode_from_indices(orig_grid)
            x_cf = vqvae.decode_from_indices(cf_grid)

        recon_img = x_orig_recon[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        cf_img = x_cf[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        orig_img = load_image(paths[idx])

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"#{rank}  SNID={snid}  |  {cls}  |  "
                     f"score={score:.3f}  |  tokens modificados={n_anomalous}/1024",
                     fontsize=12)

        axes[0].imshow(orig_img)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(recon_img)
        axes[1].set_title(u'Reconstrucci\u00f3n VQ-VAE')
        axes[1].axis('off')

        axes[2].imshow(cf_img)
        axes[2].set_title('Contrafactual')
        axes[2].axis('off')

        diff = np.abs(recon_img - cf_img).mean(axis=2)
        im = axes[3].imshow(diff, cmap='hot')
        axes[3].set_title('|Recon - CF|')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f'cf_rank{rank:02d}_{snid}_{cls}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()

        summary.append({
            'rank': rank, 'snid': snid, 'class': cls,
            'anomaly_score': score,
            'n_tokens_modified': n_anomalous,
            'frac_modified': n_anomalous / 1024,
        })

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{N_TOP} contrafactuales generados en {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
