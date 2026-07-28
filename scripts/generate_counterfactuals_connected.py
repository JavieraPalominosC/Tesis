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
from tqdm import tqdm
from scipy import ndimage

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

REP = sys.argv[1] if len(sys.argv) > 1 else 'overlay'
assert REP in ('filled', 'overlay')

VQVAE_CKPTS = {
    'filled':  'results/vqvae_exp/filled/checkpoints/vqvae-epoch=17-train/loss=0.1704.ckpt',
    'overlay': 'results/vqvae_exp/overlay/checkpoints/vqvae-epoch=18-train/loss=0.1816.ckpt',
}
MASKGIT_CKPT = glob(f'results/maskgit_exp/{REP}/checkpoints/maskgit-epoch=29-val/*.ckpt')[0]
VQVAE_CKPT = VQVAE_CKPTS[REP]
TOKENS_PATH = f'data/tokens/tokens_{REP}_exp.npy'
PATHS_FILE = f'data/tokens/paths_{REP}_exp.txt'
LABELS_FILE = 'data/images_exp/labels_subset.json'
OUTPUT_DIR = Path(f'results/maskgit_exp/{REP}/counterfactuals_connected')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOP = 8
NLL_PERCENTILE = 85     # umbral para binarizar el heatmap
MIN_REGION_SIZE = 3     # descartar componentes conexas menores a esto (ruido)
N_STEPS = 12
GRID_H, GRID_W = 32, 32
ANOMALY_LABELS = [14, 15, 16]

CLASS_NAMES = {
    0:'SNIa-SALT2',1:'SNIa-91bg',2:'SNIax',3:'SNII-Templates',4:'SNII-NMF',
    5:'SNII+HostXT',6:'SNIb-Templates',7:'SNIb+HostXT',8:'SNIc-Templates',
    9:'SNIc+HostXT',10:'SNIcBL+HostXT',11:'SNIIb+HostXT',12:'SNIIn-MOSFIT',
    13:'SNIIn+HostXT',14:'SLSN-I+host',15:'SLSN-I_no_host',16:'PISN',
}


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
        masked = tokens.clone()
        masked[mask] = model.mask_token
        logits = model(masked)
        lp = F.log_softmax(logits, dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    return total / counts.clamp(min=1.0)


def connected_mask(nll_grid, percentile=85, min_size=3):
    """
    Selecciona regiones conexas de NLL alta.
    Retorna: mascara booleana (32,32) y numero de regiones.
    """
    thr = np.percentile(nll_grid, percentile)
    binary = nll_grid > thr  # (32,32) bool

    # Etiquetar componentes conexas (conectividad 8: incluye diagonales)
    structure = np.ones((3, 3), dtype=int)
    labeled, n_comp = ndimage.label(binary, structure=structure)

    # Descartar componentes pequenas (ruido)
    final_mask = np.zeros_like(binary)
    kept = 0
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        if comp.sum() >= min_size:
            final_mask |= comp
            kept += 1

    return final_mask, kept


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{REP}] MaskGIT: {MASKGIT_CKPT}", flush=True)

    maskgit = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=device, strict=False)
    maskgit.eval(); maskgit.to(device)
    vqvae = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=device)
    vqvae.eval(); vqvae.to(device)
    print("Modelos cargados", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    with open(PATHS_FILE) as f:
        paths = [l.strip() for l in f]
    with open(LABELS_FILE) as f:
        labels_map = json.load(f)
    snids = [os.path.basename(p).replace('.png','') for p in paths]
    labels = np.array([labels_map.get(s,-1) for s in snids])
    sel_idx = np.where(np.isin(labels, ANOMALY_LABELS))[0][:N_TOP]

    for rank, idx in enumerate(tqdm(sel_idx), 1):
        snid = snids[idx]; cls = CLASS_NAMES.get(labels[idx],'?')
        tokens = torch.from_numpy(tokens_all[idx:idx+1]).long().to(device)

        # NLL por token -> heatmap 32x32
        local_nll = nll_per_token(maskgit, tokens, n_rounds=5)
        nll_grid = local_nll[0].cpu().numpy().reshape(GRID_H, GRID_W)

        # Seleccion por conexidad
        cmask_2d, n_regions = connected_mask(nll_grid, NLL_PERCENTILE, MIN_REGION_SIZE)
        amask = torch.from_numpy(cmask_2d.flatten()).unsqueeze(0).to(device)
        n_mod = int(amask.sum().item())

        cf_tokens = maskgit.generate_counterfactual(tokens, amask, n_steps=N_STEPS)

        with torch.no_grad():
            x_recon = vqvae.decode_from_indices(tokens.view(1,GRID_H,GRID_W))
            x_cf = vqvae.decode_from_indices(cf_tokens.view(1,GRID_H,GRID_W))
        recon = x_recon[0].cpu().permute(1,2,0).numpy().clip(0,1)
        cf = x_cf[0].cpu().permute(1,2,0).numpy().clip(0,1)
        orig = np.asarray(Image.open(paths[idx]).convert('RGB').resize((256,256)), dtype=np.float32)/255.0

        # Figura: original | recon | CF | mascara conexa | diff
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(f"[{REP}-conexo] #{rank} {snid} | {cls} | "
                     f"{n_regions} regiones, {n_mod}/1024 tokens", fontsize=12)
        axes[0].imshow(orig); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(recon); axes[1].set_title('Reconstruccion'); axes[1].axis('off')
        axes[2].imshow(cf); axes[2].set_title('Contrafactual'); axes[2].axis('off')
        # mascara conexa upscaled
        mask_up = np.kron(cmask_2d, np.ones((8,8)))
        axes[3].imshow(orig)
        axes[3].imshow(mask_up, cmap='cool', alpha=0.5)
        axes[3].set_title(f'Region enmascarada'); axes[3].axis('off')
        diff = np.abs(recon-cf).mean(axis=2)
        im = axes[4].imshow(diff, cmap='hot'); axes[4].set_title('|Recon-CF|'); axes[4].axis('off')
        plt.colorbar(im, ax=axes[4], fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f'cfc_{REP}_rank{rank:02d}_{snid}_{cls}.png', dpi=120, bbox_inches='tight')
        plt.close()

    print(f"[{REP}] {len(sel_idx)} contrafactuales conexos en {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
