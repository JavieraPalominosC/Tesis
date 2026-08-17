"""
Evalua las anomalias sinteticas con el pipeline entrenado (VQ-VAE + MaskGIT).
Responde: el modelo detecta la perturbacion? la localiza donde se inyecto?

Uso: python scripts/eval_synthetic_anomalies.py <MODE>
     MODE in {local, global}
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

MODE = sys.argv[1] if len(sys.argv) > 1 else 'local'
assert MODE in ('local', 'global')

if MODE == 'local':
    VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
    MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
    IMG_DIR      = 'data/images/elasticc_1/synthetic_local'
else:
    VQVAE_CKPT   = 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt'
    MASKGIT_CKPT = 'results/maskgit_full/waterfall_global/checkpoints/BEST.ckpt'
    IMG_DIR      = 'data/images/elasticc_1/synthetic_global'

OUTPUT_DIR = Path(f'results/synthetic_eval/{MODE}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 32
N_ROUNDS, MASK_RATIO = 10, 0.5
N_BANDS = 6
BAND_NAMES = ['u','g','r','i','z','Y']
TARGET_BAND_IDX = 2   # 'r'


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
        masked = tokens.clone(); masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp)*mask.float(); counts += mask.float()
    return total / counts.clamp(min=1.0)


def band_lane_map(grid=GRID, n_bands=N_BANDS):
    """fila del grid -> indice de banda (arriba = banda de indice mayor)."""
    bm = np.zeros(grid, dtype=int)
    rpb = grid / n_bands
    for gi in range(grid):
        lane = int(gi // rpb)
        bm[gi] = (n_bands-1) - lane
    return bm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=device); vq.eval(); vq.to(device)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=device, strict=False)
    mg.eval(); mg.to(device)
    print(f"[{MODE}] modelos cargados", flush=True)

    tf = T.Compose([T.Resize((256,256)), T.ToTensor(),
                    T.Normalize([0.5]*3, [0.5]*3)])

    files = sorted(Path(IMG_DIR).glob('*.png'))
    print(f"[{MODE}] {len(files)} imagenes sinteticas", flush=True)

    lanes = band_lane_map()
    rows = []

    for p in files:
        stem = p.stem                    # SNID__kind__amp  o  SNID__control
        parts = stem.split('__')
        snid = parts[0]
        kind = parts[1] if len(parts) > 1 else 'control'
        amp  = parts[2] if len(parts) > 2 else 'none'

        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            z_e = vq.encoder(x)
            _, _, idxs, *_ = vq.codebook(z_e)
        tok = idxs.view(1, -1).long().to(device)

        nll = nll_map(mg, tok, N_ROUNDS, MASK_RATIO)[0].cpu().numpy().reshape(GRID, GRID)
        score = float(nll.mean())

        # NLL medio por banda (carriles) y por mitad temporal
        band_means = np.array([nll[lanes==b, :].mean() for b in range(N_BANDS)])
        frac_target = band_means[TARGET_BAND_IDX] / band_means.sum()
        # zona inyectada: mitad temporal central (donde tc = 0.5*span)
        c0, c1 = int(GRID*0.35), int(GRID*0.65)
        center_mean = nll[:, c0:c1].mean()
        outside_mean = np.concatenate([nll[:, :c0].ravel(), nll[:, c1:].ravel()]).mean()

        rows.append(dict(snid=snid, kind=kind, amp=amp, score=score,
                         band_r_frac=float(frac_target),
                         center=float(center_mean), outside=float(outside_mean),
                         center_ratio=float(center_mean/outside_mean)))

    # ── Resumen: score de cada anomalia vs su control ──
    import collections
    by_snid = collections.defaultdict(dict)
    for r in rows:
        key = 'control' if r['kind']=='control' else f"{r['kind']}__{r['amp']}"
        by_snid[r['snid']][key] = r

    kinds = ['pico','bache','shift','ruido','meseta','escalon']
    amps  = ['suave','medio','fuerte']

    print("\n" + "="*78, flush=True)
    print(f"DETECCION: score(perturbada) - score(control)   [{MODE}]", flush=True)
    print("="*78, flush=True)
    header = f"{'anomalia':<12}" + ''.join(f"{a:>12}" for a in amps)
    print(header, flush=True)
    for k in kinds:
        line = f"{k:<12}"
        for a in amps:
            deltas = []
            for snid, d in by_snid.items():
                if 'control' in d and f'{k}__{a}' in d:
                    deltas.append(d[f'{k}__{a}']['score'] - d['control']['score'])
            line += f"{np.mean(deltas):>12.4f}" if deltas else f"{'-':>12}"
        print(line, flush=True)

    print("\n" + "="*78, flush=True)
    print(f"LOCALIZACION: NLL centro/fuera (inyeccion en el centro temporal)", flush=True)
    print("="*78, flush=True)
    print(header, flush=True)
    for k in kinds:
        line = f"{k:<12}"
        for a in amps:
            vals = [d[f'{k}__{a}']['center_ratio'] for snid, d in by_snid.items() if f'{k}__{a}' in d]
            line += f"{np.mean(vals):>12.4f}" if vals else f"{'-':>12}"
        print(line, flush=True)
    ctrl = [d['control']['center_ratio'] for d in by_snid.values() if 'control' in d]
    print(f"{'control':<12}{np.mean(ctrl):>12.4f}  <- referencia (sin anomalia)", flush=True)

    print("\n" + "="*78, flush=True)
    print(f"ATRIBUCION: fraccion del NLL en la banda r (inyectada). Esperado sin senal: {1/N_BANDS:.3f}", flush=True)
    print("="*78, flush=True)
    print(header, flush=True)
    for k in kinds:
        line = f"{k:<12}"
        for a in amps:
            vals = [d[f'{k}__{a}']['band_r_frac'] for snid, d in by_snid.items() if f'{k}__{a}' in d]
            line += f"{np.mean(vals):>12.4f}" if vals else f"{'-':>12}"
        print(line, flush=True)
    ctrl = [d['control']['band_r_frac'] for d in by_snid.values() if 'control' in d]
    print(f"{'control':<12}{np.mean(ctrl):>12.4f}  <- referencia", flush=True)

    with open(OUTPUT_DIR/'results.json','w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nGuardado en {OUTPUT_DIR}/results.json", flush=True)


if __name__ == '__main__':
    main()
