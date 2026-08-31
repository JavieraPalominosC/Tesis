"""
Contrafactual del experimento MAD (punto EXISTENTE, f += k*MAD).

Mismo layout de 5 paneles que el segundo peak:
  Control | MAD k=10 | recon VQ-VAE | CF | |recon-CF|

La linea amarilla marca el MJD del punto que se perturbo (no es un
punto inventado: es una observacion real de esa banda).

Uso: python scripts/cf_synthetic_mad.py [BAND] [K]
     default: r  10
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
IMG_DIR      = 'data/images/elasticc_1/synthetic_mad'
MANIFEST     = 'data/synthetic_mad_manifest.json'
RAW_PARQUET  = 'data/lightcurves/elasticc_1/raw/lc_SNIa-SALT2.parquet'
OUT          = Path('results/synthetic_eval/mad_cf')
OUT.mkdir(parents=True, exist_ok=True)

BAND = sys.argv[1] if len(sys.argv) > 1 else 'r'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_SHOW = 8
PRIORITY = ['12851454']

BAND_IDX = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
GRID = 32
QUANTILE = 0.85
N_STEPS = 12
N_BANDS = 6
XLIM0, XLIM1 = -0.05, 1.05
HALF_WIN = 0.04
H = 256


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
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    return total / counts.clamp(min=1.0)


def frac_to_x(f, width=H):
    x = (f - XLIM0) / (XLIM1 - XLIM0)
    return int(np.clip(np.floor(x * width), 0, width - 1))


def window_x(frac, half=HALF_WIN, width=H):
    x0 = frac_to_x(frac - half, width)
    x1 = frac_to_x(frac + half, width) + 1
    return x0, max(x1, x0 + 1)


def band_rows(band_idx, height=H, n=N_BANDS):
    visual_slot = (n - 1) - band_idx
    rpb = height / n
    r0 = int(visual_slot * rpb)
    r1 = int((visual_slot + 1) * rpb)
    return r0, r1


def mass_in(diff, r0, r1, x0, x1):
    return float(diff[r0:r1, x0:x1].sum())


def main():
    with open(MANIFEST) as f:
        man = json.load(f)
    recs = [r for r in man if r.get('band') == BAND and r.get('k') == K]
    recs = {r['snid']: r for r in recs}
    snids = [s for s in PRIORITY if s in recs]
    for s in sorted(recs):
        if s not in snids:
            snids.append(s)
        if len(snids) >= N_SHOW:
            break

    print(f'CF MAD banda={BAND} k={K}  n={len(snids)}', flush=True)
    df = pd.read_parquet(RAW_PARQUET, columns=['SNID', 'MJD'])
    df['SNID'] = df['SNID'].astype(str)
    spans = {}
    for snid in snids:
        sub = df[df['SNID'] == str(snid)]
        if sub.empty:
            print(f'  skip {snid}: no esta en parquet', flush=True)
            continue
        t0, t1 = float(sub['MJD'].min()), float(sub['MJD'].max())
        if not np.isfinite(t0) or not np.isfinite(t1):
            print(f'  skip {snid}: MJD invalido', flush=True)
            continue
        spans[snid] = (t0, max(t1 - t0, 1.0))
    del df
    snids = [s for s in snids if s in spans]

    bidx = BAND_IDX[BAND]
    r0, r1 = band_rows(bidx)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev)
    vq.eval()
    vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval()
    mg.to(dev)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    print('modelos cargados', flush=True)

    rows = []
    for snid in snids:
        rec = recs[snid]
        pc = Path(IMG_DIR) / f'{snid}__control.png'
        pa = Path(IMG_DIR) / f'{snid}__{BAND}_k{K}.png'
        if not pc.exists() or not pa.exists():
            print(f'  skip {snid}: falta imagen', flush=True)
            continue
        t0, span = spans[snid]
        frac = (float(rec['mjd']) - t0) / span
        if not np.isfinite(frac):
            print(f'  skip {snid}: frac NaN', flush=True)
            continue
        xp0, xp1 = window_x(frac)

        x = tf(Image.open(pa).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x)
            _, _, idxs, *_ = vq.codebook(z)
        tok = idxs.view(1, -1).long().to(dev)

        nll = nll_per_token(mg, tok, n_rounds=5)
        thr = torch.quantile(nll, QUANTILE)
        amask = nll > thr
        n_mod = int(amask.sum().item())
        cf_tok = mg.generate_counterfactual(tok, amask, n_steps=N_STEPS)

        with torch.no_grad():
            x_rec = vq.decode_from_indices(tok.view(1, GRID, GRID))
            x_cf = vq.decode_from_indices(cf_tok.view(1, GRID, GRID))
        rec_img = x_rec[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        cf = x_cf[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ctrl = np.asarray(Image.open(pc).convert('RGB').resize((256, 256)),
                          dtype=np.float32) / 255.0
        pert = np.asarray(Image.open(pa).convert('RGB').resize((256, 256)),
                          dtype=np.float32) / 255.0

        diff = np.abs(rec_img - cf).mean(axis=2)
        m_pt = mass_in(diff, r0, r1, xp0, xp1)
        m_band = float(diff[r0:r1, :].sum())
        frac_pt = m_pt / m_band if m_band > 1e-12 else 0.0
        rows.append(dict(snid=snid, n_mod=n_mod, frac_pt=frac_pt,
                         mjd=float(rec['mjd']), mad=float(rec['mad']),
                         flux0=float(rec['flux0']), flux1=float(rec['flux1'])))

        vmax = np.percentile(diff, 99.5)
        vmin = np.percentile(diff, 60)
        fig, ax = plt.subplots(1, 5, figsize=(22, 5))
        fig.suptitle(
            f'[waterfall] {snid} | MAD {BAND} k={K}  '
            f'(punto existente, MJD={rec["mjd"]:.2f}, '
            f'f={rec["flux0"]:.1f}->{rec["flux1"]:.1f}) | '
            f'tokens mod={n_mod}/1024 | frac_pt={frac_pt:.2f}',
            fontsize=11,
        )
        for a, img, ttl in zip(
            ax[:4],
            [ctrl, pert, rec_img, cf],
            ['Control', f'MAD {BAND} k={K} (punto existente)',
             'Reconstruccion VQ-VAE', 'Contrafactual'],
        ):
            a.imshow(img)
            a.set_title(ttl)
            a.axvline(xp0, color='yellow', lw=0.9, ls='--')
            a.axvline(xp1, color='yellow', lw=0.9, ls='--')
            a.axis('off')
        im = ax[4].imshow(diff, cmap='inferno', vmin=vmin, vmax=vmax)
        ax[4].set_title('|Recon - CF|  amarillo=punto MAD')
        ax[4].axvline(xp0, color='yellow', lw=0.9, ls='--')
        ax[4].axvline(xp1, color='yellow', lw=0.9, ls='--')
        ax[4].axis('off')
        plt.colorbar(im, ax=ax[4], fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUT / f'cf_mad_{snid}_{BAND}_k{K}.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  {snid}: mod={n_mod}  frac_pt={frac_pt:.3f}  '
              f'mjd_frac={frac:.3f}', flush=True)

    if rows:
        mean_fp = float(np.mean([r['frac_pt'] for r in rows]))
        print('\n' + '=' * 70)
        print(f'fraccion media |recon-CF| en el punto MAD ({BAND} k={K}): {mean_fp:.3f}')
        print('Si frac_pt es bajo, el CF no localiza el bump: el modelo casi no lo ve.')
        print('=' * 70)
        with open(OUT / f'summary_{BAND}_k{K}.json', 'w') as f:
            json.dump({
                'band': BAND, 'k': K, 'quantile': QUANTILE,
                'mean_frac_pt': mean_fp, 'per_object': rows,
            }, f, indent=2)
    print(f'Listo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
