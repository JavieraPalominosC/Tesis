"""
Contrafactual del segundo peak + metrica del reclamo del profe:

que fraccion del cambio |recon-CF| cae sobre el pulso extra vs sobre el
peak original de la SN.

Uso: python scripts/cf_synthetic_second_peak.py
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
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
IMG_DIR      = 'data/images/elasticc_1/synthetic_second_peak'
MANIFEST     = 'data/synthetic_second_peak_manifest.json'
OUT          = Path('results/synthetic_eval/second_peak_cf')
OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
QUANTILE = 0.85
N_STEPS = 12
N_BANDS = 6
TARGET_BAND_IDX = 2
XLIM0, XLIM1 = -0.05, 1.05
HALF_WIN = 0.06
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


def lanes(grid=GRID, n=N_BANDS):
    out = np.zeros(grid, dtype=int)
    rpb = grid / n
    for gi in range(grid):
        out[gi] = (n - 1) - int(gi // rpb)
    return out


def frac_to_x(f, width=H):
    x = (f - XLIM0) / (XLIM1 - XLIM0)
    return int(np.clip(np.floor(x * width), 0, width - 1))


def window_x(frac, half=HALF_WIN, width=H):
    x0 = frac_to_x(frac - half, width)
    x1 = frac_to_x(frac + half, width) + 1
    return x0, max(x1, x0 + 1)


def band_rows(band_idx=TARGET_BAND_IDX, height=H, n=N_BANDS):
    # waterfall: indice mayor (Y) arriba, u abajo. Cada banda ~ height/n filas.
    # Filas de band_idx: invertidas como en lanes().
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
    locs = {rec['snid']: (rec['peak_frac'], rec['second_frac']) for rec in man}

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev)
    vq.eval()
    vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval()
    mg.to(dev)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    r0, r1 = band_rows()

    rows = []
    amp_files = sorted(Path(IMG_DIR).glob('*__amp15.png'))
    print("modelos cargados", flush=True)

    for pa in amp_files:
        snid = pa.stem.split('__')[0]
        pc = Path(IMG_DIR) / f'{snid}__control.png'
        if not pc.exists() or snid not in locs:
            continue
        pf, sf = locs[snid]
        xp0, xp1 = window_x(pf)
        xs0, xs1 = window_x(sf)

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
        rec = x_rec[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        cf = x_cf[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ctrl = np.asarray(Image.open(pc).convert('RGB').resize((256, 256)),
                          dtype=np.float32) / 255.0
        pert = np.asarray(Image.open(pa).convert('RGB').resize((256, 256)),
                          dtype=np.float32) / 255.0

        diff = np.abs(rec - cf).mean(axis=2)
        m_peak = mass_in(diff, r0, r1, xp0, xp1)
        m_2nd = mass_in(diff, r0, r1, xs0, xs1)
        m_band = float(diff[r0:r1, :].sum())
        den = m_peak + m_2nd
        prec_cf = m_2nd / den if den > 1e-12 else 0.0
        frac_peak = m_peak / m_band if m_band > 1e-12 else 0.0
        frac_2nd = m_2nd / m_band if m_band > 1e-12 else 0.0
        rows.append(dict(snid=snid, n_mod=n_mod, prec_cf=prec_cf,
                         frac_peak=frac_peak, frac_2nd=frac_2nd,
                         peak_frac=pf, second_frac=sf))

        vmax = np.percentile(diff, 99.5)
        vmin = np.percentile(diff, 60)
        fig, ax = plt.subplots(1, 5, figsize=(22, 5))
        fig.suptitle(
            f'{snid} | 2nd peak amp15 | tokens mod={n_mod}/1024 | '
            f'prec_cf={prec_cf:.2f} (1=solo extra, 0.5=ambos)',
            fontsize=11,
        )
        for a, img, ttl in zip(ax[:4],
                               [ctrl, pert, rec, cf],
                               ['Control', 'Con segundo peak',
                                'Reconstruccion VQ-VAE', 'Contrafactual']):
            a.imshow(img)
            a.set_title(ttl)
            a.axvline(xp0, color='cyan', lw=0.8, ls='--')
            a.axvline(xp1, color='cyan', lw=0.8, ls='--')
            a.axvline(xs0, color='lime', lw=0.8, ls='--')
            a.axvline(xs1, color='lime', lw=0.8, ls='--')
            a.axis('off')
        im = ax[4].imshow(diff, cmap='inferno', vmin=vmin, vmax=vmax)
        ax[4].set_title('|Recon - CF|  cyan=peak SN  lime=2nd')
        ax[4].axvline(xp0, color='cyan', lw=0.8, ls='--')
        ax[4].axvline(xp1, color='cyan', lw=0.8, ls='--')
        ax[4].axvline(xs0, color='lime', lw=0.8, ls='--')
        ax[4].axvline(xs1, color='lime', lw=0.8, ls='--')
        ax[4].axis('off')
        plt.colorbar(im, ax=ax[4], fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUT / f'cf_second_{snid}.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  {snid}: mod={n_mod}  prec_cf={prec_cf:.3f}  '
              f'frac_peak={frac_peak:.3f}  frac_2nd={frac_2nd:.3f}', flush=True)

    if rows:
        mean_prec = float(np.mean([r['prec_cf'] for r in rows]))
        mean_fp = float(np.mean([r['frac_peak'] for r in rows]))
        mean_f2 = float(np.mean([r['frac_2nd'] for r in rows]))
        print('\n' + '=' * 70)
        print(f'prec_cf medio (masa CF en 2nd / (2nd+peak)): {mean_prec:.3f}')
        print(f'fraccion |recon-CF| en peak original (banda r): {mean_fp:.3f}')
        print(f'fraccion |recon-CF| en segundo peak  (banda r): {mean_f2:.3f}')
        print('Si frac_peak ~ frac_2nd, el CF suaviza ambos. Eso es el reclamo del profe.')
        print('=' * 70)
        with open(OUT / 'summary.json', 'w') as f:
            json.dump({
                'quantile': QUANTILE,
                'mean_prec_cf': mean_prec,
                'mean_frac_peak': mean_fp,
                'mean_frac_2nd': mean_f2,
                'per_object': rows,
            }, f, indent=2)
    print(f'Listo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
