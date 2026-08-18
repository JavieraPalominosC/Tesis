"""
Evalua el segundo peak: el modelo lo detecta, y lo localiza en el tiempo
del pulso extra (no en el peak original de la SN)?

Uso: python scripts/eval_synthetic_second_peak.py
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json, collections
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
IMG_DIR      = 'data/images/elasticc_1/synthetic_second_peak'
MANIFEST     = 'data/synthetic_second_peak_manifest.json'
OUT          = Path('results/synthetic_eval/second_peak')
OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
N_ROUNDS, MASK_RATIO = 10, 0.5
N_BANDS = 6
TARGET_BAND_IDX = 2   # r
XLIM0, XLIM1 = -0.05, 1.05
HALF_WIN = 0.06


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


def lanes(grid=GRID, n=N_BANDS):
    out = np.zeros(grid, dtype=int)
    rpb = grid / n
    for gi in range(grid):
        out[gi] = (n - 1) - int(gi // rpb)
    return out


def frac_to_col(f, grid=GRID):
    x = (f - XLIM0) / (XLIM1 - XLIM0)
    return int(np.clip(np.floor(x * grid), 0, grid - 1))


def window_cols(frac, half=HALF_WIN, grid=GRID):
    c0 = frac_to_col(frac - half, grid)
    c1 = frac_to_col(frac + half, grid) + 1
    return c0, max(c1, c0 + 1)


def col_mean(nll, c0, c1, row_mask=None):
    sl = nll[:, c0:c1]
    if row_mask is not None:
        sl = sl[row_mask, :]
    return float(sl.mean()) if sl.size else 0.0


def main():
    with open(MANIFEST) as f:
        man = json.load(f)
    locs = {}
    for rec in man:
        locs[rec['snid']] = (rec['peak_frac'], rec['second_frac'])

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev)
    vq.eval()
    vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval()
    mg.to(dev)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    L = lanes()
    row_r = L == TARGET_BAND_IDX

    res = collections.defaultdict(dict)
    for p in sorted(Path(IMG_DIR).glob('*.png')):
        snid, tag = p.stem.split('__')
        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x)
            _, _, i, *_ = vq.codebook(z)
        tok = i.view(1, -1).long().to(dev)
        nll = nll_map(mg, tok, N_ROUNDS, MASK_RATIO)[0].cpu().numpy().reshape(GRID, GRID)
        pf, sf = locs[snid]
        p0, p1 = window_cols(pf)
        s0, s1 = window_cols(sf)
        res[snid][tag] = dict(
            score=float(nll.mean()),
            band_r=float(nll[row_r, :].mean()),
            nll_peak=col_mean(nll, p0, p1, row_r),
            nll_second=col_mean(nll, s0, s1, row_r),
            peak_frac=pf,
            second_frac=sf,
            tokens=tok.cpu().numpy().ravel(),
        )

    tags = ['amp2', 'amp5', 'amp15']
    print('\n' + '=' * 88)
    print('SEGUNDO PEAK: diferencia respecto al control')
    print('nll_peak = NLL en la ventana del maximo real de la SN')
    print('nll_2nd  = NLL en la ventana del pulso insertado (decline tardio)')
    print('prec_t   = Δnll_2nd / (|Δnll_2nd| + |Δnll_peak|); 1 = solo el extra')
    print('=' * 88)
    hdr = f"{'':<10}{'Δscore':>10}{'Δ NLL r':>10}{'Δ peak':>10}{'Δ 2nd':>10}{'prec_t':>10}{'tok dist':>10}"
    print(hdr)
    summary = {}
    for t in tags:
        ds, dr, dp, d2, pr, dt = [], [], [], [], [], []
        for snid, d in res.items():
            if 'control' not in d or t not in d:
                continue
            c, a = d['control'], d[t]
            ds.append(a['score'] - c['score'])
            dr.append(a['band_r'] - c['band_r'])
            dp.append(a['nll_peak'] - c['nll_peak'])
            d2.append(a['nll_second'] - c['nll_second'])
            den = abs(d2[-1]) + abs(dp[-1])
            pr.append(abs(d2[-1]) / den if den > 1e-8 else 0.0)
            dt.append(int((c['tokens'] != a['tokens']).sum()))
        row = dict(dscore=float(np.mean(ds)), dnll_r=float(np.mean(dr)),
                   dpeak=float(np.mean(dp)), dsecond=float(np.mean(d2)),
                   prec_t=float(np.mean(pr)), tok=float(np.mean(dt)))
        summary[t] = row
        print(f"{t:<10}{row['dscore']:>10.4f}{row['dnll_r']:>10.4f}"
              f"{row['dpeak']:>10.4f}{row['dsecond']:>10.4f}"
              f"{row['prec_t']:>10.3f}{row['tok']:>10.0f}")

    print('\nSi prec_t ~ 0.5, el modelo trata igual el peak real y el extra.')
    print('Si prec_t se acerca a 1, localiza el segundo peak.')
    print(f'Guardado en {OUT}/')
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({
            'aggregates': summary,
            'per_object': {
                s: {k: {kk: vv for kk, vv in v.items() if kk != 'tokens'}
                    for k, v in d.items()}
                for s, d in res.items()
            },
        }, f, indent=2)


if __name__ == '__main__':
    main()
