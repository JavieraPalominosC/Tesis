"""
Evalua las imagenes con pulso insertado: detecta el modelo el pulso?
Compara score y localizacion de cada amplitud contra su control.

Uso: python scripts/eval_synthetic_insert.py
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
IMG_DIR      = 'data/images/elasticc_1/synthetic_insert'
OUT          = Path('results/synthetic_eval/insert'); OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
N_ROUNDS, MASK_RATIO = 10, 0.5
N_BANDS = 6
TARGET_BAND_IDX = 2   # r


@torch.no_grad()
def nll_map(model, tokens, n_rounds=10, mask_ratio=0.5):
    B, T = tokens.shape
    total = torch.zeros(B, T, device=tokens.device)
    counts = torch.zeros(B, T, device=tokens.device)
    for _ in range(n_rounds):
        n_mask = int(T*mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        masked = tokens.clone(); masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp)*mask.float(); counts += mask.float()
    return total/counts.clamp(min=1.0)


def lanes(grid=GRID, n=N_BANDS):
    out = np.zeros(grid, dtype=int); rpb = grid/n
    for gi in range(grid):
        out[gi] = (n-1) - int(gi//rpb)
    return out


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev); vq.eval(); vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval(); mg.to(dev)
    tf = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    L = lanes()

    res = collections.defaultdict(dict)
    for p in sorted(Path(IMG_DIR).glob('*.png')):
        snid, tag = p.stem.split('__')
        x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x); _,_,i,*_ = vq.codebook(z)
        tok = i.view(1,-1).long().to(dev)
        nll = nll_map(mg, tok, N_ROUNDS, MASK_RATIO)[0].cpu().numpy().reshape(GRID,GRID)
        c0, c1 = int(GRID*0.40), int(GRID*0.60)
        res[snid][tag] = dict(
            score = float(nll.mean()),
            band_r = float(nll[L==TARGET_BAND_IDX,:].mean()),
            centro = float(nll[:, c0:c1].mean()),
            fuera  = float(np.concatenate([nll[:,:c0].ravel(), nll[:,c1:].ravel()]).mean()),
            tokens = tok.cpu().numpy().ravel(),
        )

    tags = ['amp2','amp5','amp15']
    print("\n" + "="*70)
    print("PULSO INSERTADO: diferencia respecto al control")
    print("="*70)
    print(f"{'':<10}{'Δscore':>12}{'Δ NLL banda r':>16}{'Δ centro/fuera':>18}{'tokens dist.':>14}")
    for t in tags:
        ds, dr, dc, dt = [], [], [], []
        for snid, d in res.items():
            if 'control' not in d or t not in d: continue
            c, a = d['control'], d[t]
            ds.append(a['score'] - c['score'])
            dr.append(a['band_r'] - c['band_r'])
            dc.append((a['centro']/a['fuera']) - (c['centro']/c['fuera']))
            dt.append(int((c['tokens'] != a['tokens']).sum()))
        print(f"{t:<10}{np.mean(ds):>12.4f}{np.mean(dr):>16.4f}{np.mean(dc):>18.4f}{np.mean(dt):>14.0f}")

    print("\nReferencia: score ~3.0, dispersion intra-grupo ~0.3")
    print(f"Guardado en {OUT}/")
    with open(OUT/'summary.json','w') as f:
        json.dump({s:{k:{kk:vv for kk,vv in v.items() if kk!='tokens'}
                      for k,v in d.items()} for s,d in res.items()}, f, indent=2)


if __name__ == '__main__':
    main()
