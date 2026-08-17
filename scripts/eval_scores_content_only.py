"""
Anomaly score calculado SOLO sobre tokens con contenido (ignora el fondo blanco).
Compara contra el score estandar para ver si filtrar el fondo mejora la separacion.

Uso: python scripts/eval_scores_content_only.py <REP>
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import json, os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from scripts.train_maskgit_cls import MaskGITWithViz

REP = sys.argv[1]
CHECKPOINTS = {
    'overlay_lines': 'results/maskgit_full/overlay_lines/checkpoints/maskgit-epoch=14-val/loss=2.8207.ckpt',
    'points':        'results/maskgit_full/points/checkpoints/maskgit-epoch=13-val/loss=2.9244.ckpt',
    'waterfall':     'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt',
}
LABELS_PATHS = {
    'overlay_lines': 'data/labels_overlay_full.json',
    'points':        'data/labels_points_full.json',
    'waterfall':     'data/labels_waterfall_full.json',
}

CHECKPOINT  = CHECKPOINTS[REP]
TOKENS_PATH = f'data/tokens/tokens_{REP}_full.npy'
PATHS_FILE  = f'data/tokens/paths_{REP}_full.txt'
LABELS_FILE = LABELS_PATHS[REP]

GRID, PATCH = 32, 8
N_PER_GROUP = 2000     # muestra balanceada (rapido)
N_ROUNDS, MASK_RATIO = 10, 0.5
SEED = 42

GROUP_MAP = {0:'SNIa',1:'SNIa',2:'SNIa',3:'SNII',4:'SNII',5:'SNII',
             6:'SNIbc',7:'SNIbc',8:'SNIbc',9:'SNIbc',10:'SNIbc',11:'SNII',
             12:'SNIIn',13:'SNIIn',14:'SLSN',15:'SLSN',16:'PISN'}
GROUP_NAMES = ['SNIa','SNIbc','SNII','SNIIn','SLSN','PISN']


def content_mask(path):
    img = np.asarray(Image.open(path).convert('RGB').resize((256,256)), dtype=np.float32)/255.0
    cm = np.zeros(GRID*GRID, dtype=bool)
    k = 0
    for gi in range(GRID):
        for gj in range(GRID):
            p = img[gi*PATCH:(gi+1)*PATCH, gj*PATCH:(gj+1)*PATCH, :]
            cm[k] = (p.reshape(-1,3).mean(axis=1) < 0.9).sum() > 0
            k += 1
    return cm


@torch.no_grad()
def scores_both(model, tokens, cmask, n_rounds=10, mask_ratio=0.5):
    """Devuelve (score_estandar, score_solo_contenido) para el batch."""
    B, T = tokens.shape
    acc_std = torch.zeros(B, device=tokens.device)
    acc_cnt = torch.zeros(B, device=tokens.device)
    for _ in range(n_rounds):
        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        masked = tokens.clone(); masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        nll = -lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)   # (B,T)

        m = mask.float()
        acc_std += (nll*m).sum(1) / m.sum(1).clamp(min=1)

        mc = (mask & cmask).float()                              # enmascarados Y con contenido
        acc_cnt += (nll*mc).sum(1) / mc.sum(1).clamp(min=1)
    return acc_std/n_rounds, acc_cnt/n_rounds


def separation(scores, groups):
    means = np.array([scores[groups==g].mean() for g in GROUP_NAMES])
    stds  = np.array([scores[groups==g].std()  for g in GROUP_NAMES])
    return (means.max()-means.min())/stds.mean(), means


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskGITWithViz.load_from_checkpoint(CHECKPOINT, map_location=device, strict=False)
    model.eval(); model.to(device)
    print(f"[{REP}] modelo cargado", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    with open(PATHS_FILE) as f: paths = [l.strip() for l in f]
    with open(LABELS_FILE) as f: labels_map = json.load(f)
    snids = [os.path.basename(p).replace('.png','') for p in paths]
    labels = np.array([labels_map.get(s,-1) for s in snids])
    groups_all = np.array([GROUP_MAP.get(l,'?') for l in labels])

    rng = np.random.default_rng(SEED)
    idx = []
    for g in GROUP_NAMES:
        gi = np.where(groups_all==g)[0]
        idx.append(rng.choice(gi, size=min(N_PER_GROUP,len(gi)), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    print(f"[{REP}] muestra: {len(idx):,}", flush=True)

    s_std = np.zeros(len(idx)); s_cnt = np.zeros(len(idx))
    B = 64
    for s in tqdm(range(0, len(idx), B)):
        e = min(s+B, len(idx))
        sub = idx[s:e]
        tok = torch.from_numpy(tokens_all[sub]).long().to(device)
        cm = np.stack([content_mask(paths[i]) for i in sub])
        cmask = torch.from_numpy(cm).to(device)
        a, b = scores_both(model, tok, cmask, N_ROUNDS, MASK_RATIO)
        s_std[s:e] = a.cpu().numpy(); s_cnt[s:e] = b.cpu().numpy()

    grp = groups_all[idx]
    for name, sc in [('ESTANDAR (todos los tokens)', s_std),
                     ('SOLO CONTENIDO (sin fondo)', s_cnt)]:
        sep, means = separation(sc, grp)
        order = ' > '.join(f'{n}({m:.3f})' for n,m in sorted(zip(GROUP_NAMES,means), key=lambda x:-x[1]))
        print(f"\n[{REP}] {name}: separacion = {sep:.3f}", flush=True)
        print(f"   {order}", flush=True)


if __name__ == "__main__":
    main()
