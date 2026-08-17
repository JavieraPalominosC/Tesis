"""
Exploracion de metodos de anomaly score sobre una muestra balanceada.
Prueba varias combinaciones de mask_ratio y agregacion (mean/max/p90)
e imprime la separacion entre grupos para cada una.

Uso: python scripts/explore_score_methods.py
NOTA: esto es SOLO exploratorio. El metodo final debe ser el mismo
para las 3 representaciones (decidir con el profe).
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

from scripts.train_maskgit_cls import MaskGITWithViz

CHECKPOINT  = 'results/maskgit_full/overlay_lines/checkpoints/maskgit-epoch=14-val/loss=2.8207.ckpt'
TOKENS_PATH = 'data/tokens/tokens_overlay_lines_full.npy'
PATHS_FILE  = 'data/tokens/paths_overlay_lines_full.txt'
LABELS_FILE = 'data/labels_overlay_full.json'

N_PER_GROUP = 2000   # muestra balanceada por grupo
SEED = 42

MASK_RATIOS = [0.15, 0.30, 0.50]
AGGS        = ['mean', 'max', 'p90']
N_ROUNDS    = 10

GROUP_MAP = {
    0:'SNIa',1:'SNIa',2:'SNIa', 3:'SNII',4:'SNII',5:'SNII',
    6:'SNIbc',7:'SNIbc',8:'SNIbc',9:'SNIbc',10:'SNIbc',11:'SNII',
    12:'SNIIn',13:'SNIIn', 14:'SLSN',15:'SLSN', 16:'PISN',
}
GROUP_NAMES = ['SNIa','SNIbc','SNII','SNIIn','SLSN','PISN']


def score_batch(model, tokens, mask_ratio, agg, n_rounds=10):
    """Devuelve (B,) scores para el batch, con la agregacion pedida."""
    B, T = tokens.shape
    per_round = torch.zeros(n_rounds, B, device=tokens.device)

    for r in range(n_rounds):
        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True

        masked = tokens.clone()
        masked[mask] = model.mask_token
        logits = model(masked)
        log_probs = F.log_softmax(logits, dim=-1)
        tlp = log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # (B,T)
        nll = -tlp  # (B,T) NLL por token

        # agregacion sobre los tokens enmascarados
        m = mask.float()
        if agg == 'mean':
            val = (nll * m).sum(1) / m.sum(1)
        elif agg == 'max':
            # max del NLL entre los enmascarados
            nll_masked = nll.masked_fill(~mask, float('-inf'))
            val = nll_masked.max(dim=1).values
        elif agg == 'p90':
            # percentil 90 del NLL entre los enmascarados, por objeto
            val = torch.zeros(B, device=tokens.device)
            for i in range(B):
                vals_i = nll[i][mask[i]]
                val[i] = torch.quantile(vals_i, 0.90)
        per_round[r] = val

    return per_round.mean(0)  # promedio sobre rondas -> (B,)


def separation_metric(scores, groups):
    """Rango entre medias de grupo / dispersion media intra-grupo.
    Mayor = mejor separacion."""
    means = np.array([scores[groups == g].mean() for g in GROUP_NAMES])
    stds  = np.array([scores[groups == g].std()  for g in GROUP_NAMES])
    inter = means.max() - means.min()
    intra = stds.mean()
    return inter / intra, means


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    model = MaskGITWithViz.load_from_checkpoint(CHECKPOINT, map_location=device, strict=False)
    model.eval(); model.to(device)
    print("Modelo cargado OK", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    with open(PATHS_FILE) as f:
        paths = [l.strip() for l in f]
    with open(LABELS_FILE) as f:
        labels_map = json.load(f)
    snids = [os.path.basename(p).replace('.png','') for p in paths]
    labels = np.array([labels_map.get(s, -1) for s in snids])
    groups_all = np.array([GROUP_MAP.get(l, 'Unknown') for l in labels])

    # muestra balanceada
    rng = np.random.default_rng(SEED)
    idx_sample = []
    for g in GROUP_NAMES:
        gi = np.where(groups_all == g)[0]
        take = min(N_PER_GROUP, len(gi))
        idx_sample.append(rng.choice(gi, size=take, replace=False))
    idx_sample = np.concatenate(idx_sample)
    rng.shuffle(idx_sample)

    tok = tokens_all[idx_sample]
    grp = groups_all[idx_sample]
    print(f"Muestra: {len(tok):,} objetos ({N_PER_GROUP}/grupo)", flush=True)

    BATCH = 128
    for mask_ratio in MASK_RATIOS:
        for agg in AGGS:
            all_scores = np.zeros(len(tok))
            with torch.no_grad():
                for s in range(0, len(tok), BATCH):
                    e = min(s + BATCH, len(tok))
                    b = torch.from_numpy(tok[s:e]).long().to(device)
                    sc = score_batch(model, b, mask_ratio, agg, N_ROUNDS)
                    all_scores[s:e] = sc.cpu().numpy()

            sep, means = separation_metric(all_scores, grp)
            order = sorted(zip(GROUP_NAMES, means), key=lambda x: -x[1])
            order_str = ' > '.join(f'{n}({m:.3f})' for n, m in order)
            print(f"\n[ratio={mask_ratio} agg={agg}] separacion={sep:.3f}", flush=True)
            print(f"   {order_str}", flush=True)


if __name__ == "__main__":
    main()
