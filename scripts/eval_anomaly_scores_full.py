"""
Evaluacion de anomaly scores con el prior MaskGIT full, parametrizado por representacion.
Uso: python scripts/eval_anomaly_scores_full.py <REP>
     REP in {overlay_lines, points, waterfall}
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
from tqdm import tqdm

# Misma clase con la que se entreno el MaskGIT full
from scripts.train_maskgit_cls import MaskGITWithViz

REP = sys.argv[1]
assert REP in ('overlay_lines', 'points', 'waterfall')

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
OUTPUT_DIR  = Path(f'results/maskgit_full/{REP}/eval')
BATCH_SIZE  = 256
N_ROUNDS    = 10
MASK_RATIO  = 0.5

GROUP_MAP = {
    0:  'SNIa',   1:  'SNIa',   2:  'SNIa',
    3:  'SNII',   4:  'SNII',   5:  'SNII',
    6:  'SNIbc',  7:  'SNIbc',  8:  'SNIbc',
    9:  'SNIbc',  10: 'SNIbc',  11: 'SNII',
    12: 'SNIIn',  13: 'SNIIn',
    14: 'SLSN',   15: 'SLSN',
    16: 'PISN',
}
GROUP_NAMES = ['SNIa', 'SNIbc', 'SNII', 'SNIIn', 'SLSN', 'PISN']


def fast_anomaly_score(model, tokens, n_rounds=10, mask_ratio=0.5):
    B, T = tokens.shape
    total_scores = torch.zeros(B, device=tokens.device)

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

        token_log_probs = log_probs.gather(
            2, tokens.unsqueeze(-1)
        ).squeeze(-1)

        nll = -(token_log_probs * mask.float()).sum(dim=1) / mask.float().sum(dim=1)
        total_scores += nll

    return total_scores / n_rounds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{REP}] Device: {device}", flush=True)
    print(f"[{REP}] Checkpoint: {CHECKPOINT}", flush=True)
    print(f"[{REP}] N rounds: {N_ROUNDS}, mask ratio: {MASK_RATIO}", flush=True)

    model = MaskGITWithViz.load_from_checkpoint(CHECKPOINT, map_location=device, strict=False)
    model.eval()
    model.to(device)
    print(f"[{REP}] Modelo cargado OK", flush=True)

    tokens_all = np.load(TOKENS_PATH).astype(np.int64)
    print(f"[{REP}] Tokens: {tokens_all.shape}", flush=True)

    with open(PATHS_FILE) as f:
        paths = [line.strip() for line in f]

    with open(LABELS_FILE) as f:
        labels_map = json.load(f)

    snids = [os.path.basename(p).replace('.png', '') for p in paths]
    labels = np.array([labels_map.get(s, -1) for s in snids])
    groups = np.array([GROUP_MAP.get(l, 'Unknown') for l in labels])

    print(f"[{REP}] Distribucion de grupos:", flush=True)
    for g in GROUP_NAMES:
        n = (groups == g).sum()
        print(f"  {g}: {n:,}", flush=True)

    print(f"\n[{REP}] Calculando anomaly scores ({N_ROUNDS} rondas)...", flush=True)

    all_scores = np.zeros(len(tokens_all))
    n_batches = (len(tokens_all) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluando"):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(tokens_all))
            batch = torch.from_numpy(tokens_all[start:end]).long().to(device)
            scores = fast_anomaly_score(model, batch, N_ROUNDS, MASK_RATIO)
            all_scores[start:end] = scores.cpu().numpy()

    valid_mask = np.isfinite(all_scores)
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        print(f"WARNING: {n_nan} scores NaN encontrados, se excluyen", flush=True)

    print("\n" + "="*60, flush=True)
    print(f"ANOMALY SCORES POR GRUPO [{REP}]", flush=True)
    print("="*60, flush=True)

    results = {}
    for g in GROUP_NAMES:
        mask = (groups == g) & valid_mask
        g_scores = all_scores[mask]
        results[g] = {
            'mean': float(np.mean(g_scores)),
            'std': float(np.std(g_scores)),
            'median': float(np.median(g_scores)),
            'min': float(np.min(g_scores)),
            'max': float(np.max(g_scores)),
            'q25': float(np.percentile(g_scores, 25)),
            'q75': float(np.percentile(g_scores, 75)),
            'n': int(mask.sum()),
        }
        print(f"  {g:6s}: mean={results[g]['mean']:.4f} +/- {results[g]['std']:.4f}  "
              f"median={results[g]['median']:.4f}  n={results[g]['n']:,}", flush=True)

    np.save(OUTPUT_DIR / 'anomaly_scores.npy', all_scores)
    with open(OUTPUT_DIR / 'scores_by_group.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[{REP}] Guardado en {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
