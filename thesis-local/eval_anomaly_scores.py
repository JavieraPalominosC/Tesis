"""
Evaluación final de anomaly scores con el prior MaskGIT entrenado.

Calcula -log p(s) para cada supernova del dataset usando múltiples
rondas de masked prediction (más robusto que la aprox. de 1 ronda
usada durante training).

Genera:
  - results/maskgit_cls/eval/anomaly_scores.json  (scores por grupo)
  - results/maskgit_cls/eval/anomaly_scores_all.npy (scores individuales)
  - results/maskgit_cls/eval/histogram_scores.png
  - results/maskgit_cls/eval/boxplot_scores.png
  - results/maskgit_cls/eval/tabla_scores.txt

Uso:
  python scripts/eval_anomaly_scores.py
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

from src.models.maskgit.maskgit import MaskGIT

# ── Config ────────────────────────────────────────────────────────
CHECKPOINT = 'results/maskgit_cls/checkpoints/maskgit-epoch=41-val/loss=3.0498.ckpt'
TOKENS_PATH = 'data/tokens/tokens_cls.npy'
PATHS_FILE = 'data/tokens/paths_cls.txt'
LABELS_FILE = 'data/folds/labels.json'
OUTPUT_DIR = Path('results/maskgit_cls/eval')
BATCH_SIZE = 256
N_ROUNDS = 10       # rondas de masking para promediar (más robusto que 1)
MASK_RATIO = 0.5    # fracción de tokens a enmascarar por ronda

# ── Agrupamiento de 17 clases → 6 grupos ─────────────────────────
GROUP_MAP = {
    0:  'SNIa',   # SNIa-SALT2
    1:  'SNIa',   # SNIa-91bg
    2:  'SNIa',   # SNIax
    3:  'SNII',   # SNII-Templates
    4:  'SNII',   # SNII-NMF
    5:  'SNII',   # SNII+HostXT_V19
    6:  'SNIbc',  # SNIb-Templates
    7:  'SNIbc',  # SNIb+HostXT_V19
    8:  'SNIbc',  # SNIc-Templates
    9:  'SNIbc',  # SNIc+HostXT_V19
    10: 'SNIbc',  # SNIcBL+HostXT_V19
    11: 'SNII',   # SNIIb+HostXT_V19
    12: 'SNIIn',  # SNIIn-MOSFIT
    13: 'SNIIn',  # SNIIn+HostXT_V19
    14: 'SLSN',   # SLSN-I+host
    15: 'SLSN',   # SLSN-I_no_host
    16: 'PISN',   # PISN
}

GROUP_NAMES = ['SNIa', 'SNIbc', 'SNII', 'SNIIn', 'SLSN', 'PISN']


def fast_anomaly_score(model, tokens, n_rounds=10, mask_ratio=0.5):
    """
    Anomaly score robusto: promedio de N rondas con masking aleatorio.
    
    Cada ronda enmascara mask_ratio de los tokens, predice, y acumula
    la NLL sobre los tokens enmascarados. Al promediar N rondas se
    obtiene un estimador menos ruidoso que una sola ronda.
    
    Args:
        model: MaskGIT entrenado
        tokens: (B, seq_len) tensor de tokens
        n_rounds: número de rondas de masking
        mask_ratio: fracción de tokens a enmascarar
    
    Returns:
        scores: (B,) anomaly scores promediados
    """
    B, T = tokens.shape
    total_scores = torch.zeros(B, device=tokens.device)
    
    for _ in range(n_rounds):
        # Generar máscara aleatoria
        n_mask = int(T * mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        
        # Enmascarar tokens
        masked = tokens.clone()
        masked[mask] = model.mask_token
        
        # Forward pass
        logits = model(masked)  # (B, T, num_tokens)
        
        # NLL solo sobre tokens enmascarados
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, num_tokens)
        
        # Recoger log-prob del token correcto en cada posición enmascarada
        token_log_probs = log_probs.gather(
            2, tokens.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)
        
        # Sumar NLL solo en posiciones enmascaradas, normalizar
        nll = -(token_log_probs * mask.float()).sum(dim=1) / mask.float().sum(dim=1)
        total_scores += nll
    
    return total_scores / n_rounds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {CHECKPOINT}", flush=True)
    print(f"N rounds: {N_ROUNDS}, mask ratio: {MASK_RATIO}", flush=True)
    
    # ── Cargar modelo ─────────────────────────────────────────────
    model = MaskGIT.load_from_checkpoint(CHECKPOINT, map_location=device)
    model.eval()
    model.to(device)
    print("Modelo cargado OK", flush=True)
    
    # ── Cargar tokens y labels ────────────────────────────────────
    tokens_all = np.load(TOKENS_PATH).astype(np.int64)  # (N, 1024)
    print(f"Tokens: {tokens_all.shape}", flush=True)
    
    with open(PATHS_FILE) as f:
        paths = [line.strip() for line in f]
    
    with open(LABELS_FILE) as f:
        labels_map = json.load(f)
    
    # Extraer SNID de cada path y buscar su label
    snids = [os.path.basename(p).replace('.png', '') for p in paths]
    labels = np.array([labels_map.get(s, -1) for s in snids])
    
    # Mapear a grupos
    groups = np.array([GROUP_MAP.get(l, 'Unknown') for l in labels])
    
    print(f"Distribución de grupos:", flush=True)
    for g in GROUP_NAMES:
        n = (groups == g).sum()
        print(f"  {g}: {n:,}", flush=True)
    
    # ── Calcular anomaly scores ───────────────────────────────────
    print(f"\nCalculando anomaly scores ({N_ROUNDS} rondas)...", flush=True)
    
    all_scores = np.zeros(len(tokens_all))
    n_batches = (len(tokens_all) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluando"):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(tokens_all))
            batch = torch.from_numpy(tokens_all[start:end]).long().to(device)
            
            scores = fast_anomaly_score(model, batch, N_ROUNDS, MASK_RATIO)
            all_scores[start:end] = scores.cpu().numpy()
    
    # Filtrar posibles NaN
    valid_mask = np.isfinite(all_scores)
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        print(f"⚠️  {n_nan} scores NaN encontrados, se excluyen del análisis", flush=True)
    
    # ── Estadísticas por grupo ────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("ANOMALY SCORES POR GRUPO (final, epoch 41)", flush=True)
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
        print(f"  {g:6s}: mean={results[g]['mean']:.4f} ± {results[g]['std']:.4f}  "
              f"median={results[g]['median']:.4f}  n={results[g]['n']:,}", flush=True)
    
    # Ranking
    print("\nRanking (mayor score = más anómalo):", flush=True)
    sorted_groups = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for rank, (g, stats) in enumerate(sorted_groups, 1):
        print(f"  {rank}. {g}: {stats['mean']:.4f}", flush=True)
    
    # ── Guardar resultados ────────────────────────────────────────
    # JSON con estadísticas
    with open(OUTPUT_DIR / 'anomaly_scores.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEstadísticas guardadas en {OUTPUT_DIR / 'anomaly_scores.json'}", flush=True)
    
    # Scores individuales
    np.save(OUTPUT_DIR / 'anomaly_scores_all.npy', all_scores)
    np.save(OUTPUT_DIR / 'groups_all.npy', groups)
    print(f"Scores individuales guardados ({len(all_scores):,} objetos)", flush=True)
    
    # ── Histograma por grupo ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'SNIa': '#1f77b4', 'SNIbc': '#ff7f0e', 'SNII': '#2ca02c',
        'SNIIn': '#d62728', 'SLSN': '#9467bd', 'PISN': '#8c564b'
    }
    
    for g in GROUP_NAMES:
        mask = (groups == g) & valid_mask
        g_scores = all_scores[mask]
        ax.hist(g_scores, bins=100, alpha=0.5, label=f"{g} (n={mask.sum():,})",
                color=colors[g], density=True)
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title('Distribución de Anomaly Scores por Grupo de SN\n'
                 f'(MaskGIT cls, epoch 41, {N_ROUNDS} rondas)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'histogram_scores.png', dpi=150, bbox_inches='tight')
    print(f"Histograma guardado", flush=True)
    plt.close()
    
    # ── Boxplot por grupo ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ordenar por score medio (menor a mayor)
    order = [g for g, _ in sorted(results.items(), key=lambda x: x[1]['mean'])]
    data_boxes = [all_scores[(groups == g) & valid_mask] for g in order]
    box_colors = [colors[g] for g in order]
    
    bp = ax.boxplot(data_boxes, labels=order, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Añadir medias como puntos
    means = [results[g]['mean'] for g in order]
    ax.scatter(range(1, len(order)+1), means, color='black', marker='D',
               s=40, zorder=5, label='Media')
    
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_title('Anomaly Score por Grupo de SN\n'
                 f'(MaskGIT cls, epoch 41)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplot_scores.png', dpi=150, bbox_inches='tight')
    print(f"Boxplot guardado", flush=True)
    plt.close()
    
    # ── Tabla para la tesis ───────────────────────────────────────
    with open(OUTPUT_DIR / 'tabla_scores.txt', 'w') as f:
        f.write("Grupo     | Media  | Std    | Mediana | Q25    | Q75    | N\n")
        f.write("-" * 75 + "\n")
        for g, _ in sorted_groups:
            s = results[g]
            f.write(f"{g:9s} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['median']:.4f} | {s['q25']:.4f} | {s['q75']:.4f} | "
                    f"{s['n']:,}\n")
    print(f"Tabla guardada", flush=True)
    
    print(f"\n✅ Evaluación completa. Resultados en {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
