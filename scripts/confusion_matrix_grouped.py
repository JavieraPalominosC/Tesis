"""
Genera matriz de confusión agrupada (6 grupos) desde el checkpoint del VQ-VAE cls.
Reporta F1 macro, F1 weighted y F1 por grupo.
"""

import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import json
import glob

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset

# ── Agrupamiento ──────────────────────────────────────────────────
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
GROUP_IDX   = {name: i for i, name in enumerate(GROUP_NAMES)}

CHECKPOINT  = 'results/vqvae/checkpoints_cls01/vqvae-epoch=07-train/loss=0.1188.ckpt'
OUTPUT_DIR  = Path('results/vqvae/eval_cls')
BATCH_SIZE  = 64
NUM_WORKERS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Modelo ────────────────────────────────────────────────────────
model = VQVAE.load_from_checkpoint(CHECKPOINT, map_location=device)
model.eval()
model.to(device)
print("Modelo cargado OK", flush=True)

# ── Dataset ───────────────────────────────────────────────────────
with open('data/folds/folds.json') as f:
    folds = json.load(f)
with open('data/folds/labels.json') as f:
    labels_map = json.load(f)

val_paths = folds['0']['val']
dataset   = VQVAEDataset(val_paths, 256, labels_map)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS)
print(f"Val set: {len(val_paths):,} imágenes ({len(loader)} batches)", flush=True)

# ── Inference ─────────────────────────────────────────────────────
all_true, all_pred = [], []

with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}...", flush=True)
        images = images.to(device)
        x_hat, vq_loss, indices, perplexity, cb_loss, cm_loss, z_q = model(images)
        logits = model.classifier(z_q)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
print(f"Inference completa: {len(all_true):,} muestras", flush=True)

# ── Agrupamiento ──────────────────────────────────────────────────
def to_group(arr):
    return np.array([GROUP_IDX[GROUP_MAP[int(x)]] for x in arr])

g_true = to_group(all_true)
g_pred = to_group(all_pred)

# ── Métricas ──────────────────────────────────────────────────────
acc_grouped = float(np.mean(g_true == g_pred))
f1_macro    = float(f1_score(g_true, g_pred, average='macro'))
f1_weighted = float(f1_score(g_true, g_pred, average='weighted'))
f1_per_class = f1_score(g_true, g_pred, average=None)

print(f"\nExactitud agrupada: {acc_grouped:.3f}", flush=True)
print(f"F1 macro:           {f1_macro:.3f}", flush=True)
print(f"F1 weighted:        {f1_weighted:.3f}", flush=True)
print("\nF1 por grupo:", flush=True)
for name, f1 in zip(GROUP_NAMES, f1_per_class):
    print(f"  {name}: {f1:.3f}", flush=True)
print("\n", classification_report(g_true, g_pred, target_names=GROUP_NAMES), flush=True)

# Guardar métricas
metrics = {
    'acc_grouped':  acc_grouped,
    'f1_macro':     f1_macro,
    'f1_weighted':  f1_weighted,
    'f1_per_class': {name: float(f1) for name, f1 in zip(GROUP_NAMES, f1_per_class)}
}
with open(OUTPUT_DIR / 'metrics_grouped.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Métricas guardadas en {OUTPUT_DIR / 'metrics_grouped.json'}", flush=True)

# ── Matriz de confusión normalizada por fila ───────────────────────
n  = len(GROUP_NAMES)
cm = np.zeros((n, n), dtype=float)
for t, p in zip(g_true, g_pred):
    cm[t, p] += 1
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(GROUP_NAMES, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(GROUP_NAMES, fontsize=11)
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title(
    f'Matriz de confusión agrupada\n'
    f'F1 macro={f1_macro:.3f}  |  F1 weighted={f1_weighted:.3f}',
    fontsize=11
)

for i in range(n):
    for j in range(n):
        val = cm_norm[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=12, color=color)

plt.tight_layout()
out_path = OUTPUT_DIR / 'confusion_matrix_grouped.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Guardado: {out_path}", flush=True)
