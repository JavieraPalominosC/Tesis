"""
Recalcula la accuracy de clasificacion de validacion para una representacion.
El val/acc del entrenamiento salio mal (logging on_step), esto lo computa limpio.

Uso: python scripts/eval_classification.py <REP>
     REP in {overlay_lines, points, waterfall, waterfall_global}
"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from src.models.vqvae.model import VQVAE

REP = sys.argv[1]

VQVAE_CKPTS = {
    'overlay_lines':    'results/vqvae_overlay_full/checkpoints/vqvae-epoch=17-train/loss=0.1378.ckpt',
    'points':           'results/vqvae_points_full/checkpoints/vqvae-epoch=52-train/loss=0.0658.ckpt',
    'waterfall':        'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt',
    'waterfall_global': 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt',
}
FOLDS_PATHS = {
    'overlay_lines':    'data/folds_overlay_full.json',
    'points':           'data/folds_points_full.json',
    'waterfall':        'data/folds_waterfall_full.json',
    'waterfall_global': 'data/folds_waterfall_global_full.json',
}
LABELS_PATHS = {
    'overlay_lines':    'data/labels_overlay_full.json',
    'points':           'data/labels_points_full.json',
    'waterfall':        'data/labels_waterfall_full.json',
    'waterfall_global': 'data/labels_waterfall_global_full.json',
}

N_SAMPLE = 30000  # muestra del set de validacion (suficiente para acc estable)
SEED = 42

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = VQVAE_CKPTS[REP]
    print(f"[{REP}] Cargando VQ-VAE: {ckpt}", flush=True)

    model = VQVAE.load_from_checkpoint(ckpt, map_location=device)
    model.eval(); model.to(device)
    if model.classifier is None:
        print(f"[{REP}] Este modelo no tiene cabeza de clasificacion.", flush=True)
        return

    with open(FOLDS_PATHS[REP]) as f:
        folds = json.load(f)
    val_paths = folds["0"]["val"]
    with open(LABELS_PATHS[REP]) as f:
        labels_map = json.load(f)

    rng = np.random.default_rng(SEED)
    if len(val_paths) > N_SAMPLE:
        val_paths = list(rng.choice(val_paths, size=N_SAMPLE, replace=False))
    print(f"[{REP}] Evaluando {len(val_paths):,} imagenes de validacion", flush=True)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    correct = 0; total = 0
    BATCH = 128
    xs = []; ys = []
    with torch.no_grad():
        for p in tqdm(val_paths):
            snid = Path(p).stem
            y = labels_map.get(snid, -1)
            if y < 0:
                continue
            img = Image.open(p).convert('RGB')
            xs.append(transform(img)); ys.append(y)
            if len(xs) == BATCH:
                x = torch.stack(xs).to(device)
                z_e = model.encoder(x)
                z_q, *_ = model.codebook(z_e)
                logits = model.classifier(z_q)
                pred = logits.argmax(1).cpu().numpy()
                correct += (pred == np.array(ys)).sum(); total += len(ys)
                xs = []; ys = []
        if xs:
            x = torch.stack(xs).to(device)
            z_e = model.encoder(x)
            z_q, *_ = model.codebook(z_e)
            logits = model.classifier(z_q)
            pred = logits.argmax(1).cpu().numpy()
            correct += (pred == np.array(ys)).sum(); total += len(ys)

    acc = correct / total if total else 0.0
    print(f"\n[{REP}] Accuracy de validacion (17 clases): {acc:.4f}  ({correct:,}/{total:,})", flush=True)

if __name__ == "__main__":
    main()
