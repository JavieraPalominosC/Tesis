"""
Evaluación del clasificador supervisado sobre el codebook del VQ-VAE.

Genera:
    - Matriz de confusión
    - Accuracy por tipo de SN
    - UMAP del codebook coloreado por clase dominante
    - Análisis de pureza del codebook

Uso:
    python scripts/eval_classifier.py --checkpoint results/vqvae/checkpoints_cls01/MEJOR.ckpt
"""
import sys
sys.path.insert(0, '.')

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from omegaconf import OmegaConf

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset


def load_model(checkpoint_path, device):
    print(f"Cargando checkpoint: {checkpoint_path}")
    model = VQVAE.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def get_val_loader(folds_path, labels_path, image_size, batch_size):
    with open(folds_path) as f:
        folds = json.load(f)
    with open(labels_path) as f:
        labels_map = json.load(f)

    val_paths = folds['0']['val']
    dataset   = VQVAEDataset(val_paths, image_size, labels_map)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=False)
    print(f"Val set: {len(val_paths):,} imágenes")
    return loader


def evaluate_classifier(model, loader, class_names, device, output_dir):
    all_preds  = []
    all_labels = []

    print("Evaluando clasificador...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            valid  = labels >= 0
            if not valid.any():
                continue
            _, _, _, _, _, _, z_q = model(images[valid])
            logits = model.classifier(z_q)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels[valid].cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f"\nAccuracy global: {acc:.4f} ({acc*100:.1f}%)")

    names = [class_names.get(str(i), str(i)) for i in range(len(class_names))]
    print("\nReporte por clase:")
    print(classification_report(all_labels, all_preds, target_names=names, digits=3))

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusión (acc={acc:.3f})')
    for i in range(len(names)):
        for j in range(len(names)):
            val   = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
    plt.tight_layout()
    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Matriz de confusión guardada: {save_path}")
    plt.close()

    return all_preds, all_labels, acc


def analyze_codebook(model, loader, class_names, device, output_dir):
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP no instalado. Instala con: pip install umap-learn")
        return

    K = model.codebook.num_embeddings
    C = len(class_names)
    counts = torch.zeros(K, C, dtype=torch.long)

    print("\nAnalizando afinidad del codebook...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            valid  = labels >= 0
            if not valid.any():
                continue
            indices  = model.encode_to_indices(images[valid])
            B, H, W  = indices.shape
            flat_idx = indices.view(-1)
            flat_lbl = labels[valid].unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
            for k in range(K):
                mask = flat_idx == k
                if mask.any():
                    for c in flat_lbl[mask]:
                        if 0 <= c < C:
                            counts[k, c] += 1

    total    = counts.sum(dim=1).float().clamp(min=1)
    dominant = counts.argmax(dim=1)
    purity   = counts.max(dim=1).values.float() / total
    active   = counts.sum(dim=1) > 0

    mean_purity = purity.mean().item()
    print(f"Pureza media del codebook: {mean_purity:.3f}")
    print(f"Embeddings activos: {active.sum().item()} / {K}")

    embeddings_np = model.codebook.embedding.detach().cpu().float().numpy()
    print("Corriendo UMAP...")
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb_2d  = reducer.fit_transform(embeddings_np)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    names  = [class_names.get(str(i), str(i)) for i in range(C)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for c in range(C):
        mask = (dominant == c) & active
        if mask.any():
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=colors[c % len(colors)], label=names[c], s=20, alpha=0.7)
    ax.set_title(f'UMAP codebook — clase dominante\n(pureza media={mean_purity:.3f})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ax = axes[1]
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=purity.numpy(), cmap='RdYlGn', vmin=0, vmax=1, s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Pureza')
    ax.set_title('UMAP codebook — pureza por embedding')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    save_path = output_dir / 'umap_codebook_cls.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"UMAP guardado: {save_path}")
    plt.close()

    return {'mean_purity': mean_purity, 'active_embeddings': int(active.sum().item()), 'total_embeddings': K}


def main(args):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open('data/folds/class_names.json') as f:
        class_names = json.load(f)

    model = load_model(args.checkpoint, device)

    if model.classifier is None:
        print("ERROR: Este checkpoint no tiene clasificador.")
        return

    loader = get_val_loader('data/folds/folds.json', 'data/folds/labels.json', 256, args.batch_size)

    preds, labels, acc = evaluate_classifier(model, loader, class_names, device, output_dir)
    stats = analyze_codebook(model, loader, class_names, device, output_dir)

    if stats:
        stats['accuracy'] = float(acc)
        with open(output_dir / 'codebook_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats guardadas: {output_dir / 'codebook_stats.json'}")

    print(f"\nEvaluación completada. Resultados en {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='results/vqvae/eval_cls')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
