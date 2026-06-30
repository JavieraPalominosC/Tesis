"""
Evaluación del clasificador supervisado sobre el codebook del VQ-VAE.
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

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset


def load_model(checkpoint_path, device):
    print(f"Cargando checkpoint: {checkpoint_path}", flush=True)
    model = VQVAE.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    print("Modelo cargado OK", flush=True)
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
    print(f"Val set: {len(val_paths):,} imágenes ({len(loader)} batches)", flush=True)
    return loader


def evaluate_classifier(model, loader, class_names, device, output_dir):
    all_preds  = []
    all_labels = []
    print("\n[1/3] Evaluando clasificador...", flush=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i % 200 == 0:
                print(f"  batch {i}/{len(loader)}...", flush=True)
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
    print(f"\nAccuracy global: {acc:.4f} ({acc*100:.1f}%)", flush=True)
    names = [class_names.get(str(i), str(i)) for i in range(len(class_names))]
    print(classification_report(all_labels, all_preds, target_names=names, digits=3))

    print("\n[2/3] Generando matriz de confusión...", flush=True)
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
    print(f"Matriz guardada: {save_path}", flush=True)
    plt.close()
    return all_preds, all_labels, acc


def analyze_codebook(model, loader, class_names, device, output_dir):
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP no instalado.", flush=True)
        return

    K = model.codebook.num_embeddings
    C = len(class_names)
    counts = torch.zeros(K, C, dtype=torch.long)

    print("\n[3/3] Analizando afinidad del codebook (vectorizado)...", flush=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i % 200 == 0:
                print(f"  batch {i}/{len(loader)}...", flush=True)
            images = images.to(device)
            labels = labels.to(device)
            valid  = labels >= 0
            if not valid.any():
                continue

            indices  = model.encode_to_indices(images[valid])
            B, H, W  = indices.shape
            flat_idx = indices.view(-1).cpu()
            flat_lbl = labels[valid].unsqueeze(1).unsqueeze(2)\
                                    .expand(B, H, W).reshape(-1).cpu()

            valid_mask = (flat_lbl >= 0) & (flat_lbl < C)
            flat_idx   = flat_idx[valid_mask]
            flat_lbl   = flat_lbl[valid_mask]

            counts.index_put_(
                (flat_idx, flat_lbl),
                torch.ones(len(flat_idx), dtype=torch.long),
                accumulate=True
            )

    print("Conteo completado, calculando pureza...", flush=True)
    total    = counts.sum(dim=1).float().clamp(min=1)
    dominant = counts.argmax(dim=1)
    purity   = counts.max(dim=1).values.float() / total
    active   = counts.sum(dim=1) > 0

    mean_purity = purity.mean().item()
    print(f"Pureza media del codebook: {mean_purity:.3f}", flush=True)
    print(f"Embeddings activos: {active.sum().item()} / {K}", flush=True)

    print("Corriendo UMAP...", flush=True)
    embeddings_np = model.codebook.embedding.detach().cpu().float().numpy()
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb_2d  = reducer.fit_transform(embeddings_np)
    print("UMAP listo", flush=True)

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
    print(f"UMAP guardado: {save_path}", flush=True)
    plt.close()

    return {'mean_purity': mean_purity, 'active_embeddings': int(active.sum().item()), 'total_embeddings': K}


def main(args):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)

    with open('data/folds/class_names.json') as f:
        class_names = json.load(f)

    model  = load_model(args.checkpoint, device)
    loader = get_val_loader('data/folds/folds.json', 'data/folds/labels.json', 256, args.batch_size)

    acc = None
    if not args.skip_classifier:
        _, _, acc = evaluate_classifier(model, loader, class_names, device, output_dir)
    else:
        print("\n[1/3] y [2/3] omitidos (--skip_classifier)", flush=True)
        # Intentar cargar accuracy previa si existe
        stats_path = output_dir / 'codebook_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                prev = json.load(f)
                acc = prev.get('accuracy', None)
                if acc:
                    print(f"Accuracy previa cargada: {acc:.4f}", flush=True)

    stats = analyze_codebook(model, loader, class_names, device, output_dir)

    if stats:
        if acc is not None:
            stats['accuracy'] = float(acc)
        with open(output_dir / 'codebook_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Stats: {output_dir / 'codebook_stats.json'}", flush=True)

    print(f"\n✅ Completado. Resultados en {output_dir}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='results/vqvae/eval_cls')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--skip_classifier', action='store_true',
                        help='Saltar pasos 1 y 2 (clasificador y matriz de confusión)')
    args = parser.parse_args()
    main(args)
