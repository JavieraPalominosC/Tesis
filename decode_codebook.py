"""
Decodificación individual de cada token del codebook del VQ-VAE.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys


def load_vqvae_model(checkpoint_path, device):
    sys.path.insert(0, '.')
    from src.models.vqvae.model import VQVAE

    print(f"Cargando checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        print(f"Hyperparams: {hparams}")
        model = VQVAE(**{k: v for k, v in hparams.items()
                        if k in VQVAE.__init__.__code__.co_varnames})
        state_dict = ckpt['state_dict']
        clean_state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
                       for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
    else:
        model = VQVAE()
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    model.eval()
    model.to(device)
    print(f"Modelo cargado. Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_codebook_embeddings(model):
    for name in ['vq', 'quantizer', 'codebook']:
        if hasattr(model, name):
            vq = getattr(model, name)
            for emb_name in ['embedding', 'embeddings', 'codebook', '_embedding']:
                if hasattr(vq, emb_name):
                    emb = getattr(vq, emb_name)
                    if hasattr(emb, 'weight'):
                        return emb.weight.data
                    if isinstance(emb, torch.Tensor):
                        return emb.data
    raise RuntimeError("No se pudo encontrar el codebook en el modelo")


def decode_tokens_batched(model, codebook, K, grid_size, device, batch_size=8):
    D = codebook.shape[1]
    images = []
    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        B = end - start
        embs = codebook[start:end]
        z_q = embs.view(B, 1, 1, D).expand(B, grid_size, grid_size, D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous().to(device)

        with torch.no_grad():
            x_hat = model.decoder(z_q)

        for i in range(B):
            img = x_hat[i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2 if img.min() < 0 else img
            img = np.clip(img, 0, 1)
            images.append(img)

        del x_hat, z_q
        if (start // batch_size) % 8 == 0:
            print(f"  {end}/{K}")

    return images


def compute_token_usage(tokens_path, K):
    print(f"Cargando tokens desde {tokens_path}...")
    tokens = np.load(tokens_path)
    print(f"  Shape: {tokens.shape}")
    counts = np.bincount(tokens.ravel(), minlength=K)
    print(f"  Tokens usados: {(counts > 0).sum()}/{K}")
    print(f"  Token más frecuente: {counts.argmax()} (n={counts.max():,})")
    print(f"  Tokens muertos (count=0): {(counts == 0).sum()}")
    return counts


def plot_codebook_grid(images, counts, output_path, title="Codebook completo", cols=32):
    K = len(images)
    rows = (K + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 0.7, rows * 0.7))
    fig.patch.set_facecolor('#0d0d0d')

    for k in range(K):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.imshow(images[k])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#222')
            spine.set_linewidth(0.3)
        if counts[k] == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('#ff4444')
                spine.set_linewidth(0.6)

    fig.suptitle(f'{title} (K={K})  ·  rojo = token muerto',
                 color='white', fontsize=11, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Guardado: {output_path}")
    plt.close()


def plot_top_tokens(images, counts, output_path, n=32, top=True):
    if top:
        indices = np.argsort(counts)[::-1][:n]
        title = f"Top {n} tokens MAS usados"
    else:
        used = np.where(counts > 0)[0]
        indices = used[np.argsort(counts[used])[:n]]
        title = f"Top {n} tokens MENOS usados"

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    fig.patch.set_facecolor('#0d0d0d')
    axes = axes.flatten()

    for ax, k in zip(axes, indices):
        ax.imshow(images[k])
        ax.set_title(f'#{k}\nn={counts[k]:,}', color='white', fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
            spine.set_linewidth(0.4)

    for ax in axes[len(indices):]:
        ax.axis('off')

    fig.suptitle(title, color='white', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Guardado: {output_path}")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_vqvae_model(args.checkpoint, device)

    codebook = get_codebook_embeddings(model)
    K, D = codebook.shape
    print(f"Codebook: K={K} embeddings de dimensión D={D}")

    print(f"\nDecodificando {K} tokens individualmente (batched)...")
    images = decode_tokens_batched(model, codebook, K,
                                   grid_size=args.grid_size,
                                   device=device,
                                   batch_size=args.batch_size)

    counts = compute_token_usage(args.tokens_path, K)
    np.save(output_dir / 'token_counts.npy', counts)

    images_arr = np.stack(images)
    np.save(output_dir / 'token_images.npy', images_arr)
    print(f"Imagenes de tokens guardadas: {images_arr.shape}")

    plot_codebook_grid(images, counts,
                       output_dir / 'codebook_grid.png',
                       title="Tokens decodificados individualmente")

    plot_top_tokens(images, counts,
                    output_dir / 'codebook_top_used.png', n=32, top=True)

    plot_top_tokens(images, counts,
                    output_dir / 'codebook_least_used.png', n=32, top=False)

    print(f"\nListo! Resultados en {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
        default='results/vqvae/checkpoints/vqvae-epoch=06-train/loss=0.0003.ckpt')
    parser.add_argument('--tokens_path', default='data/tokens/tokens.npy')
    parser.add_argument('--output_dir',  default='results/codebook')
    parser.add_argument('--grid_size',   type=int, default=8)
    parser.add_argument('--batch_size',  type=int, default=8)
    args = parser.parse_args()
    main(args)
