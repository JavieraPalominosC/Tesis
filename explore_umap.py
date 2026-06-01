"""
Exploración del UMAP de tokens VQ-VAE.

1. Zoom en las regiones extremas (brazos del UMAP)
2. Muestra imágenes de ejemplo de cada región
3. Pie chart de tipos de SN por región

Uso (en tu Mac, después de descargar los resultados del cluster):
    python explore_umap.py \
        --umap_dir    results/umap \
        --images_dir  data/images/elasticc_1/2grid

Si tienes los archivos en otro lugar:
    python explore_umap.py --umap_dir ~/Desktop/umap --images_dir /ruta/a/imagenes
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
from PIL import Image

# ── Colores por tipo ────────────────────────────────────────────────────────
SN_COLORS = {
    'SNIa-SALT2':        '#e6194b',
    'SNIa-91bg':         '#f58231',
    'SNIax':             '#ffe119',
    'SNII-Templates':    '#3cb44b',
    'SNII-NMF':          '#42d4f4',
    'SNII+HostXT_V19':   '#4363d8',
    'SNIb-Templates':    '#911eb4',
    'SNIb+HostXT_V19':   '#f032e6',
    'SNIc-Templates':    '#a9a9a9',
    'SNIc+HostXT_V19':   '#9a6324',
    'SNIcBL+HostXT_V19': '#800000',
    'SNIIb+HostXT_V19':  '#469990',
    'SNIIn-MOSFIT':      '#000075',
    'SNIIn+HostXT_V19':  '#dcbeff',
    'SLSN-I+host':       '#aaffc3',
    'SLSN-I_no_host':    '#ffd8b1',
    'PISN':              '#bbbbbb',
    'Unknown':           '#555555',
}


def define_regions(embedding):
    """
    Define regiones de interés basadas en los percentiles del embedding.
    Divide el espacio UMAP en: centro, brazo derecho, brazo inferior, extremos.
    """
    x, y = embedding[:, 0], embedding[:, 1]

    regions = {
        'Centro (masa principal)': (
            (x > np.percentile(x, 20)) & (x < np.percentile(x, 70)) &
            (y > np.percentile(y, 30)) & (y < np.percentile(y, 80))
        ),
        'Brazo derecho': (
            x > np.percentile(x, 90)
        ),
        'Brazo inferior': (
            y < np.percentile(y, 10)
        ),
        'Extremo superior': (
            y > np.percentile(y, 92)
        ),
        'Borde izquierdo': (
            x < np.percentile(x, 8)
        ),
    }
    return regions


def plot_zoom_regions(embedding, labels, regions, output_path):
    """UMAP completo con las regiones marcadas."""
    unique_labels = sorted(set(labels))

    fig, axes = plt.subplots(1, len(regions) + 1, figsize=(6 * (len(regions) + 1), 6))
    fig.patch.set_facecolor('#0d0d0d')

    def scatter_ax(ax, mask_highlight=None, title=''):
        ax.set_facecolor('#0d0d0d')
        for sn_type in unique_labels:
            type_mask = np.array([l == sn_type for l in labels])
            alpha = 0.15 if mask_highlight is not None else 0.4
            ax.scatter(embedding[type_mask, 0], embedding[type_mask, 1],
                       c=SN_COLORS.get(sn_type, '#aaa'), s=1, alpha=alpha,
                       linewidths=0, rasterized=True)
        if mask_highlight is not None:
            ax.scatter(embedding[mask_highlight, 0], embedding[mask_highlight, 1],
                       c='white', s=4, alpha=0.8, linewidths=0, rasterized=True)
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='#555')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # Panel 1: UMAP completo con todas las regiones marcadas
    ax0 = axes[0]
    ax0.set_facecolor('#0d0d0d')
    for sn_type in unique_labels:
        type_mask = np.array([l == sn_type for l in labels])
        ax0.scatter(embedding[type_mask, 0], embedding[type_mask, 1],
                    c=SN_COLORS.get(sn_type, '#aaa'), s=1, alpha=0.3,
                    linewidths=0, rasterized=True)

    region_colors = ['#ff4444', '#44aaff', '#44ff88', '#ffaa00', '#ff44ff']
    for (name, mask), color in zip(regions.items(), region_colors):
        if mask.sum() > 0:
            cx = embedding[mask, 0].mean()
            cy = embedding[mask, 1].mean()
            ax0.annotate(name, (cx, cy), color=color, fontsize=7,
                         ha='center', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', alpha=0.7))
    ax0.set_title('UMAP completo + regiones', color='white', fontsize=9)
    ax0.tick_params(colors='#555')
    for spine in ax0.spines.values():
        spine.set_edgecolor('#333')

    # Un panel por región con zoom
    for ax, (name, mask), color in zip(axes[1:], regions.items(), region_colors):
        scatter_ax(ax, mask_highlight=mask, title=f'{name}\n(n={mask.sum():,})')
        if mask.sum() > 0:
            xm, ym = embedding[mask, 0], embedding[mask, 1]
            pad_x = max((xm.max() - xm.min()) * 0.3, 0.5)
            pad_y = max((ym.max() - ym.min()) * 0.3, 0.5)
            ax.set_xlim(xm.min() - pad_x, xm.max() + pad_x)
            ax.set_ylim(ym.min() - pad_y, ym.max() + pad_y)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Guardado: {output_path}")
    plt.close()


def plot_region_composition(labels, regions, output_path):
    """Pie charts de composición por tipo de SN en cada región."""
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.patch.set_facecolor('#0d0d0d')
    if n == 1:
        axes = [axes]

    for ax, (name, mask) in zip(axes, regions.items()):
        ax.set_facecolor('#0d0d0d')
        region_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        counts = Counter(region_labels)
        # Mostrar solo tipos con >1%
        total = sum(counts.values())
        filtered = {k: v for k, v in counts.items() if v / total > 0.01}
        other = total - sum(filtered.values())
        if other > 0:
            filtered['Otros'] = other

        colors = [SN_COLORS.get(k, '#888') for k in filtered.keys()]
        wedges, texts, autotexts = ax.pie(
            filtered.values(), labels=filtered.keys(),
            colors=colors, autopct='%1.1f%%',
            textprops={'color': 'white', 'fontsize': 7},
            pctdistance=0.8,
        )
        for at in autotexts:
            at.set_fontsize(6)
        ax.set_title(f'{name}\n(n={mask.sum():,})', color='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Guardado: {output_path}")
    plt.close()


def plot_example_images(embedding, labels, paths_file, regions, images_dir, output_path,
                        n_examples=6):
    """
    Muestra imágenes de ejemplo de cada región.
    Toma puntos de los extremos de cada región (los más alejados del centro).
    """
    if not Path(paths_file).exists():
        print(f"[SKIP] No se encontró {paths_file} — saltando imágenes de ejemplo")
        return

    with open(paths_file) as f:
        all_paths = [line.strip() for line in f if line.strip()]

    images_dir = Path(images_dir)
    cx = embedding[:, 0].mean()
    cy = embedding[:, 1].mean()

    n_regions = len(regions)
    fig = plt.figure(figsize=(n_examples * 2, n_regions * 2.5))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle('Imágenes de ejemplo por región UMAP', color='white', fontsize=11)

    gs = gridspec.GridSpec(n_regions, n_examples, figure=fig,
                           hspace=0.4, wspace=0.1)

    for row, (name, mask) in enumerate(regions.items()):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Ordenar por distancia al centro (más extremos primero)
        dists = np.sqrt((embedding[indices, 0] - cx)**2 +
                        (embedding[indices, 1] - cy)**2)
        sorted_idx = indices[np.argsort(dists)[::-1]]
        selected = sorted_idx[:n_examples]

        for col, idx in enumerate(selected):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#111')

            # Buscar imagen
            snid = Path(all_paths[idx]).stem
            img_path = images_dir / f"{snid}.png"

            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f'SNID\n{snid}', ha='center', va='center',
                        color='white', fontsize=6, transform=ax.transAxes)

            label = labels[idx] if idx < len(labels) else '?'
            ax.set_title(label, color=SN_COLORS.get(label, '#aaa'),
                         fontsize=5, pad=2)
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(name, color='white', fontsize=7, rotation=90,
                              labelpad=4)

    plt.savefig(output_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Guardado: {output_path}")
    plt.close()


def main(args):
    umap_dir   = Path(args.umap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar embedding y etiquetas ───────────────────────────────
    embedding = np.load(umap_dir / 'umap_embedding.npy')
    labels_raw = np.load(umap_dir / 'umap_labels.npy', allow_pickle=True)
    labels = list(labels_raw)
    print(f"Embedding: {embedding.shape}, etiquetas: {len(labels)}")

    print("\nDistribución global:")
    for t, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
        print(f"  {t:25s}: {cnt:,} ({cnt/len(labels)*100:.1f}%)")

    # ── Definir regiones ───────────────────────────────────────────
    regions = define_regions(embedding)
    print("\nRegiones:")
    for name, mask in regions.items():
        print(f"  {name:30s}: {mask.sum():,} puntos")

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerando figuras...")

    plot_zoom_regions(
        embedding, labels, regions,
        output_dir / 'umap_zoom_regions.png'
    )

    plot_region_composition(
        labels, regions,
        output_dir / 'umap_composition_pies.png'
    )

    paths_file = umap_dir / '..' / '..' / 'tokens' / 'paths.txt'
    # Intentar varias rutas posibles
    for candidate in [
        umap_dir / '..' / '..' / 'tokens' / 'paths.txt',
        Path('data/tokens/paths.txt'),
        Path(args.paths_file) if args.paths_file else None,
    ]:
        if candidate and Path(candidate).exists():
            paths_file = candidate
            break

    plot_example_images(
        embedding, labels,
        paths_file=str(paths_file),
        regions=regions,
        images_dir=args.images_dir,
        output_path=output_dir / 'umap_example_images.png',
        n_examples=6,
    )

    print(f"\n✅ Listo! Figuras en: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--umap_dir',    default='results/umap',
                        help='Directorio con umap_embedding.npy y umap_labels.npy')
    parser.add_argument('--images_dir',  default='data/images/elasticc_1/2grid',
                        help='Directorio con las imágenes PNG')
    parser.add_argument('--output_dir',  default='results/umap',
                        help='Dónde guardar las figuras')
    parser.add_argument('--paths_file',  default=None,
                        help='Path explícito a paths.txt (opcional)')
    args = parser.parse_args()
    main(args)
