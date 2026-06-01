"""
Visualización UMAP de tokens VQ-VAE coloreados por tipo de supernova.

Pipeline:
    1. Cargar tokens (N, 1024) desde data/tokens/tokens.npy
    2. Cargar SNID → tipo de SN desde los parquets
    3. Calcular bag-of-codes: histograma de 512 tokens por imagen
    4. UMAP a 2D
    5. Scatter plot coloreado por tipo de SN → results/umap_tokens.png

Uso:
    python umap_tokens.py
    python umap_tokens.py --max_samples 50000  # para prueba rápida
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Colores por tipo de SN ──────────────────────────────────────────────────
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
    'PISN':              '#ffffff',
}

# Nombre del archivo parquet → nombre corto del tipo
PARQUET_TO_TYPE = {
    'lc_SNIa-SALT2.parquet':        'SNIa-SALT2',
    'lc_SNIa-91bg.parquet':         'SNIa-91bg',
    'lc_SNIax.parquet':             'SNIax',
    'lc_SNII-Templates.parquet':    'SNII-Templates',
    'lc_SNII-NMF.parquet':          'SNII-NMF',
    'lc_SNII+HostXT_V19.parquet':   'SNII+HostXT_V19',
    'lc_SNIb-Templates.parquet':    'SNIb-Templates',
    'lc_SNIb+HostXT_V19.parquet':   'SNIb+HostXT_V19',
    'lc_SNIc-Templates.parquet':    'SNIc-Templates',
    'lc_SNIc+HostXT_V19.parquet':   'SNIc+HostXT_V19',
    'lc_SNIcBL+HostXT_V19.parquet': 'SNIcBL+HostXT_V19',
    'lc_SNIIb+HostXT_V19.parquet':  'SNIIb+HostXT_V19',
    'lc_SNIIn-MOSFIT.parquet':      'SNIIn-MOSFIT',
    'lc_SNIIn+HostXT_V19.parquet':  'SNIIn+HostXT_V19',
    'lc_SLSN-I+host.parquet':       'SLSN-I+host',
    'lc_SLSN-I_no_host.parquet':    'SLSN-I_no_host',
    'lc_PISN.parquet':              'PISN',
}


def build_snid_to_type(raw_dir: Path) -> dict:
    """
    Lee todos los parquets de supernovas y construye un dict SNID → tipo.
    Solo lee la columna SNID de cada archivo para no cargar todo en memoria.
    """
    print("Construyendo mapa SNID → tipo de SN...")
    snid_to_type = {}

    for parquet_name, sn_type in PARQUET_TO_TYPE.items():
        path = raw_dir / parquet_name
        if not path.exists():
            print(f"  [SKIP] No encontrado: {parquet_name}")
            continue
        try:
            # Solo leer columna SNID (mucho más rápido)
            df = pd.read_parquet(path, columns=['SNID'])
            for snid in df['SNID'].unique():
                snid_to_type[snid] = sn_type
            print(f"  {sn_type}: {len(df['SNID'].unique()):,} SNIDs")
        except Exception as e:
            print(f"  [ERROR] {parquet_name}: {e}")

    print(f"Total SNIDs mapeados: {len(snid_to_type):,}")
    return snid_to_type


def load_snid_index(tokens_dir: Path) -> list:
    """
    Carga el índice de SNIDs desde paths.txt.
    Cada línea es del tipo: data/images/elasticc_1/2grid/1000007.png
    El SNID es el nombre del archivo sin extensión.
    """
    paths_file = tokens_dir / 'paths.txt'
    if paths_file.exists():
        with open(paths_file) as f:
            snids = [Path(line.strip()).stem for line in f if line.strip()]
        print(f"Cargados {len(snids):,} SNIDs desde {paths_file}")
        return snids

    print(f"[WARN] No se encontró paths.txt en {tokens_dir}")
    print("       Se usará índice numérico sin etiquetas de tipo.")
    return None


def compute_bag_of_codes(tokens: np.ndarray, num_codes: int = 512) -> np.ndarray:
    """
    Calcula el histograma de tokens (bag-of-codes) por imagen.

    Args:
        tokens:    (N, 1024) uint16 — tokens del VQ-VAE
        num_codes: tamaño del codebook (K)

    Returns:
        histograms: (N, K) float32 — frecuencia normalizada de cada token
    """
    N = tokens.shape[0]
    print(f"Calculando bag-of-codes para {N:,} imágenes (K={num_codes})...")
    
    histograms = np.zeros((N, num_codes), dtype=np.float32)
    batch = 5000
    for start in tqdm(range(0, N, batch), desc="Bag-of-codes"):
        end = min(start + batch, N)
        chunk = tokens[start:end].astype(np.int32)  # (B, 1024)
        for i, row in enumerate(chunk):
            hist = np.bincount(row, minlength=num_codes)
            histograms[start + i] = hist / hist.sum()  # normalizar a frecuencias

    return histograms


def run_umap(histograms: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
             random_state: int = 42) -> np.ndarray:
    """Reduce histogramas a 2D con UMAP."""
    try:
        import umap
    except ImportError:
        print("Instalando umap-learn...")
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'])
        import umap

    print(f"Ejecutando UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='cosine',   # bueno para histogramas de frecuencia
        random_state=random_state,
        low_memory=False,
        verbose=True,
    )
    embedding = reducer.fit_transform(histograms)
    print(f"UMAP completado. Shape: {embedding.shape}")
    return embedding


def plot_umap(embedding: np.ndarray, labels: list, output_path: Path,
              title: str = "UMAP de tokens VQ-VAE por tipo de supernova"):
    """
    Scatter plot del embedding UMAP coloreado por tipo de SN.
    """
    unique_labels = sorted(set(labels))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    for sn_type in unique_labels:
        mask = np.array([l == sn_type for l in labels])
        color = SN_COLORS.get(sn_type, '#aaaaaa')
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=sn_type,
            s=1.5, alpha=0.4, linewidths=0,
            rasterized=True,
        )

    # Leyenda
    patches = [
        mpatches.Patch(color=SN_COLORS.get(t, '#aaaaaa'), label=t)
        for t in unique_labels
    ]
    legend = ax.legend(
        handles=patches, loc='upper right', fontsize=7,
        framealpha=0.3, facecolor='#1a1a1a', edgecolor='#444',
        markerscale=3, ncol=2,
    )
    for text in legend.get_texts():
        text.set_color('white')

    ax.set_title(title, color='white', fontsize=13, pad=12)
    ax.set_xlabel("UMAP 1", color='#aaaaaa', fontsize=10)
    ax.set_ylabel("UMAP 2", color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Figura guardada: {output_path}")
    plt.close()


def main(args):
    tokens_dir = Path(args.tokens_dir)
    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar tokens ───────────────────────────────────────────
    tokens_path = tokens_dir / 'tokens.npy'
    print(f"Cargando tokens desde {tokens_path}...")
    tokens = np.load(tokens_path)  # (N, 1024) uint16
    print(f"  Shape: {tokens.shape}, dtype: {tokens.dtype}")

    num_codes = int(tokens.max()) + 1  # inferir K desde los datos
    print(f"  Número de códigos inferido: {num_codes}")

    # ── 2. Subsamplear si se pide ──────────────────────────────────
    N = tokens.shape[0]
    if args.max_samples and args.max_samples < N:
        print(f"Submuestreando {args.max_samples:,} de {N:,} imágenes...")
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=args.max_samples, replace=False)
        idx.sort()
        tokens = tokens[idx]
    else:
        idx = np.arange(N)

    # ── 3. Cargar SNIDs y construir etiquetas ──────────────────────
    snids_all = load_snid_index(tokens_dir)
    
    if snids_all is not None:
        snids = [snids_all[i] for i in idx]
        snid_to_type = build_snid_to_type(raw_dir)
        labels = [snid_to_type.get(s, 'Unknown') for s in snids]
        print(f"Distribución de tipos:")
        from collections import Counter
        for t, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
            print(f"  {t:25s}: {cnt:,}")
    else:
        labels = ['Unknown'] * len(idx)
        print("Sin etiquetas de tipo — UMAP sin color por clase.")

    # ── 4. Bag-of-codes ────────────────────────────────────────────
    histograms = compute_bag_of_codes(tokens, num_codes=num_codes)
    np.save(output_dir / 'bag_of_codes.npy', histograms)
    print(f"Bag-of-codes guardado: {output_dir / 'bag_of_codes.npy'}")

    # ── 5. UMAP ────────────────────────────────────────────────────
    embedding = run_umap(
        histograms,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    np.save(output_dir / 'umap_embedding.npy', embedding)
    np.save(output_dir / 'umap_labels.npy',    np.array(labels))
    print(f"Embedding UMAP guardado: {output_dir / 'umap_embedding.npy'}")

    # ── 6. Plot ────────────────────────────────────────────────────
    plot_umap(
        embedding, labels,
        output_path=output_dir / 'umap_tokens.png',
    )

    print("\n✅ Listo! Resultados en:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_dir',  default='data/tokens',
                        help='Directorio con tokens.npy y snids.npy')
    parser.add_argument('--raw_dir',     default='data/lightcurves/elasticc_1/raw',
                        help='Directorio con los parquets originales')
    parser.add_argument('--output_dir',  default='results/umap',
                        help='Dónde guardar la figura y los arrays')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Máximo de imágenes a procesar (None = todas)')
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--min_dist',    type=float, default=0.1)
    args = parser.parse_args()
    main(args)
