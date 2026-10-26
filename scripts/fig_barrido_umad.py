"""Ejemplos de imagenes 2grid variando U, el numero de MAD que entran en el carril.

Pedido en la reunion del 28 de septiembre: ver como cambia la imagen normalizada
al mover U logaritmicamente entre 5 y 300. La decision original se tomo
comparando solo U = 100 y U = 300, asi que este barrido cubre el rango completo.

U gobierna el compromiso de la codificacion:
  U chico  el carril se aprovecha y la morfologia normal se ve con detalle,
           pero se satura antes y un outlier grande se apoya en el techo
  U grande hay margen para el outlier, pero la curva normal se aplana contra
           la linea media y se pierde resolucion en la forma

Produce:
  umad_ejemplos.png   grilla de imagenes, una fila por U
  umad_curvas.png     funcion de transferencia y uso efectivo del carril

Uso: python scripts/fig_barrido_umad.py
"""
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

import sys
sys.path.insert(0, '.')
from scripts.enc_flujo import codificar

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
SALIDA = Path(os.environ.get('SALIDA_FIG', 'results/figs_profe'))

# Siete valores espaciados logaritmicamente entre 5 y 300.
U_VALS = np.logspace(np.log10(5), np.log10(300), 7)

# Un objeto por tipo, cubriendo morfologias distintas. PISN y SLSN estan entre
# las clases que el profe pidio excluir del entrenamiento, asi que sirve ver
# como se ven bajo cada U.
EJEMPLOS = [
    ('lc_SNIa-SALT2.parquet', 'SNIa-SALT2'),
    ('lc_SNII-NMF.parquet', 'SNII-NMF'),
    ('lc_SLSN-I+host.parquet', 'SLSN-I+host'),
    ('lc_PISN.parquet', 'PISN'),
]

N_BANDS, STEP, AMP = 6, 2.0, 1.8
BANDAS = ['Y', 'z', 'i', 'r', 'g', 'u']


def cargar_config():
    cfg = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'),
                                 resolve=True)['ft_classification']
    ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'),
                                resolve=True)['elasticc_1']
    return cfg['imgs_params']['fig_params'], ds


def render(obj_df, cols, bands, fig_params, u_max):
    """Dibuja un objeto con la codificacion mad<u_max>, misma geometria que el
    renderizador de entrenamiento."""
    col_mjd, col_flux, col_band = cols
    obj_flux = obj_df[col_flux].astype('float64').values
    mjd_min, mjd_max = obj_df[col_mjd].min(), obj_df[col_mjd].max()
    rng = mjd_max - mjd_min if mjd_max > mjd_min else 1.0

    fig = plt.figure(figsize=fig_params['figsize'])
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    us = []
    for band_key, j in bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        u = codificar(bd[col_flux].values, obj_flux, f'mad{u_max}')
        us.append(u)
        ax.plot((bd[col_mjd] - mjd_min) / rng, u * AMP + j * STEP, 'o-',
                color=fig_params['colors'][j], markersize=2, linewidth=1.0)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS - 1) * STEP + AMP + 0.2])
    ax.axis('off')
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                             facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB'))
    buf.close()

    todos = np.concatenate(us) if us else np.array([0.5])
    return img, todos


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig_params, ds = cargar_config()
    dc = ds['dict_columns']
    cols = (dc['mjd'], dc['flux'], dc['band'])
    bands = ds['all_bands']

    objetos = []
    for fname, nombre in EJEMPLOS:
        df = pd.read_parquet(os.path.join(RAW_DIR, fname))
        df['SNID'] = df['SNID'].astype(str)
        # Se toma el primer SNID con fotometria en las seis bandas, para que la
        # comparacion entre valores de U no dependa de que falte una banda.
        elegido = None
        for snid, g in df.groupby('SNID'):
            if g[dc['band']].nunique() == len(bands) and len(g) > 50:
                elegido = (snid, g)
                break
        if elegido is None:
            snid, g = next(iter(df.groupby('SNID')))
            elegido = (snid, g)
        objetos.append((nombre, elegido[0], elegido[1]))
        print(f'{nombre}: SNID {elegido[0]}, {len(elegido[1])} puntos', flush=True)

    nU, nO = len(U_VALS), len(objetos)
    fig, axes = plt.subplots(nU, nO, figsize=(2.35 * nO, 2.42 * nU))
    uso = {nombre: [] for nombre, _, _ in objetos}

    for i, u_max in enumerate(U_VALS):
        for j, (nombre, snid, g) in enumerate(objetos):
            img, us = render(g, cols, bands, fig_params, u_max)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            # Rango intercuartil de u: cuanto del carril ocupa realmente la curva.
            uso[nombre].append(float(np.percentile(us, 95) - np.percentile(us, 5)))
            if i == 0:
                ax.set_title(f'{nombre}\nSNID {snid}', fontsize=9, pad=8)
            if j == 0:
                ax.set_ylabel(f'U = {u_max:.0f}', fontsize=11, fontweight='bold')

    paso = 256 / len(BANDAS)
    axes[0, 0].set_yticks([(k + 0.5) * paso for k in range(len(BANDAS))])
    axes[0, 0].set_yticklabels(BANDAS, fontsize=7)

    fig.suptitle('Imagenes 2grid variando $U$, el numero de MAD que entran en el carril de cada banda.\n'
                 'Al subir $U$ la curva se aplana contra la linea media, que es el precio de dejar '
                 'margen para que un outlier grande no sature.', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(SALIDA / 'umad_ejemplos.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Segunda figura: la funcion de transferencia y el uso efectivo del carril.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.linspace(-400, 400, 4000)
    cmap = plt.get_cmap('viridis')
    for k, u_max in enumerate(U_VALS):
        y = np.clip(np.arcsinh(x) / np.arcsinh(u_max), -1, 1) * 0.5 + 0.5
        axes[0].plot(x, y, color=cmap(k / (len(U_VALS) - 1)), lw=1.8,
                     label=f'U = {u_max:.0f}')
    axes[0].set_xscale('symlog', linthresh=1)
    axes[0].axhline(1, color='gray', ls=':', lw=0.9)
    axes[0].axhline(0, color='gray', ls=':', lw=0.9)
    axes[0].set_xlabel('desviacion respecto de la mediana, en MAD')
    axes[0].set_ylabel('altura normalizada $u$ dentro del carril')
    axes[0].set_title('A. Funcion de transferencia de la codificacion', fontsize=10)
    axes[0].legend(fontsize=7.5, ncol=2)
    axes[0].grid(alpha=0.25)

    for nombre, vals in uso.items():
        axes[1].plot(U_VALS, vals, '-o', ms=4, lw=1.5, label=nombre)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$U$')
    axes[1].set_ylabel('rango del 90% central de $u$')
    axes[1].set_title('B. Fraccion del carril que ocupa la curva normal', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle('Compromiso que gobierna $U$: mas margen para el outlier cuesta resolucion '
                 'en la morfologia normal.', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    fig.savefig(SALIDA / 'umad_curvas.png', dpi=150, bbox_inches='tight')

    with open(SALIDA / 'umad_resumen.json', 'w') as f:
        json.dump({'U': [float(u) for u in U_VALS], 'uso_carril': uso}, f, indent=1)

    print('\nfiguras en', SALIDA)
    print(f'{"U":>6} ' + ' '.join(f'{n[:12]:>13s}' for n in uso))
    for i, u_max in enumerate(U_VALS):
        print(f'{u_max:6.0f} ' + ' '.join(f'{uso[n][i]:13.3f}' for n in uso))


if __name__ == '__main__':
    main()
