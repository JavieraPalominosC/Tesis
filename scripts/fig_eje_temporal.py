"""Comparacion visual de los dos ejes temporales, antes de reentrenar.

Arriba el renderizado actual, que normaliza el tiempo por el span de cada objeto
y hace que toda curva ocupe el ancho completo. Abajo el eje absoluto, con una
escala comun de T_ABS dias para todos, donde el ancho que ocupa la curva es su
duracion.

Sirve para ver el precio antes de gastar el computo: con escala comun las curvas
cortas se comprimen, y la pregunta es si queda forma suficiente para que el
VQ-VAE la represente.

Uso: python scripts/fig_eje_temporal.py [T_ABS]
"""
import sys
sys.path.insert(0, '.')
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

from scripts.enc_flujo import codificar

T_ABS = float(sys.argv[1]) if len(sys.argv) > 1 else 1110.0
ENC = 'mad300'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT = Path('imagenes')

# Un caso tipico, el caso peor de compresion, y las dos clases excluidas donde
# la duracion es la variable que mas separa.
CASOS = [
    ('lc_SNIa-SALT2.parquet', 532.0, 'SNIa-SALT2, duracion mediana'),
    ('lc_SNIa-SALT2.parquet', 170.0, 'SNIa-SALT2, decil mas corto'),
    ('lc_SLSN-I_no_host.parquet', 895.0, 'SLSN-I sin host (excluida)'),
    ('lc_PISN.parquet', 558.0, 'PISN (excluida)'),
]

cfg = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
fig_params = cfg['ft_classification']['imgs_params']['fig_params']
ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dc = ds['dict_columns']
all_bands = ds['all_bands']
col_mjd, col_flux, col_band = dc['mjd'], dc['flux'], dc['band']
N_BANDS, STEP, AMP = len(all_bands), 2.0, 1.8


def render(obj_df, t_abs, fig, ax):
    """Replica exacta de render_subset_enc.py, con el eje temporal como opcion."""
    obj_flux = obj_df[col_flux].astype('float64').values
    mjd_min, mjd_max = obj_df[col_mjd].min(), obj_df[col_mjd].max()
    if t_abs > 0:
        rango = t_abs
    else:
        rango = mjd_max - mjd_min if mjd_max > mjd_min else 1.0

    ax.clear()
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        y = codificar(bd[col_flux].values, obj_flux, ENC) * AMP + j * STEP
        ax.plot((bd[col_mjd] - mjd_min) / rango, y, 'o-', color=fig_params['colors'][j],
                markersize=2, linewidth=1.0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS - 1) * STEP + AMP + 0.2])
    ax.axis('off')
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                             facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((256, 256)).copy()
    buf.close()
    rect.remove()
    return img


def elegir(fname, span_objetivo):
    """El objeto con las seis bandas cuyo span esta mas cerca del objetivo."""
    df = pd.read_parquet(os.path.join(RAW_DIR, fname))
    df['SNID'] = df['SNID'].astype(str)
    g = df.groupby('SNID')
    span = (g[col_mjd].max() - g[col_mjd].min())
    bandas = g[col_band].nunique()
    cand = span[(bandas == N_BANDS) & (span > 0)]
    snid = (cand - span_objetivo).abs().idxmin()
    return snid, float(cand[snid]), df[df['SNID'] == snid].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chica = plt.figure(figsize=(fig_params['figsize']))
    ax = chica.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig, axes = plt.subplots(2, len(CASOS), figsize=(3.1 * len(CASOS), 6.6))
    resumen = {'T_ABS': T_ABS, 'casos': []}

    for k, (fname, objetivo, titulo) in enumerate(CASOS):
        snid, span, obj = elegir(fname, objetivo)
        frac = min(span / T_ABS, 1.0)
        print(f'{titulo:34s} snid={snid:<12s} span={span:7.1f} d  '
              f'ancho={100 * frac:5.1f}%  ({256 * frac / 8:.1f} tokens de 32)', flush=True)

        for fila, t_abs in ((0, 0.0), (1, T_ABS)):
            a = axes[fila, k]
            a.imshow(np.asarray(render(obj, t_abs, chica, ax)))
            a.set_xticks([])
            a.set_yticks([])
            if fila == 0:
                a.set_title(f'{titulo}\n{span:.0f} d', fontsize=9)
            else:
                a.set_xlabel(f'ocupa {100 * frac:.0f}% del ancho', fontsize=8)

        resumen['casos'].append({'titulo': titulo, 'snid': snid, 'span_d': round(span, 1),
                                 'frac_ancho': round(frac, 3),
                                 'tokens_ancho': round(32 * frac, 1)})

    axes[0, 0].set_ylabel('eje por objeto\n(actual)', fontsize=10)
    axes[1, 0].set_ylabel(f'eje absoluto\n(T = {T_ABS:g} d)', fontsize=10)
    fig.suptitle('Eje temporal: normalizado por objeto contra escala comun', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'eje_temporal.png', dpi=150)
    plt.close(fig)
    plt.close(chica)

    with open(OUT / 'eje_temporal.json', 'w') as f:
        json.dump(resumen, f, indent=1)
    print(f'\nListo: {OUT / "eje_temporal.png"}', flush=True)


if __name__ == '__main__':
    main()
