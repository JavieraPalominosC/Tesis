"""
Cuanta resolucion vertical tiene realmente una banda en la imagen.

El VQ-VAE convierte 256x256 px en 32x32 tokens, o sea 8x8 px por token.
Si al subir el flujo el pico se mueve menos de 8 px, el mapa de tokens no
cambia y el MaskGIT no puede notar nada, por buena que sea la codificacion.

Aca se mide, para varias codificaciones, cuanto se desplaza el pico de la
banda r (en px y en tokens) al subir df, y cuanto espacio vertical tiene
una banda en total.

Uso: python scripts/medir_resolucion_banda.py
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
TARGET_BAND = 'r'
MIN_OBS_BAND = 8
MIN_TAIL, TAIL_FRAC = 6, 0.30
DFS = [0, 10, 20, 50, 100, 200, 500, 1000]
STEP, AMP_LAYOUT = 2.0, 1.8
PX_POR_TOKEN = 8

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
fig_params = full_config['ft_classification']['imgs_params']['fig_params']
ds_cfg = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dc = ds_cfg['dict_columns']
all_bands = ds_cfg['all_bands']
col_mjd, col_flux, col_band = dc['mjd'], dc['flux'], dc['band']
N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())

Y_LO, Y_HI = -0.2, (N_BANDS - 1) * STEP + AMP_LAYOUT + 0.2
Y_SPAN = Y_HI - Y_LO


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def alto_px():
    fig = plt.figure(figsize=(fig_params['figsize']))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    from PIL import Image
    h = np.asarray(Image.open(buf).convert('L')).shape[0]
    buf.close()
    plt.close(fig)
    return h


# --- codificaciones -------------------------------------------------------
def enc_actual(f, obj_f):
    p01, p99 = np.quantile(obj_f, 0.01), np.quantile(obj_f, 0.99)
    rng = (p99 - p01) if p99 > p01 else 1.0
    return np.clip((f - p01) / rng, 0.0, 1.0)


def mk_asinh(beta, fmax):
    def enc(f, obj_f):
        u = np.arcsinh(f / beta) / np.arcsinh(fmax / beta)
        return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0
    return enc


ENCS = {
    'actual (lineal p01-p99 + clip)': enc_actual,
    'asinh fijo  FMAX=1e4': mk_asinh(4.0, 1.0e4),
    'asinh fijo  FMAX=1e3': mk_asinh(4.0, 1.0e3),
    'asinh fijo  FMAX=500': mk_asinh(4.0, 5.0e2),
}


def y_pico(obj_df, idx, enc, H):
    """Posicion del pico perturbado, en px desde arriba de la imagen."""
    obj_f = obj_df[col_flux].astype('float64').values
    f_pico = obj_df.loc[idx, col_flux].astype('float64').max()
    j = all_bands[TARGET_BAND]
    y = float(enc(np.array([f_pico]), obj_f)[0]) * AMP_LAYOUT + j * STEP
    return (Y_HI - y) / Y_SPAN * H


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = alto_px()
    lane_px = AMP_LAYOUT / Y_SPAN * H
    print(f'imagen {H}px de alto | una banda ocupa {lane_px:.0f} px '
          f'= {lane_px / PX_POR_TOKEN:.1f} tokens de {PX_POR_TOKEN}px', flush=True)

    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df['SNID'] = df['SNID'].astype(str)
    cand = []
    for s in df['SNID'].unique()[:20000]:
        sub = df[df['SNID'] == s]
        if all((sub[col_band] == b).sum() >= MIN_OBS_BAND for b in BAND_NAMES):
            cand.append(str(s))
        if len(cand) >= 40:
            break
    chosen = list(rng.choice(cand, size=12, replace=False))

    res = {k: [] for k in ENCS}
    for snid in chosen:
        obj = df[df['SNID'] == snid].copy()
        obj[col_flux] = obj[col_flux].astype('float64')
        scale = mad(obj.loc[obj[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(scale) or scale <= 0:
            continue
        sub = obj[obj[col_band] == TARGET_BAND].sort_values(col_mjd)
        t0, t1 = float(obj[col_mjd].min()), float(obj[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])
        for name, enc in ENCS.items():
            fila = []
            for d in DFS:
                p = obj.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * scale
                fila.append(y_pico(p, idx, enc, H))
            res[name].append(fila)

    print(f'\ndesplazamiento del pico respecto del control, en TOKENS '
          f'({PX_POR_TOKEN}px), mediana sobre {len(res[list(ENCS)[0]])} SNIa')
    print('  ' + 'codificacion'.ljust(32) + ''.join(f'df{d:<6}' for d in DFS[1:]))
    tabla = {}
    for name in ENCS:
        A = np.array(res[name])
        desp = (A[:, 0][:, None] - A[:, 1:]) / PX_POR_TOKEN
        med = np.median(desp, axis=0)
        tabla[name] = med.tolist()
        print('  ' + name.ljust(32) + ''.join(f'{v:<8.2f}' for v in med))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    cols = ['#d7191c', '#2c7bb6', '#4C9F70', '#F58518']
    for (name, med), c in zip(tabla.items(), cols):
        ax.plot(DFS[1:], med, 'o-', color=c, lw=2, label=name)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.text(DFS[1], 1.08, '1 token: por debajo de esto el MaskGIT no ve nada',
            fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.set_xticks(DFS[1:], [str(d) for d in DFS[1:]])
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_ylabel('desplazamiento del pico  [tokens]')
    ax.set_title(f'Cuanto sube el pico en el mapa de tokens\n'
                 f'una banda entera mide {lane_px / PX_POR_TOKEN:.1f} tokens, '
                 f'ese es el techo real', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'resolucion_banda.png', dpi=140)
    plt.close(fig)

    with open(OUT / 'resolucion_banda.json', 'w') as f:
        json.dump({'alto_px': H, 'banda_px': lane_px,
                   'banda_tokens': lane_px / PX_POR_TOKEN,
                   'dfs': DFS[1:], 'desplazamiento_tokens': tabla}, f, indent=2)
    print(f'\nFigura en {OUT}/resolucion_banda.png', flush=True)


if __name__ == '__main__':
    main()
