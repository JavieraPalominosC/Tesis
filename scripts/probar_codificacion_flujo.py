"""
Compara como codificar el flujo en la imagen para que la deteccion sea monotona.

Codificacion ACTUAL (waterfall_global):

    flux_norm = ((f - p01) / (p99 - p01)).clip(0, 1)

con p01/p99 del propio objeto. Tiene dos techos: el .clip(0,1) recorta todo
lo que pase de p99, y si p99 se recalcula sobre la curva perturbada el pico
queda siempre pegado al techo del panel. En los dos casos, a partir de cierto
df la imagen deja de cambiar.

Codificacion propuesta (asinh con escala FIJA para todo el dataset):

    u = asinh(f / beta) / asinh(FMAX / beta)      en [-1, 1]
    flux_norm = (u + 1) / 2                        en [0, 1]

asinh es lineal cerca de cero, asi que conserva el ruido y la estructura
debil, y logaritmica arriba, asi que comprime los picos sin recortarlos.
Como beta y FMAX son constantes del dataset:
  - un pico mas alto SIEMPRE queda mas arriba, sin techo
  - el resto de la curva no se aplana al agregar el pico
  - el flujo es comparable entre objetos

Mide, para cada codificacion, cuanto cambia la imagen respecto del control
a medida que sube df. La actual se aplana; la propuesta deberia seguir subiendo.

Uso: python scripts/probar_codificacion_flujo.py [N_OBJ]
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

N_OBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
TARGET_BAND = 'r'
MIN_OBS_BAND = 8
MIN_TAIL = 6
TAIL_FRAC = 0.30
DFS = [10, 20, 50, 100, 200, 500, 1000]
STEP, AMP_LAYOUT = 2.0, 1.8
UMBRAL_GRIS = 8

# Escala fija del dataset para asinh. beta ~ ruido tipico, FMAX ~ techo amplio.
BETA = 4.0
FMAX = 1.0e4

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
ds_cfg = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = ds_cfg['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = ds_cfg['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']
N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def enc_actual(f, obj_f):
    """Lineal con p01-p99 del objeto y recorte. Escala libre: p01/p99 se
    recalculan sobre la curva que se esta dibujando."""
    p01, p99 = np.quantile(obj_f, 0.01), np.quantile(obj_f, 0.99)
    rng = (p99 - p01) if p99 > p01 else 1.0
    return np.clip((f - p01) / rng, 0.0, 1.0)


def enc_asinh(f, obj_f):
    """asinh con escala fija del dataset. No depende del objeto ni recorta."""
    u = np.arcsinh(f / BETA) / np.arcsinh(FMAX / BETA)
    return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0


ENCS = {'actual': enc_actual, 'asinh_fijo': enc_asinh}


def render(obj_df, enc, out_path=None):
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    mjd_min = float(obj_df[col_mjd].min())
    mjd_rng = max(float(obj_df[col_mjd].max()) - mjd_min, 1.0)
    obj_f = obj_df[col_flux].astype('float64').values
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        x = (bd[col_mjd] - mjd_min) / mjd_rng
        y = enc(bd[col_flux].astype('float64').values, obj_f) * AMP_LAYOUT + j * STEP
        ax.plot(x, y, 'o-', color=fig_params['colors'][j], markersize=2, linewidth=1.0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS - 1) * STEP + AMP_LAYOUT + 0.2])
    ax.axis('off')
    fig.add_artist(patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                                     facecolor='none', transform=fig.transFigure))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    if out_path:
        im.save(out_path)
    arr = np.asarray(im.convert('L'), dtype=np.float32)
    buf.close()
    plt.close(fig)
    return arr


def tail_indices(obj_df):
    sub = obj_df[obj_df[col_band] == TARGET_BAND].sort_values(col_mjd)
    t0, t1 = float(obj_df[col_mjd].min()), float(obj_df[col_mjd].max())
    hit = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
    return hit if len(hit) >= MIN_TAIL else list(sub.index[-MIN_TAIL:])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df['SNID'] = df['SNID'].astype(str)

    cand = []
    for s in df['SNID'].unique()[:20000]:
        sub = df[df['SNID'] == s]
        if all((sub[col_band] == b).sum() >= MIN_OBS_BAND for b in BAND_NAMES):
            cand.append(str(s))
        if len(cand) >= N_OBJ * 2:
            break
    chosen = list(rng.choice(cand, size=min(N_OBJ, len(cand)), replace=False))
    print(f'{len(chosen)} objetos | beta={BETA} FMAX={FMAX:.0e} | df={DFS}', flush=True)

    res = {k: [] for k in ENCS}
    ejemplo = {}
    for n, snid in enumerate(chosen):
        obj = df[df['SNID'] == snid].copy()
        obj[col_flux] = obj[col_flux].astype('float64')
        scale = mad(obj.loc[obj[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(scale) or scale <= 0:
            continue
        idx = tail_indices(obj)
        for name, enc in ENCS.items():
            base = render(obj, enc)
            fila = []
            for d in DFS:
                pert = obj.copy()
                pert.loc[idx, col_flux] = pert.loc[idx, col_flux] + d * scale
                arr = render(pert, enc)
                fila.append(float((np.abs(arr - base) > UMBRAL_GRIS).mean() * 100))
                if n == 0 and d in (10, 100, 1000):
                    ejemplo[(name, d)] = pert.copy()
            res[name].append(fila)
        if n == 0:
            ejemplo[('control', 0)] = obj.copy()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    cols = {'actual': '#d7191c', 'asinh_fijo': '#2c7bb6'}
    etiq = {'actual': 'actual: lineal p01-p99 + clip',
            'asinh_fijo': f'propuesta: asinh fijo (beta={BETA:.0f})'}
    for name in ENCS:
        A = np.array(res[name])
        med = np.median(A, axis=0)
        ax.plot(DFS, med, 'o-', color=cols[name], lw=2, label=etiq[name])
        ax.fill_between(DFS, *np.percentile(A, [25, 75], axis=0),
                        color=cols[name], alpha=0.15)
        print(f'{name}: ' + '  '.join(f'df{d}={m:.2f}%' for d, m in zip(DFS, med)),
              flush=True)
    ax.set_xscale('log')
    ax.set_xticks(DFS, [str(d) for d in DFS])
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_ylabel('% de pixeles distintos del control')
    ax.set_title(f'Cuanto cambia la imagen al subir el pico\nmediana sobre {len(res["actual"])} SNIa',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # altura del pico en el panel, en unidades de la banda
    ff = np.logspace(0, 4, 200)
    ax2.plot(ff, np.clip((ff - 0) / 200.0, 0, 1), color='#d7191c', lw=2,
             label='actual (p99-p01 ~ 200): se satura en 1.0')
    ax2.plot(ff, (np.arcsinh(ff / BETA) / np.arcsinh(FMAX / BETA) + 1) / 2,
             color='#2c7bb6', lw=2, label='asinh fijo: nunca se satura')
    ax2.set_xscale('log')
    ax2.set_xlabel('flujo del punto  [FLUXCAL]')
    ax2.set_ylabel('altura dentro de la banda  [0-1]')
    ax2.set_title('La curva de codificacion\nel recorte es lo que rompe la monotonia',
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'codificacion_flujo.png', dpi=140)
    plt.close(fig)

    # ejemplos visuales
    if ejemplo:
        niveles = [('control', 0), ('actual', 10), ('actual', 100), ('actual', 1000),
                   ('asinh_fijo', 10), ('asinh_fijo', 100), ('asinh_fijo', 1000)]
        niveles = [k for k in niveles if k in ejemplo]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
        for a in axes.ravel():
            a.axis('off')
        for a, k in zip(axes.ravel(), niveles):
            name, d = k
            enc = ENCS.get(name, enc_actual)
            a.imshow(render(ejemplo[k], enc).astype(np.uint8), cmap='gray')
            a.set_title('control (actual)' if name == 'control'
                        else f'{etiq[name].split(":")[0]}  df={d}', fontsize=9)
            a.axis('off')
        fig.suptitle('Con la codificacion actual el pico deja de crecer; con asinh fijo sigue subiendo',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / 'codificacion_ejemplos.png', dpi=130)
        plt.close(fig)

    with open(OUT / 'codificacion.json', 'w') as f:
        json.dump({'beta': BETA, 'fmax': FMAX, 'dfs': DFS, 'umbral_gris': UMBRAL_GRIS,
                   'pct_pixeles': {k: np.median(np.array(v), axis=0).tolist()
                                   for k, v in res.items()}}, f, indent=2)
    print(f'Figuras en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
