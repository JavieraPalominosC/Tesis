"""
Que estadistico usar para normalizar el flujo por objeto.

El problema de p01-p99: un solo punto por encima de p99 FIJA la escala. Si el
outlier define la escala, deja de ser outlier. Por eso al perturbar la cola la
curva entera se re-escala y el prior ve una supernova debil y plana normal.

El MAD no tiene ese problema: hace falta contaminar mas del 50% de los puntos
para moverlo. Asi que:

    u = asinh( (f - mediana) / (K * MAD) ) / asinh(U)

  - centro y escala del propio objeto, asi que los objetos debiles se siguen
    viendo (que es la razon por la que se normaliza por objeto)
  - MAD robusto, asi que el pico NO fija la escala
  - asinh sin recorte, asi que un pico mas alto siempre queda mas arriba

Este script mide, sobre curvas reales perturbadas en la cola:
  1) cuanto se mueve cada estadistico de escala al agregar el pico
  2) cuanto se desplaza el pico en el mapa de tokens con cada codificacion

Uso: python scripts/medir_robustez_escala.py [N_OBJ]
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

N_OBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = 42
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
BASE_FILE = 'lc_SNIa-SALT2.parquet'
OUT = os.environ.get('PROFE_FIG_OUT', 'results/figs_profe')
TARGET_BAND = 'r'
MIN_OBS_BAND, MIN_TAIL, TAIL_FRAC = 8, 6, 0.30
DFS = [10, 20, 50, 100, 200, 500, 1000]
STEP, AMP_LAYOUT, PX_TOK = 2.0, 1.8, 8
K_MAD, U_MAD = 3.0, 1000.0

ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dc = ds['dict_columns']
all_bands = ds['all_bands']
col_mjd, col_flux, col_band = dc['mjd'], dc['flux'], dc['band']
N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())
Y_LO, Y_HI = -0.2, (N_BANDS - 1) * STEP + AMP_LAYOUT + 0.2
Y_SPAN = Y_HI - Y_LO
H = 256


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def enc_actual(f, obj_f):
    p01, p99 = np.quantile(obj_f, 0.01), np.quantile(obj_f, 0.99)
    rng = (p99 - p01) if p99 > p01 else 1.0
    return np.clip((f - p01) / rng, 0.0, 1.0)


def enc_asinh_global(f, obj_f, beta=5.0, fmax=3e4):
    u = np.arcsinh(f / beta) / np.arcsinh(fmax / beta)
    return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0


def enc_asinh_mad(f, obj_f):
    """Robusto por objeto: centro mediana, escala K*MAD, asinh sin recorte."""
    med = np.median(obj_f)
    s = mad(obj_f)
    s = s if (np.isfinite(s) and s > 0) else 1.0
    u = np.arcsinh((f - med) / (K_MAD * s)) / np.arcsinh(U_MAD)
    return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0


ENCS = {'actual (p01-p99 + clip)': enc_actual,
        'asinh global fijo': enc_asinh_global,
        'asinh + MAD por objeto': enc_asinh_mad}


def y_px(f_val, obj_f, enc):
    j = all_bands[TARGET_BAND]
    y = float(enc(np.array([f_val]), obj_f)[0]) * AMP_LAYOUT + j * STEP
    return (Y_HI - y) / Y_SPAN * H


def main():
    os.makedirs(OUT, exist_ok=True)
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

    drift = {'p99': [], 'MAD': []}
    desp = {k: [] for k in ENCS}
    for snid in chosen:
        o = df[df['SNID'] == snid].copy()
        o[col_flux] = o[col_flux].astype('float64')
        sc = mad(o.loc[o[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(sc) or sc <= 0:
            continue
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        t0, t1 = float(o[col_mjd].min()), float(o[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])

        f0 = o[col_flux].values
        p99_0, mad_0 = np.quantile(f0, 0.99), mad(f0)
        fila_p, fila_m = [], []
        d_por_enc = {k: [] for k in ENCS}
        for d in DFS:
            p = o.copy()
            p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
            f1 = p[col_flux].values
            fila_p.append(np.quantile(f1, 0.99) / max(p99_0, 1e-6))
            fila_m.append(mad(f1) / max(mad_0, 1e-6))
            f_pico = p.loc[idx, col_flux].max()
            for k, enc in ENCS.items():
                d_por_enc[k].append((y_px(o.loc[idx, col_flux].max(), f0, enc)
                                     - y_px(f_pico, f1, enc)) / PX_TOK)
        drift['p99'].append(fila_p)
        drift['MAD'].append(fila_m)
        for k in ENCS:
            desp[k].append(d_por_enc[k])

    print(f'{len(drift["p99"])} objetos\n')
    print('CUANTO SE MUEVE LA ESCALA al agregar el pico (factor vs control)')
    print('  df      ' + ''.join(f'{d:<9}' for d in DFS))
    for k in ('p99', 'MAD'):
        m = np.median(np.array(drift[k]), axis=0)
        print(f'  {k:<8}' + ''.join(f'{v:<9.1f}' for v in m))

    print('\nDESPLAZAMIENTO DEL PICO en tokens')
    print('  ' + 'codificacion'.ljust(26) + ''.join(f'{d:<9}' for d in DFS))
    tabla = {}
    for k in ENCS:
        m = np.median(np.array(desp[k]), axis=0)
        tabla[k] = m.tolist()
        print('  ' + k.ljust(26) + ''.join(f'{v:<9.2f}' for v in m))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.9))
    for k, c in zip(('p99', 'MAD'), ('#d7191c', '#2c7bb6')):
        ax.plot(DFS, np.median(np.array(drift[k]), axis=0), 'o-', color=c, lw=2,
                label=f'{k} de la curva perturbada')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(DFS, [str(d) for d in DFS])
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_ylabel('escala perturbada / escala del control')
    ax.set_title('El pico arrastra a p99 pero casi no mueve al MAD\n'
                 'por eso con p01-p99 el outlier fija su propia escala', fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    for (k, m), c in zip(tabla.items(), ('#d7191c', '#4C9F70', '#2c7bb6')):
        ax2.plot(DFS, m, 'o-', color=c, lw=2, label=k)
    ax2.axhline(1.0, color='gray', ls='--', lw=1)
    ax2.text(DFS[0], 1.05, '1 token', fontsize=8, color='gray')
    ax2.set_xscale('log')
    ax2.set_xticks(DFS, [str(d) for d in DFS])
    ax2.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax2.set_ylabel('desplazamiento del pico  [tokens]')
    ax2.set_title(f'Una banda mide {AMP_LAYOUT / Y_SPAN * H / PX_TOK:.1f} tokens',
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'robustez_escala.png'), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT, 'robustez_escala.json'), 'w') as f:
        json.dump({'dfs': DFS, 'K_MAD': K_MAD, 'U_MAD': U_MAD,
                   'drift': {k: np.median(np.array(v), axis=0).tolist()
                             for k, v in drift.items()},
                   'desplazamiento_tokens': tabla}, f, indent=2)
    print(f'\nFigura en {OUT}/robustez_escala.png')


if __name__ == '__main__':
    main()
