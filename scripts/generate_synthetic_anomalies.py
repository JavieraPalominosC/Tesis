"""
Anomalias sinteticas controladas, renderizadas con normalizacion por-banda (local)
y global, para comparar cual preserva la anomalia.

Correcciones respecto a la version anterior:
 - SNID es string en el parquet (dtype object)
 - pico/bache mas anchos (0.20*span) porque hay pocas observaciones por banda
 - amplitud definida como fraccion del rango p01-p99 de la banda (no multiplos de sigma)
 - imprime diagnostico de cuanto cambio realmente el flujo

Uso: python scripts/generate_synthetic_anomalies.py [N_BASE]
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, io, json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

N_BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_LOCAL  = 'data/images/elasticc_1/synthetic_local'
OUT_GLOBAL = 'data/images/elasticc_1/synthetic_global'
os.makedirs(OUT_LOCAL, exist_ok=True)
os.makedirs(OUT_GLOBAL, exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

N_BANDS = len(all_bands)
STEP_LOCAL,  AMP_LOCAL  = 1.15, 1.0
STEP_GLOBAL, AMP_GLOBAL = 2.0,  1.8

TARGET_BAND = 'r'
# amplitud como FRACCION del rango p01-p99 de la banda
AMPLITUDES = {'suave': 0.3, 'medio': 0.8, 'fuerte': 1.5}
WIDTH_FRAC = 0.20   # ancho del pico/bache como fraccion del span temporal


def inject(obj_df, kind, amp_factor, rng, verbose=False):
    df = obj_df.copy()
    df[col_flux] = df[col_flux].astype('float64')
    m = df[col_band] == TARGET_BAND
    if m.sum() < 3:
        return None
    sub = df.loc[m].sort_values(col_mjd)
    idx = sub.index.values
    t = sub[col_mjd].values
    f0 = sub[col_flux].values.astype(float)
    f = f0.copy()

    t0, t1 = t.min(), t.max()
    span = t1 - t0 if t1 > t0 else 1.0
    rng_flux = np.percentile(f0, 99) - np.percentile(f0, 1)
    if rng_flux <= 0:
        rng_flux = np.abs(np.mean(f0)) + 1.0
    A = amp_factor * rng_flux           # amplitud relativa al rango de la banda
    tc = t0 + 0.5*span
    w = WIDTH_FRAC*span

    if kind == 'pico':
        f = f + A*np.exp(-0.5*((t-tc)/w)**2)
    elif kind == 'bache':
        f = f - A*np.exp(-0.5*((t-tc)/w)**2)
    elif kind == 'shift':
        delta = 0.15*span
        f = np.interp(t, t + delta, f0)
    elif kind == 'ruido':
        win = np.abs(t-tc) < 1.5*w
        f[win] = f[win] + rng.normal(0, A, win.sum())
    elif kind == 'meseta':
        j = np.argmin(np.abs(t-tc)); after = t >= tc
        f[after] = f0[j]
    elif kind == 'escalon':
        after = t >= tc
        f[after] = f[after] + A
    else:
        raise ValueError(kind)

    if verbose:
        d = np.abs(f - f0)
        n_ch = (d > 1e-6).sum()
        print(f'    {kind}/{amp_factor}: {n_ch}/{len(f0)} puntos modificados, '
              f'cambio max = {d.max():.2f} ({d.max()/rng_flux:.2f}x rango banda)', flush=True)

    df.loc[idx, col_flux] = f
    return df


def render(obj_df, out_path, mode):
    mjd_all = obj_df[col_mjd]
    mjd_min, mjd_max = mjd_all.min(), mjd_all.max()
    mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0

    if mode == 'global':
        fa = obj_df[col_flux]
        g_min, g_max = fa.quantile(0.01), fa.quantile(0.99)
        g_range = g_max - g_min if g_max > g_min else 1.0
        STEP, AMP = STEP_GLOBAL, AMP_GLOBAL
    else:
        STEP, AMP = STEP_LOCAL, AMP_LOCAL

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - mjd_min) / mjd_range
        if mode == 'global':
            flux_norm = ((bd[col_flux] - g_min) / g_range).clip(0.0, 1.0)
        else:
            b_min = bd[col_flux].quantile(0.01); b_max = bd[col_flux].quantile(0.99)
            b_range = b_max - b_min if b_max > b_min else 1.0
            flux_norm = ((bd[col_flux] - b_min) / b_range).clip(0.0, 1.0)
        y = flux_norm * AMP + j * STEP
        ax.plot(mjd_norm, y, 'o-', color=fig_params['colors'][j], markersize=2, linewidth=1.0)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS-1)*STEP + AMP + 0.2])
    ax.axis('off')
    rect = patches.Rectangle((0,0),1,1, linewidth=1.5, edgecolor='black',
                             facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)
    buf = io.BytesIO(); fig.savefig(buf, format='png', pad_inches=0); buf.seek(0)
    Image.open(buf).convert('RGB').save(out_path); buf.close()
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    print(f'Cargando {BASE_FILE}...', flush=True)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    snids = df['SNID'].unique()          # strings
    # elegir objetos con al menos 8 observaciones en la banda objetivo
    cand = []
    for s in snids[:5000]:
        sub = df[df['SNID'] == s]
        if (sub[col_band] == TARGET_BAND).sum() >= 8:
            cand.append(s)
        if len(cand) >= N_BASE*4:
            break
    chosen = list(rng.choice(cand, size=min(N_BASE, len(cand)), replace=False))
    print(f'SNIDs base ({len(chosen)}): {chosen}', flush=True)

    manifest = []
    kinds = ['pico','bache','shift','ruido','meseta','escalon']

    for snid in chosen:
        obj = df[df['SNID'] == snid]
        n_r = (obj[col_band] == TARGET_BAND).sum()
        print(f'\n  SNID {snid}: {n_r} obs en banda {TARGET_BAND}', flush=True)
        for mode, outdir in (('local', OUT_LOCAL), ('global', OUT_GLOBAL)):
            p = f'{outdir}/{snid}__control.png'
            render(obj, p, mode)
            manifest.append({'snid': str(snid), 'kind':'control', 'amp':None, 'mode':mode, 'path':p})

        for kind in kinds:
            for amp_name, amp_val in AMPLITUDES.items():
                pert = inject(obj, kind, amp_val, rng, verbose=(amp_name=='fuerte'))
                if pert is None:
                    continue
                for mode, outdir in (('local', OUT_LOCAL), ('global', OUT_GLOBAL)):
                    p = f'{outdir}/{snid}__{kind}__{amp_name}.png'
                    render(pert, p, mode)
                    manifest.append({'snid':str(snid),'kind':kind,'amp':amp_name,'mode':mode,'path':p})

    with open('data/synthetic_manifest.json','w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nListo: {len(manifest)} imagenes', flush=True)


if __name__ == '__main__':
    main()
