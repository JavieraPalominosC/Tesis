"""
Inserta observaciones NUEVAS (no modifica las existentes) en el centro temporal
de la banda objetivo, con flujo muy por encima del rango de la curva.

Garantiza que el pulso exista aunque no haya observaciones cerca del centro.
Barrido de amplitudes para encontrar el umbral de deteccion.

Uso: python scripts/generate_synthetic_insert.py [N_BASE]
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

N_BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = 'data/images/elasticc_1/synthetic_insert'
os.makedirs(OUT_DIR, exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

N_BANDS = len(all_bands)
STEP, AMP_LAYOUT = 1.15, 1.0

TARGET_BAND = 'r'
MIN_OBS_BAND = 15
N_INSERT = 3                      # puntos nuevos consecutivos
AMPLITUDES = [2.0, 5.0, 15.0]     # multiplos del rango p01-p99 de la banda
DT_FRAC = 0.01                    # separacion entre los puntos insertados


def insert_pulse(obj_df, amp_factor):
    """Agrega N_INSERT filas nuevas en el centro temporal de TARGET_BAND."""
    df = obj_df.copy()
    df[col_flux] = df[col_flux].astype('float64')
    sub = df[df[col_band] == TARGET_BAND].sort_values(col_mjd)
    if len(sub) < 3:
        return None, 0, 0.0

    t = sub[col_mjd].values.astype(float)
    f0 = sub[col_flux].values.astype(float)
    t0, t1 = t.min(), t.max()
    span = t1 - t0 if t1 > t0 else 1.0
    rng_flux = np.percentile(f0, 99) - np.percentile(f0, 1)
    if rng_flux <= 0:
        rng_flux = abs(np.mean(f0)) + 1.0

    base = float(np.median(f0))
    A = amp_factor * rng_flux
    tc = t0 + 0.5*span
    dt = DT_FRAC * span

    plantilla = sub.iloc[0].copy()      # copia una fila real para heredar columnas
    nuevas = []
    offsets = np.linspace(-(N_INSERT//2), N_INSERT//2, N_INSERT)
    for k, off in enumerate(offsets):
        fila = plantilla.copy()
        fila[col_mjd] = tc + off*dt
        # forma de pulso: el del medio es el mas alto
        peso = np.exp(-0.5*(off/0.8)**2)
        fila[col_flux] = base + A*peso
        nuevas.append(fila)

    df_new = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)
    return df_new, N_INSERT, float(A/rng_flux)


def render(obj_df, out_path):
    mjd_all = obj_df[col_mjd]
    mjd_min, mjd_max = mjd_all.min(), mjd_all.max()
    mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty: continue
        bd = bd.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - mjd_min) / mjd_range
        b_min = bd[col_flux].quantile(0.01); b_max = bd[col_flux].quantile(0.99)
        b_range = b_max - b_min if b_max > b_min else 1.0
        flux_norm = ((bd[col_flux] - b_min) / b_range).clip(0.0, 1.0)
        y = flux_norm * AMP_LAYOUT + j * STEP
        ax.plot(mjd_norm, y, 'o-', color=fig_params['colors'][j], markersize=2, linewidth=1.0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS-1)*STEP + AMP_LAYOUT + 0.2])
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
    snids = df['SNID'].unique()
    cand = []
    for s in snids[:20000]:
        sub = df[df['SNID'] == s]
        if (sub[col_band] == TARGET_BAND).sum() >= MIN_OBS_BAND:
            cand.append(s)
        if len(cand) >= N_BASE*3: break
    chosen = list(rng.choice(cand, size=min(N_BASE, len(cand)), replace=False))
    print(f'SNIDs elegidos ({len(chosen)}): {chosen}', flush=True)

    manifest = []
    for snid in chosen:
        obj = df[df['SNID'] == snid]
        n_r = int((obj[col_band] == TARGET_BAND).sum())
        print(f'\n  SNID {snid}: {n_r} obs en banda {TARGET_BAND}', flush=True)
        p = f'{OUT_DIR}/{snid}__control.png'
        render(obj, p)
        manifest.append({'snid':str(snid),'amp':None,'path':p})
        for a in AMPLITUDES:
            pert, n_ins, ratio = insert_pulse(obj, a)
            if pert is None: continue
            p = f'{OUT_DIR}/{snid}__amp{int(a)}.png'
            render(pert, p)
            print(f'    amp={a:.0f}x rango: {n_ins} puntos insertados, '
                  f'altura del pulso = {ratio:.1f}x rango banda', flush=True)
            manifest.append({'snid':str(snid),'amp':a,'path':p})

    with open('data/synthetic_insert_manifest.json','w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nListo: {len(manifest)} imagenes en {OUT_DIR}/', flush=True)


if __name__ == '__main__':
    main()
