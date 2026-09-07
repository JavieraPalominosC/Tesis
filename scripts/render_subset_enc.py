"""
Renderiza waterfall global con codificacion de flujo intercambiable, sobre un
subconjunto de objetos por clase.

Sirve para la comparacion pareja: mismas curvas, mismo layout, mismo tamano de
subconjunto, lo unico que cambia es como se mapea el flujo a la altura del pixel.

  actual    u = clip( (f - p01) / (p99 - p01), 0, 1 )
            El p99 lo fija el punto mas brillante, asi que un outlier define su
            propia escala y la curva entera se re-escala hasta que el pico deja
            de crecer.

  mad<U>    u = ( asinh( (f - mediana) / MAD ) / asinh(U) + 1 ) / 2
            Mediana y MAD del objeto, asi que los objetos debiles se siguen
            viendo. El MAD aguanta hasta 50% de contaminacion, asi que el pico
            no mueve la escala. asinh sin recorte, asi que mas flujo siempre es
            mas altura. U fija cuantos MAD entran en el carril: mas chico da mas
            resolucion para la forma normal, mas grande da mas techo para el
            outlier.

Uso: python scripts/render_subset_enc.py <parquet> <label_id> <enc> <n_por_clase> <out_root>
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import io
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

from scripts.enc_flujo import codificar, escala_de

fname = sys.argv[1]
label_id = int(sys.argv[2])
ENC = sys.argv[3]
N_PER_CLASS = int(sys.argv[4])
OUT_ROOT = sys.argv[5]

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = f'{OUT_ROOT}/images'
LBL_DIR = f'{OUT_ROOT}/labels'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

escala_de(ENC)  # valida el nombre antes de leer el parquet
SEED = 1234

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']

dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

N_BANDS = len(all_bands)
STEP = 2.0
AMP = 1.8


print(f'{fname} [{ENC}]: cargando parquet...', flush=True)
df = pd.read_parquet(os.path.join(RAW_DIR, fname))
df['SNID'] = df['SNID'].astype(str)

# Mismo subconjunto para las dos ramas: seed fija y orden estable.
snids = np.sort(df['SNID'].unique())
if len(snids) > N_PER_CLASS:
    snids = np.random.default_rng(SEED).choice(snids, size=N_PER_CLASS, replace=False)
snids = set(snids.tolist())
df = df[df['SNID'].isin(snids)]
grouped = df.groupby('SNID')
print(f'{fname} [{ENC}]: {len(grouped)} SNIDs seleccionados', flush=True)

fig = plt.figure(figsize=(fig_params['figsize']))
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

labels = {}
done = generated = 0

for snid, obj_df in tqdm(grouped, total=len(grouped)):
    out_path = f'{OUT_DIR}/{snid}.png'
    if os.path.exists(out_path):
        labels[str(snid)] = label_id
        done += 1
        continue
    try:
        mjd_all = obj_df[col_mjd]
        obj_flux = obj_df[col_flux].astype('float64').values
        mjd_min, mjd_max = mjd_all.min(), mjd_all.max()
        mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0

        ax.clear()
        for band_key, j in all_bands.items():
            band_data = obj_df[obj_df[col_band] == band_key]
            if not band_data.empty:
                bd = band_data.sort_values(col_mjd)
                mjd_norm = (bd[col_mjd] - mjd_min) / mjd_range
                y = codificar(bd[col_flux].values, obj_flux, ENC) * AMP + j * STEP
                ax.plot(mjd_norm, y, 'o-', color=fig_params['colors'][j],
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
        Image.open(buf).convert('RGB').save(out_path)
        buf.close()
        rect.remove()

        labels[str(snid)] = label_id
        generated += 1
    except Exception as e:
        print(f'  ERROR {snid}: {e}', flush=True)

plt.close(fig)

part = fname.replace('lc_', '').replace('.parquet', '')
with open(f'{LBL_DIR}/{part}.json', 'w') as f:
    json.dump(labels, f)
print(f'Listo {fname} [{ENC}]: {len(labels)} total ({done} existian, {generated} nuevas)', flush=True)
