import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import io
import json
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

fname = sys.argv[1]
label_id = int(sys.argv[2])

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = 'data/images/elasticc_1/overlay'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('data/images/elasticc_1/overlay_labels', exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']

dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

print(f'{fname}: cargando parquet...', flush=True)
df = pd.read_parquet(os.path.join(RAW_DIR, fname))
print(f'{fname}: agrupando por SNID...', flush=True)
grouped = df.groupby('SNID')
total = len(grouped)
print(f'{fname}: {total} SNIDs totales', flush=True)

# Figura reutilizada
fig = plt.figure(figsize=(fig_params['figsize']))
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

labels = {}
done = 0
generated = 0

for snid, obj_df in tqdm(grouped, total=total):
    out_path = f'{OUT_DIR}/{snid}.png'
    if os.path.exists(out_path):
        labels[str(snid)] = label_id
        done += 1
        continue

    try:
        mjd_all = obj_df[col_mjd]; flux_all = obj_df[col_flux]
        mjd_min, mjd_max = mjd_all.min(), mjd_all.max()
        mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
        flux_min = flux_all.quantile(0.01); flux_max = flux_all.quantile(0.99)
        flux_range = flux_max - flux_min if flux_max > flux_min else 1.0

        ax.clear()
        for band_key, j in all_bands.items():
            band_data = obj_df[obj_df[col_band] == band_key]
            if not band_data.empty:
                bd = band_data.sort_values(col_mjd)
                mjd_norm = (bd[col_mjd] - mjd_min) / mjd_range
                flux_norm = ((bd[col_flux] - flux_min) / flux_range).clip(-0.05, 1.15)
                ax.plot(mjd_norm, flux_norm, 'o-',
                        color=fig_params['colors'][j],
                        markersize=2, linewidth=1.0)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.15])
        ax.axis('off')

        # Borde identico al original
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5,
                                 edgecolor='black', facecolor='none',
                                 transform=fig.transFigure)
        fig.add_artist(rect)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        Image.open(buf).convert('RGB').save(out_path)
        buf.close()

        rect.remove()  # quitar el borde para la siguiente iteracion

        labels[str(snid)] = label_id
        generated += 1
    except Exception as e:
        print(f'  ERROR {snid}: {e}', flush=True)

plt.close(fig)

part = fname.replace('lc_', '').replace('.parquet', '')
with open(f'data/images/elasticc_1/overlay_labels/{part}.json', 'w') as f:
    json.dump(labels, f)
print(f'Listo {fname}: {len(labels)} total ({done} existian, {generated} nuevas)', flush=True)
