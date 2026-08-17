import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, io
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
EXISTING_DIR = 'data/images/elasticc_1/overlay'

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

df = pd.read_parquet(os.path.join(RAW_DIR, 'lc_SLSN-I_no_host.parquet'))
grouped = df.groupby('SNID')

# Tomar el primer SNID que exista en overlay
snid = None
for s, _ in grouped:
    if os.path.exists(f'{EXISTING_DIR}/{s}.png'):
        snid = s
        break
print(f"Usando SNID={snid} (tipo {type(snid)})", flush=True)

obj_df = grouped.get_group(snid)
mjd_all = obj_df[col_mjd]; flux_all = obj_df[col_flux]
mjd_min, mjd_max = mjd_all.min(), mjd_all.max()
mjd_range = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
flux_min = flux_all.quantile(0.01); flux_max = flux_all.quantile(0.99)
flux_range = flux_max - flux_min if flux_max > flux_min else 1.0

fig = plt.figure(figsize=(fig_params['figsize']))
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
for band_key, j in all_bands.items():
    band_data = obj_df[obj_df[col_band] == band_key]
    if not band_data.empty:
        bd = band_data.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - mjd_min) / mjd_range
        flux_norm = ((bd[col_flux] - flux_min) / flux_range).clip(-0.05, 1.15)
        ax.plot(mjd_norm, flux_norm, 'o-', color=fig_params['colors'][j], markersize=2, linewidth=1.0)
ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.15]); ax.axis('off')
buf = io.BytesIO()
fig.savefig(buf, format='png', pad_inches=0); buf.seek(0)
img_fast = np.asarray(Image.open(buf).convert('RGB'))
plt.close(fig)

img_existing = np.asarray(Image.open(f'{EXISTING_DIR}/{snid}.png').convert('RGB'))

figc, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img_existing); axes[0].set_title(f'Existente {img_existing.shape}'); axes[0].axis('off')
axes[1].imshow(img_fast); axes[1].set_title(f'Fast {img_fast.shape}'); axes[1].axis('off')
if img_fast.shape == img_existing.shape:
    diff = np.abs(img_fast.astype(int) - img_existing.astype(int)).mean(axis=2)
    im = axes[2].imshow(diff, cmap='hot'); axes[2].set_title('Diferencia'); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig('results/test_fast_visual.png', dpi=100, bbox_inches='tight')
print(f"existing shape: {img_existing.shape}, fast shape: {img_fast.shape}", flush=True)
print("Guardado results/test_fast_visual.png", flush=True)
