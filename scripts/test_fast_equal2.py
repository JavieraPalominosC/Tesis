import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
snids = [s for s,_ in grouped if os.path.exists(f'{EXISTING_DIR}/{s}.png')][:5]

fig = plt.figure(figsize=(fig_params['figsize']))
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

for snid in snids:
    obj_df = grouped.get_group(snid)
    mjd_all=obj_df[col_mjd]; flux_all=obj_df[col_flux]
    mjd_min,mjd_max=mjd_all.min(),mjd_all.max()
    mjd_range=mjd_max-mjd_min if mjd_max>mjd_min else 1.0
    flux_min=flux_all.quantile(0.01); flux_max=flux_all.quantile(0.99)
    flux_range=flux_max-flux_min if flux_max>flux_min else 1.0
    ax.clear()
    for band_key,j in all_bands.items():
        bd_=obj_df[obj_df[col_band]==band_key]
        if not bd_.empty:
            bd=bd_.sort_values(col_mjd)
            mn=(bd[col_mjd]-mjd_min)/mjd_range
            fn=((bd[col_flux]-flux_min)/flux_range).clip(-0.05,1.15)
            ax.plot(mn,fn,'o-',color=fig_params['colors'][j],markersize=2,linewidth=1.0)
    ax.set_xlim([-0.05,1.05]); ax.set_ylim([-0.05,1.15]); ax.axis('off')
    rect=patches.Rectangle((0,0),1,1,linewidth=1.5,edgecolor='black',facecolor='none',transform=fig.transFigure)
    fig.add_artist(rect)
    buf=io.BytesIO(); fig.savefig(buf,format='png',pad_inches=0); buf.seek(0)
    img_fast=np.asarray(Image.open(buf).convert('RGB')); buf.close()
    rect.remove()
    img_ex=np.asarray(Image.open(f'{EXISTING_DIR}/{snid}.png').convert('RGB'))
    diff=np.abs(img_fast.astype(int)-img_ex.astype(int))
    print(f"{snid}: max_diff={diff.max()} mean_diff={diff.mean():.3f} pct={((diff>10).mean()*100):.3f}%", flush=True)
plt.close(fig)
