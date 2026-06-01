import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, '.')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import io
from PIL import Image
from omegaconf import OmegaConf
import os

os.makedirs('results/eda/figures', exist_ok=True)

dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml')['elasticc_1'])
dict_columns = dataset_config['dict_columns']
all_bands = dataset_config['all_bands']
colors = {0:'#9400D3', 1:'#00CC44', 2:'#FF2222', 3:'#FF8800', 4:'#2255FF', 5:'#00BBCC'}

def make_2grid(obj_df):
    mjd_min = obj_df[dict_columns['mjd']].min()
    mjd_max = obj_df[dict_columns['mjd']].max()
    flux_vals = obj_df[dict_columns['flux']]
    flux_min = flux_vals.quantile(0.01)
    flux_max = flux_vals.quantile(0.99)
    margin = (flux_max - flux_min) * 0.1

    fig, axs = plt.subplots(6, 1, figsize=(2.56, 2.56))
    for band_key, j in all_bands.items():
        band_data = obj_df[obj_df[dict_columns['band']] == band_key]
        if band_data.empty:
            axs[j].add_patch(patches.Rectangle((0,0),1,1,color='white',transform=axs[j].transAxes))
        else:
            axs[j].errorbar(
                band_data[dict_columns['mjd']],
                band_data[dict_columns['flux']],
                yerr=band_data[dict_columns['flux_err']],
                color=colors[j], fmt='-o', alpha=0.8, markersize=1.5, linewidth=0.5
            )
            axs[j].set_xlim([mjd_min, mjd_max])
            axs[j].set_ylim([flux_min - margin, flux_max + margin])
        axs[j].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def get_good_snid(df, min_obs=80):
    counts = df.groupby('SNID').size()
    good = counts[counts >= min_obs].index
    return good[len(good)//2] if len(good) > 0 else df['SNID'].unique()[0]

df_snia = pd.read_parquet('data/lightcurves/elasticc_1/raw/lc_SNIa-SALT2.parquet')
df_slsn = pd.read_parquet('data/lightcurves/elasticc_1/raw/lc_SLSN-I_no_host.parquet')

snid_snia = get_good_snid(df_snia)
snid_slsn = get_good_snid(df_slsn)

img_snia = make_2grid(df_snia[df_snia['SNID'] == snid_snia])
img_slsn = make_2grid(df_slsn[df_slsn['SNID'] == snid_slsn])

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(img_snia)
axes[0].set_title('SNIa-SALT2\n(supernova típica)', fontsize=13, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(img_slsn)
axes[1].set_title('SLSN-I sin host\n(candidata a outlier)', fontsize=13, fontweight='bold', color='darkred')
axes[1].axis('off')

plt.suptitle('Comparación: supernova típica vs candidata a outlier', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('results/eda/figures/comparacion_snia_slsn.png', dpi=150, bbox_inches='tight')
print('Listo!', flush=True)
