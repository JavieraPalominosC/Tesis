import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import os
import json
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from src.data.processing.create_images import create_2grid_filled, create_overlay_norm

N_PER_TYPE = 1750
fname = sys.argv[1]
label_id = int(sys.argv[2])

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_FILLED = 'data/images_exp/filled'
OUT_OVERLAY = 'data/images_exp/overlay'
os.makedirs(OUT_FILLED, exist_ok=True)
os.makedirs(OUT_OVERLAY, exist_ok=True)
os.makedirs('data/images_exp/labels_parts', exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']

df = pd.read_parquet(os.path.join(RAW_DIR, fname))
snids = df['SNID'].unique()[:N_PER_TYPE]
print(f'{fname}: {len(snids)} imagenes', flush=True)

labels = {}
for snid in tqdm(snids):
    obj_df = df[df['SNID'] == snid]
    f_path = f'{OUT_FILLED}/{snid}.png'
    o_path = f'{OUT_OVERLAY}/{snid}.png'
    if os.path.exists(f_path) and os.path.exists(o_path):
        labels[str(snid)] = label_id
        continue
    try:
        create_2grid_filled(obj_df, config, dataset_config).save(f_path)
        create_overlay_norm(obj_df, config, dataset_config).save(o_path)
        labels[str(snid)] = label_id
    except Exception as e:
        print(f'  ERROR {snid}: {e}', flush=True)

part = fname.replace('lc_', '').replace('.parquet', '')
with open(f'data/images_exp/labels_parts/{part}.json', 'w') as f:
    json.dump(labels, f)
print(f'Listo {fname}: {len(labels)}', flush=True)
