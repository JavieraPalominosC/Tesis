import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import os
import json
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from src.data.processing.create_images import create_overlay_norm

fname = sys.argv[1]
label_id = int(sys.argv[2])

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = 'data/images/elasticc_1/overlay'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('data/images/elasticc_1/overlay_labels', exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']

df = pd.read_parquet(os.path.join(RAW_DIR, fname))
snids = df['SNID'].unique()  # TODOS, sin limite
print(f'{fname}: {len(snids)} SNIDs totales', flush=True)

labels = {}
done = 0
for snid in tqdm(snids):
    out_path = f'{OUT_DIR}/{snid}.png'
    if os.path.exists(out_path):
        labels[str(snid)] = label_id
        done += 1
        continue
    try:
        obj_df = df[df['SNID'] == snid]
        create_overlay_norm(obj_df, config, dataset_config).save(out_path)
        labels[str(snid)] = label_id
    except Exception as e:
        print(f'  ERROR {snid}: {e}', flush=True)

part = fname.replace('lc_', '').replace('.parquet', '')
with open(f'data/images/elasticc_1/overlay_labels/{part}.json', 'w') as f:
    json.dump(labels, f)
print(f'Listo {fname}: {len(labels)} imagenes ({done} ya existian)', flush=True)
