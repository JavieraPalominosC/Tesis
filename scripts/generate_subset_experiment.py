"""
Genera un subconjunto balanceado de imagenes con dos representaciones nuevas
(relleno apilado y superpuesto normalizado) para el experimento de prototipo.

Salida:
  data/images_exp/filled/{SNID}.png
  data/images_exp/overlay/{SNID}.png
  data/images_exp/labels_subset.json
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')

import os
import json
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.processing.create_images import (
    create_2grid_filled, create_overlay_norm
)

N_PER_TYPE = 1750

SN_FILES = [
    ('SNIa-SALT2',        'lc_SNIa-SALT2.parquet',        0),
    ('SNIa-91bg',         'lc_SNIa-91bg.parquet',         1),
    ('SNIax',             'lc_SNIax.parquet',             2),
    ('SNII-Templates',    'lc_SNII-Templates.parquet',    3),
    ('SNII-NMF',          'lc_SNII-NMF.parquet',          4),
    ('SNII+HostXT_V19',   'lc_SNII+HostXT_V19.parquet',   5),
    ('SNIb-Templates',    'lc_SNIb-Templates.parquet',    6),
    ('SNIb+HostXT_V19',   'lc_SNIb+HostXT_V19.parquet',   7),
    ('SNIc-Templates',    'lc_SNIc-Templates.parquet',    8),
    ('SNIc+HostXT_V19',   'lc_SNIc+HostXT_V19.parquet',   9),
    ('SNIcBL+HostXT_V19', 'lc_SNIcBL+HostXT_V19.parquet', 10),
    ('SNIIb+HostXT_V19',  'lc_SNIIb+HostXT_V19.parquet',  11),
    ('SNIIn-MOSFIT',      'lc_SNIIn-MOSFIT.parquet',      12),
    ('SNIIn+HostXT_V19',  'lc_SNIIn+HostXT_V19.parquet',  13),
    ('SLSN-I+host',       'lc_SLSN-I+host.parquet',       14),
    ('SLSN-I_no_host',    'lc_SLSN-I_no_host.parquet',    15),
    ('PISN',              'lc_PISN.parquet',              16),
]

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_FILLED = 'data/images_exp/filled'
OUT_OVERLAY = 'data/images_exp/overlay'
os.makedirs(OUT_FILLED, exist_ok=True)
os.makedirs(OUT_OVERLAY, exist_ok=True)

full_config = OmegaConf.to_container(
    OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(
    OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']

labels = {}

for tipo, fname, label_id in SN_FILES:
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        print(f'SKIP {tipo}: no existe {path}', flush=True)
        continue
    df = pd.read_parquet(path)
    snids = df['SNID'].unique()[:N_PER_TYPE]
    print(f'{tipo}: generando {len(snids)} imagenes...', flush=True)

    for snid in tqdm(snids, desc=tipo):
        obj_df = df[df['SNID'] == snid]

        f_path = f'{OUT_FILLED}/{snid}.png'
        o_path = f'{OUT_OVERLAY}/{snid}.png'
        if os.path.exists(f_path) and os.path.exists(o_path):
            labels[str(snid)] = label_id
            continue

        try:
            img_f = create_2grid_filled(obj_df, config, dataset_config)
            img_f.save(f_path)
            img_o = create_overlay_norm(obj_df, config, dataset_config)
            img_o.save(o_path)
            labels[str(snid)] = label_id
        except Exception as e:
            print(f'  ERROR {snid}: {e}', flush=True)

with open('data/images_exp/labels_subset.json', 'w') as f:
    json.dump(labels, f)

print(f'\nListo. {len(labels)} imagenes por representacion.', flush=True)
print(f'  Filled:  {OUT_FILLED}', flush=True)
print(f'  Overlay: {OUT_OVERLAY}', flush=True)
