import sys
sys.path.insert(0, '.')
import os
import glob
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

os.makedirs('data/folds', exist_ok=True)

# Cargar todas las imágenes y sus etiquetas (tipo de SN)
image_dir = 'data/images/elasticc_1/2grid'
paths = sorted(glob.glob(f'{image_dir}/*.png'))
print(f'Total imágenes: {len(paths):,}', flush=True)

# Extraer el SNID y mapear a tipo de SN
# Los SNIDs están en los parquets — cargamos el mapeo SNID → tipo
import pandas as pd

SN_FILES = [
    ('SNIa-SALT2',       'lc_SNIa-SALT2.parquet'),
    ('SNIa-91bg',        'lc_SNIa-91bg.parquet'),
    ('SNIax',            'lc_SNIax.parquet'),
    ('SNII-Templates',   'lc_SNII-Templates.parquet'),
    ('SNII-NMF',         'lc_SNII-NMF.parquet'),
    ('SNII+HostXT_V19',  'lc_SNII+HostXT_V19.parquet'),
    ('SNIb-Templates',   'lc_SNIb-Templates.parquet'),
    ('SNIb+HostXT_V19',  'lc_SNIb+HostXT_V19.parquet'),
    ('SNIc-Templates',   'lc_SNIc-Templates.parquet'),
    ('SNIc+HostXT_V19',  'lc_SNIc+HostXT_V19.parquet'),
    ('SNIcBL+HostXT_V19','lc_SNIcBL+HostXT_V19.parquet'),
    ('SNIIb+HostXT_V19', 'lc_SNIIb+HostXT_V19.parquet'),
    ('SNIIn-MOSFIT',     'lc_SNIIn-MOSFIT.parquet'),
    ('SNIIn+HostXT_V19', 'lc_SNIIn+HostXT_V19.parquet'),
    ('SLSN-I+host',      'lc_SLSN-I+host.parquet'),
    ('SLSN-I_no_host',   'lc_SLSN-I_no_host.parquet'),
    ('PISN',             'lc_PISN.parquet'),
]

print('Cargando mapeo SNID → tipo...', flush=True)
snid_to_type = {}
for tipo, fname in SN_FILES:
    df = pd.read_parquet(f'data/lightcurves/elasticc_1/raw/{fname}', columns=['SNID'])
    for snid in df['SNID'].unique():
        snid_to_type[str(snid)] = tipo
    print(f'  {tipo}: {df["SNID"].nunique():,} SNIDs', flush=True)

# Filtrar solo imágenes con SNID conocido
snids = [os.path.basename(p).replace('.png', '') for p in paths]
valid = [(p, snid_to_type[s]) for p, s in zip(paths, snids) if s in snid_to_type]
paths_valid = [v[0] for v in valid]
labels = [v[1] for v in valid]
print(f'\nImágenes con tipo conocido: {len(paths_valid):,}', flush=True)

# Crear 5 folds estratificados
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = {}
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths_valid, labels)):
    folds[fold_idx] = {
        'train': [paths_valid[i] for i in train_idx],
        'val':   [paths_valid[i] for i in val_idx],
    }
    train_types = [labels[i] for i in train_idx]
    val_types   = [labels[i] for i in val_idx]
    print(f'Fold {fold_idx}: train={len(train_idx):,} | val={len(val_idx):,}', flush=True)

# Guardar folds
with open('data/folds/folds.json', 'w') as f:
    json.dump(folds, f)

print('\nFolds guardados en data/folds/folds.json', flush=True)
