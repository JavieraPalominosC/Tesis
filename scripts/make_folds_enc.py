"""Folds y labels para una rama de la comparacion de codificacion.

El sufijo permite tener varias corridas sobre el mismo directorio de imagenes
sin pisarse: el subconjunto de 196k escribe folds.json y la corrida completa de
1.39M escribe folds_full.json, apuntando ambas al mismo images/.

Uso: python scripts/make_folds_enc.py <out_root> [sufijo]
"""
import sys
import json
import glob
import random

random.seed(42)
OUT_ROOT = sys.argv[1]
SUF = f'_{sys.argv[2]}' if len(sys.argv) > 2 else ''

paths = sorted(glob.glob(f'{OUT_ROOT}/images/*.png'))
random.shuffle(paths)
n_val = int(len(paths) * 0.1)
folds = {"0": {"train": paths[n_val:], "val": paths[:n_val]}}
with open(f'{OUT_ROOT}/folds{SUF}.json', 'w') as f:
    json.dump(folds, f)
print(f'{OUT_ROOT} [{SUF or "subset"}]: {len(paths) - n_val:,} train, {n_val:,} val')

labels = {}
for p in glob.glob(f'{OUT_ROOT}/labels/*.json'):
    with open(p) as f:
        labels.update(json.load(f))
with open(f'{OUT_ROOT}/labels.json', 'w') as f:
    json.dump(labels, f)
print(f'{OUT_ROOT}: {len(labels):,} SNIDs etiquetados')
