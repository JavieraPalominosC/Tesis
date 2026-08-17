import json, glob
from pathlib import Path
import random
random.seed(42)

img_dir = 'data/images/elasticc_1/waterfall'
paths = sorted(glob.glob(f'{img_dir}/*.png'))
random.shuffle(paths)
n_val = int(len(paths) * 0.1)
folds = {"0": {"train": paths[n_val:], "val": paths[:n_val]}}
with open('data/folds_waterfall_full.json', 'w') as f:
    json.dump(folds, f)
print(f'waterfall full: {len(paths)-n_val:,} train, {n_val:,} val')

labels = {}
for p in glob.glob('data/images/elasticc_1/waterfall_labels/*.json'):
    with open(p) as f:
        labels.update(json.load(f))
with open('data/labels_waterfall_full.json', 'w') as f:
    json.dump(labels, f)
print(f'Labels: {len(labels):,} SNIDs')
