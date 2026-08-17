import json, glob, random
from pathlib import Path
random.seed(42)

img_dir = 'data/images/elasticc_1/overlay'
paths = sorted(glob.glob(f'{img_dir}/*.png'))
random.shuffle(paths)
n_val = int(len(paths) * 0.1)
folds = {"0": {"train": paths[n_val:], "val": paths[:n_val]}}
with open('data/folds_overlay_full.json', 'w') as f:
    json.dump(folds, f)
print(f'overlay full: {len(paths)-n_val:,} train, {n_val:,} val')

labels = {}
for p in glob.glob('data/images/elasticc_1/overlay_labels/*.json'):
    with open(p) as f:
        labels.update(json.load(f))
with open('data/labels_overlay_full.json', 'w') as f:
    json.dump(labels, f)
print(f'Labels: {len(labels):,} SNIDs')
