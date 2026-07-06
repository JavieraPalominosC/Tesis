import os, json, glob, random
random.seed(42)

for rep in ['filled', 'overlay']:
    img_dir = f'data/images_exp/{rep}'
    paths = sorted(glob.glob(f'{img_dir}/*.png'))
    random.shuffle(paths)
    n_val = int(len(paths) * 0.1)
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    folds = {"0": {"train": train_paths, "val": val_paths}}
    out = f'data/images_exp/folds_{rep}.json'
    with open(out, 'w') as f:
        json.dump(folds, f)
    print(f'{rep}: {len(train_paths)} train, {len(val_paths)} val -> {out}')

labels = {}
for p in glob.glob('data/images_exp/labels_parts/*.json'):
    with open(p) as f:
        labels.update(json.load(f))
with open('data/images_exp/labels_subset.json', 'w') as f:
    json.dump(labels, f)
print(f'Labels unidos: {len(labels)} SNIDs')
