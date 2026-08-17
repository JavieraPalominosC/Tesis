import json, glob
from pathlib import Path
from collections import Counter

# labels: {snid: clase_int}
labels = json.load(open('data/labels_overlay_full.json'))
# nombres de clase
class_names = json.load(open('data/folds/class_names.json'))  # ajustar si la ruta difiere

empty = [Path(l.strip()).stem for l in open('data/empty_overlay.txt')]
print(f'Total vacías: {len(empty):,}\n', flush=True)

cnt = Counter()
sin_label = 0
for snid in empty:
    if snid in labels:
        cnt[labels[snid]] += 1
    else:
        sin_label += 1

print('Vacías por tipo:')
for cls_int, n in cnt.most_common():
    name = class_names[cls_int] if isinstance(class_names, list) else class_names.get(str(cls_int), cls_int)
    print(f'  {name:25s} {n:>8,}')
if sin_label:
    print(f'  (sin label en dict): {sin_label:,}')
