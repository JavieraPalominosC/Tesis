"""
Compara los anomaly scores de las 3 representaciones lado a lado.
Carga los scores_by_group.json de cada una y calcula la metrica de
separacion (rango inter-grupo / dispersion media intra-grupo).

Uso: python scripts/compare_representations.py
"""

import json
import numpy as np
from pathlib import Path

REPS = ['overlay_lines', 'points', 'waterfall']
GROUP_NAMES = ['SNIa', 'SNIbc', 'SNII', 'SNIIn', 'SLSN', 'PISN']

print("="*75)
print("COMPARACION DE REPRESENTACIONES - ANOMALY SCORES")
print("="*75)

data = {}
for rep in REPS:
    path = Path(f'results/maskgit_full/{rep}/eval/scores_by_group.json')
    if not path.exists():
        print(f"\n[{rep}] AUN NO DISPONIBLE ({path})")
        continue
    with open(path) as f:
        data[rep] = json.load(f)

# Tabla de medias por grupo
print(f"\n{'Grupo':<8}", end='')
for rep in REPS:
    if rep in data:
        print(f"{rep:>16}", end='')
print()
print("-"*75)
for g in GROUP_NAMES:
    print(f"{g:<8}", end='')
    for rep in REPS:
        if rep in data:
            m = data[rep][g]['mean']
            print(f"{m:>16.4f}", end='')
    print()

# Metrica de separacion por representacion
print("\n" + "="*75)
print("METRICA DE SEPARACION (mayor = mejor discriminacion entre tipos)")
print("  separacion = (media_max - media_min) / dispersion_media_intra_grupo")
print("="*75)
for rep in REPS:
    if rep not in data:
        continue
    means = np.array([data[rep][g]['mean'] for g in GROUP_NAMES])
    stds  = np.array([data[rep][g]['std']  for g in GROUP_NAMES])
    sep = (means.max() - means.min()) / stds.mean()
    # orden de grupos por score
    order = sorted(zip(GROUP_NAMES, means), key=lambda x: -x[1])
    order_str = ' > '.join(f'{n}({m:.3f})' for n, m in order)
    print(f"\n[{rep}] separacion = {sep:.3f}")
    print(f"   {order_str}")

print("\n" + "="*75)
print("Nota: jerarquia fisica esperada -> raros arriba (SLSN, PISN, SNIIn),")
print("      comunes abajo (SNIa, SNIbc).")
print("="*75)
