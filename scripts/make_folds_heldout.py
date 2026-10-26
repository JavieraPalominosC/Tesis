"""Folds para el experimento de clases excluidas.

Deja seis clases completamente fuera del entrenamiento, para poder usarlas
despues como conjunto de anomalias reales en vez de perturbaciones sinteticas.
Las clases excluidas no aparecen ni en train ni en validacion: si aparecieran en
validacion, la seleccion del mejor checkpoint las habria visto de forma
indirecta.

Escribe:
  folds_heldout.json    train/val solo con las once clases vistas
  labels_heldout.json   etiquetas remapeadas a 0..10 para la cabeza de clasificacion
  heldout_eval.json     rutas de las clases excluidas, agrupadas por clase

Uso: python scripts/make_folds_heldout.py [out_root]
"""
import sys
import json
import glob
import random
from pathlib import Path

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'data/enc_cmp/mad300'
SEED = 42
VAL_FRAC = 0.1

# Lo que pidio el profe, traducido a las etiquetas que fija SN_FILES en
# run_mad300_full.sh: Supernova 2b, las dos SLSN, SNIax, SNIa-91bg y PISN.
EXCLUIDAS = {1, 2, 11, 14, 15, 16}

NOMBRES = [
    'SNIa-SALT2', 'SNIa-91bg', 'SNIax', 'SNII-Templates', 'SNII-NMF',
    'SNII+HostXT_V19', 'SNIb-Templates', 'SNIb+HostXT_V19',
    'SNIc-Templates', 'SNIc+HostXT_V19', 'SNIcBL+HostXT_V19',
    'SNIIb+HostXT_V19', 'SNIIn-MOSFIT', 'SNIIn+HostXT_V19',
    'SLSN-I+host', 'SLSN-I_no_host', 'PISN',
]


def main():
    labels = json.load(open(f'{ROOT}/labels.json'))
    paths = sorted(glob.glob(f'{ROOT}/images/*.png'))
    print(f'imagenes encontradas: {len(paths):,}')

    vistas, excluidas = [], {}
    sin_etiqueta = 0
    for p in paths:
        lab = labels.get(Path(p).stem)
        if lab is None:
            sin_etiqueta += 1
            continue
        if lab in EXCLUIDAS:
            excluidas.setdefault(lab, []).append(p)
        else:
            vistas.append(p)

    if sin_etiqueta:
        print(f'  aviso: {sin_etiqueta:,} imagenes sin etiqueta, quedan fuera')

    # Las etiquetas originales quedan con huecos al sacar seis clases, y la
    # cabeza de clasificacion necesita indices contiguos.
    quedan = sorted({labels[Path(p).stem] for p in vistas})
    remapeo = {viejo: nuevo for nuevo, viejo in enumerate(quedan)}

    random.Random(SEED).shuffle(vistas)
    n_val = int(len(vistas) * VAL_FRAC)
    folds = {'0': {'train': vistas[n_val:], 'val': vistas[:n_val]}}

    with open(f'{ROOT}/folds_heldout.json', 'w') as f:
        json.dump(folds, f)

    labels_rm = {Path(p).stem: remapeo[labels[Path(p).stem]] for p in vistas}
    with open(f'{ROOT}/labels_heldout.json', 'w') as f:
        json.dump(labels_rm, f)

    eval_set = {
        'excluidas': {str(k): v for k, v in sorted(excluidas.items())},
        'nombres': {str(k): NOMBRES[k] for k in sorted(excluidas)},
        'remapeo': {str(k): v for k, v in remapeo.items()},
        'nombres_vistas': {str(remapeo[k]): NOMBRES[k] for k in quedan},
    }
    with open(f'{ROOT}/heldout_eval.json', 'w') as f:
        json.dump(eval_set, f)

    print(f'\nentrenamiento: {len(vistas):,} imagenes, {len(quedan)} clases')
    print(f'  train {len(vistas) - n_val:,} | val {n_val:,}')
    for viejo in quedan:
        n = sum(1 for p in vistas if labels[Path(p).stem] == viejo)
        print(f'    {remapeo[viejo]:2d} (era {viejo:2d})  {NOMBRES[viejo]:22s} {n:8,d}')

    total_exc = sum(len(v) for v in excluidas.values())
    print(f'\nexcluidas: {total_exc:,} imagenes, {len(excluidas)} clases')
    for k in sorted(excluidas):
        print(f'    {k:2d}  {NOMBRES[k]:22s} {len(excluidas[k]):8,d}')
    print(f'\ntotal {len(vistas) + total_exc:,} '
          f'({100 * total_exc / (len(vistas) + total_exc):.1f}% excluido)')


if __name__ == '__main__':
    main()
