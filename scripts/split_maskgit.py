"""Reproduce la particion que uso el prior MaskGIT de la corrida completa.

Hace falta porque esa particion no se guardo en ningun archivo: se generaba al
vuelo dentro de train_maskgit_enc.py con

    random_split(dataset, [n_train, n - n_train],
                 generator=torch.Generator().manual_seed(42))

sobre el arreglo de tokens en el orden de paths_full.txt, y con razon 80/20. Es
independiente de folds_full.json, que es la particion 90/10 del VQ-VAE, por lo
que un objeto podia quedar en validacion del VQ-VAE y en entrenamiento del
prior a la vez. Medido: el 79,9% de la validacion del VQ-VAE cayo en el
entrenamiento del prior, y con ella 32 de los 40 objetos de evaluacion.

Los dos priors de la corrida completa (mascara aleatoria y de ventana) comparten
esta particion, porque usan los mismos tokens y la misma semilla; lo unico que
los diferencia es la forma de la mascara.

Las corridas nuevas ya no dependen de esto: train_maskgit_enc.py acepta
FOLDS_SPLIT y toma la particion del archivo de folds del VQ-VAE.
"""
import json
import os
from pathlib import Path

import torch
from torch.utils.data import random_split

PATHS = 'data/enc_cmp/mad300/paths_full.txt'
CACHE = 'data/enc_cmp/mad300/split_maskgit_full.json'
FRAC_TRAIN, SEMILLA = 0.8, 42


def val_maskgit(paths_file=PATHS, cache=CACHE):
    """SNIDs que el prior de la corrida completa NO vio entrenando."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return set(json.load(f)['val'])

    with open(paths_file) as f:
        paths = [l.strip() for l in f]

    n = len(paths)
    n_train = int(n * FRAC_TRAIN)
    _, va = random_split(range(n), [n_train, n - n_train],
                         generator=torch.Generator().manual_seed(SEMILLA))
    val = sorted({Path(paths[i]).stem for i in va.indices})

    if cache:
        with open(cache, 'w') as f:
            json.dump({'val': val, 'n_total': n, 'n_train': n_train}, f)
    return set(val)


if __name__ == '__main__':
    v = val_maskgit()
    print(f'objetos en validacion del prior: {len(v):,}')
