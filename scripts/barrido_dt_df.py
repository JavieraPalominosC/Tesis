"""Barrido 2D: duracion de la anomalia (Delta t) contra su amplitud (Delta F).

La evaluacion anterior solo movio la amplitud del pico; siempre perturbaba la
misma cola del 30% final. Para el contorno que pidio el profe hace falta el otro
eje: cuanto dura la anomalia. Aca se perturban los puntos de la banda r que caen
en los ultimos Delta t dias, con Delta t desde 0 (un solo punto, un fogonazo de
una epoca) hasta 80 dias.

El instante de referencia es el ULTIMO punto de la banda r, no el ultimo de la
curva completa. Asi la ventana siempre contiene al menos un punto y la grilla
queda rectangular, sin celdas vacias que ensucien el contorno.

Mismos 40 SNIa del fold de validacion que la evaluacion definitiva: ninguno fue
visto en entrenamiento por el VQ-VAE ni por el MaskGIT.

Uso: python scripts/barrido_dt_df.py
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.eval_enc_cmp import col_band, col_flux, col_mjd, fig_params, mejor_ckpt, render
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.eval_mascara_cruzada import nll_map
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

TARGET_BAND = 'r'
OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 30, 70, 150, 300, 500]
DTS = [0, 5, 10, 20, 40, 80]
# La celda ganadora y la de referencia. Entrenado siempre con mascara aleatoria;
# lo que cambia es la forma de la mascara al puntuar.
CELDAS = [('aleatoria', 'ventana'), ('aleatoria', 'aleatoria')]
N_OBJ, CTRL_Q, SEED = 40, 0.95, 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'dispositivo: {dev}', flush=True)

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
    mads = {s: curvas[s][2] for s in snids}
    print(f'objetos: {len(snids)} (ninguno visto en entrenamiento)', flush=True)

    # Ventanas temporales por objeto: indices de banda r en los ultimos dt dias.
    ventanas, n_pts, spans = {}, {dt: [] for dt in DTS}, {dt: [] for dt in DTS}
    for s in snids:
        o = curvas[s][0]
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        t_fin = float(sub[col_mjd].max())
        for dt in DTS:
            idx = list(sub.index[sub[col_mjd] >= t_fin - dt])
            if not idx:
                idx = list(sub.index[-1:])
            ventanas[(s, dt)] = idx
            n_pts[dt].append(len(idx))
            mjds = sub.loc[idx, col_mjd].values
            spans[dt].append(float(mjds.max() - mjds.min()))

    print('\n  dt[dias]   puntos perturbados (mediana, rango)   span real mediano')
    for dt in DTS:
        p = np.array(n_pts[dt])
        print(f'  {dt:6d}     {np.median(p):5.1f}  ({p.min()}-{p.max()})'
              f'                    {np.median(spans[dt]):6.1f} d', flush=True)

    vq_ck = mejor_ckpt(VQ_CKPT)
    vq = VQVAE.load_from_checkpoint(vq_ck, map_location=dev).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print('\nrenderizando...', flush=True)
    imgs = {s: render(curvas[s][0], ENC, fig, ax) for s in snids}
    for s in snids:
        o, _, sc = curvas[s]
        for dt in DTS:
            idx = ventanas[(s, dt)]
            for d in DFS:
                p = o.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
                imgs[(s, dt, d)] = render(p, ENC, fig, ax)
    print(f'imagenes: {len(imgs)}', flush=True)

    modelos = {}
    resultados = {}
    for entrenado, puntuado in CELDAS:
        if entrenado not in modelos:
            ck = mejor_ckpt(MODELOS[entrenado])
            modelos[entrenado] = MaskGIT.load_from_checkpoint(
                ck, map_location=dev, strict=False).eval().to(dev)
        mg = modelos[entrenado]

        g = torch.Generator(device=dev)
        g.manual_seed(SEED)
        ctrl = {s: nll_map(vq, mg, imgs[s], dev, puntuado, g)[0] for s in snids}
        cs = np.array([ctrl[s] for s in snids])
        tau = float(np.quantile(cs, CTRL_Q))

        recall = np.zeros((len(DTS), len(DFS)))
        pareada = np.zeros((len(DTS), len(DFS)))
        for i, dt in enumerate(DTS):
            for j, d in enumerate(DFS):
                v = np.array([nll_map(vq, mg, imgs[(s, dt, d)], dev, puntuado, g)[0]
                              for s in snids])
                recall[i, j] = float((v > tau).mean())
                pareada[i, j] = float(np.mean([v[k] > ctrl[s]
                                               for k, s in enumerate(snids)]))
            print(f'  [{entrenado}|{puntuado}] dt={dt:3d}d  '
                  f'recall={np.round(recall[i], 2).tolist()}', flush=True)

        resultados[f'{entrenado}|{puntuado}'] = {
            'tau': tau, 'recall': recall.tolist(), 'pareada': pareada.tolist(),
        }

    meta = {
        'dfs': DFS, 'dts': DTS, 'objetos': snids, 'mads': mads,
        'mad_medio': float(np.mean(list(mads.values()))),
        'puntos_por_dt': {str(dt): n_pts[dt] for dt in DTS},
        'span_mediano_por_dt': {str(dt): float(np.median(spans[dt])) for dt in DTS},
        'celdas': resultados,
    }
    with open(OUT / 'barrido_dt_df.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nGuardado en {OUT}/barrido_dt_df.json', flush=True)


if __name__ == '__main__':
    main()
