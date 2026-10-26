"""Barrido extendido en los dos ejes de la anomalia, pedido el 28 de septiembre.

Los barridos anteriores dejaban sin cubrir los dos extremos que interesan:

  Parte A, amplitudes chicas. El barrido de amplitud arrancaba en Delta f = 10
  MAD, donde el macro F1 ya era 0.615 con recall 0.35, de modo que el limite de
  sensibilidad quedaba por debajo del rango medido. Aca se baja hasta 1 MAD.

  Parte B, duraciones largas. El barrido 2D llegaba hasta Delta t = 80 dias, que
  sobre estas curvas cubre apenas el 32% de los puntos de la banda r, cuyo span
  mediano es de 499 dias. Aca se llega hasta 320.

Se evaluan solo dos celdas del cruce de mascaras: la combinacion recomendada
(prior entrenado con mascara aleatoria, puntuado con ventana) y la de referencia
(puntuado con mascara aleatoria), que es la que muestra cuanto aporta la forma
de la mascara al puntuar.

Uso: python scripts/barrido_extendido.py [n_objetos]
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
from scipy.stats import spearmanr

from scripts.eval_enc_cmp import (
    TARGET_BAND, auc, col_band, col_flux, col_mjd, fig_params, mejor_ckpt, render,
)
from scripts.eval_final_mad300 import ENC, MODELOS, PRIOR_TAG, VQ_CKPT, candidatos
from scripts.eval_mascara_cruzada import nll_map
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')

# Parte A: se agregan 1, 2, 3, 5 y 7 por debajo del rango anterior.
DFS_1D = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500]

# Parte B: se agregan 160 y 320 por encima del rango anterior.
DFS_2D = [3, 10, 30, 70, 150, 300, 500]
DTS = [0, 5, 10, 20, 40, 80, 160, 320]

CELDAS = ['ventana', 'aleatoria']  # forma de la mascara al puntuar
PRIOR = 'aleatoria'                # prior entrenado con mascara aleatoria
N_OBJ, CTRL_Q, SEED = 40, 0.95, 0


def macro_f1(neg, pos, tau):
    """F1 promediado sobre las clases normal y anomala."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    tp = float((pos > tau).sum()); fn = float(len(pos) - tp)
    fp = float((neg > tau).sum()); tn = float(len(neg) - fp)

    def f1(vp, vfp, vfn):
        p = vp / (vp + vfp) if (vp + vfp) else 0.0
        r = vp / (vp + vfn) if (vp + vfn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    return 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))


def ventana_dt(o, dt):
    """Indices de la banda r dentro de los ultimos dt dias.

    El instante de referencia es el ultimo punto de la banda r, no el de la
    curva completa, de modo que la ventana nunca queda vacia y la grilla sale
    rectangular. Con dt = 0 se perturba un solo punto.
    """
    sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
    t_fin = float(sub[col_mjd].max())
    if dt <= 0:
        return sub.index[-1:]
    return sub.index[sub[col_mjd] >= t_fin - dt]


def puntuar(vq, mg, imgs, claves, dev, forma, zona):
    """NLL maxima y acierto de localizacion para un conjunto de imagenes."""
    r0, r1, c0, c1 = zona
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    scores, aciertos = {}, {}
    for k in claves:
        v, mapa = nll_map(vq, mg, imgs[k], dev, forma, g)
        rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
        scores[k] = float(v)
        aciertos[k] = bool(r0 <= rr <= r1 and c0 <= cc <= c1)
    return scores, aciertos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    zona = zona_objetivo()
    print(f'dispositivo: {dev} | zona objetivo {zona}', flush=True)

    curvas = candidatos(n_obj)
    snids = sorted(curvas)
    mads = {s: curvas[s][2] for s in snids}
    print(f'objetos: {len(snids)}', flush=True)

    vq_ck = mejor_ckpt(VQ_CKPT)
    vq = VQVAE.load_from_checkpoint(vq_ck, map_location=dev).eval().to(dev)
    mg_ck = mejor_ckpt(MODELOS[PRIOR])
    mg = MaskGIT.load_from_checkpoint(mg_ck, map_location=dev, strict=False).eval().to(dev)
    print(f'vqvae: {vq_ck}\nmaskgit: {mg_ck}', flush=True)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Renderizado unico: las imagenes no dependen de la forma de la mascara.
    imgs = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        imgs[(s, 'control')] = render(o, ENC, fig, ax)
        for d in DFS_1D:
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            imgs[(s, '1d', d)] = render(p, ENC, fig, ax)
        for dt in DTS:
            idx = ventana_dt(o, dt)
            for d in DFS_2D:
                p = o.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
                imgs[(s, '2d', d, dt)] = render(p, ENC, fig, ax)
    print(f'imagenes renderizadas: {len(imgs)}', flush=True)

    cobertura = {dt: float(np.mean([len(ventana_dt(curvas[s][0], dt)) /
                                    max((curvas[s][0][col_band] == TARGET_BAND).sum(), 1)
                                    for s in snids])) for dt in DTS}

    salida = {'objetos': snids, 'mads': mads, 'dfs_1d': DFS_1D,
              'dfs_2d': DFS_2D, 'dts': DTS, 'cobertura_dt': cobertura,
              'prior': PRIOR, 'celdas': {}}

    for forma in CELDAS:
        print(f'\npuntuando con mascara {forma}...', flush=True)
        claves = [(s, 'control') for s in snids]
        claves += [(s, '1d', d) for s in snids for d in DFS_1D]
        claves += [(s, '2d', d, dt) for s in snids for d in DFS_2D for dt in DTS]
        sc, ac = puntuar(vq, mg, imgs, claves, dev, forma, zona)

        cs = np.array([sc[(s, 'control')] for s in snids])
        tau = float(np.quantile(cs, CTRL_Q))

        una_d = {}
        for d in DFS_1D:
            ps = np.array([sc[(s, '1d', d)] for s in snids])
            una_d[str(d)] = {
                'recall': float((ps > tau).mean()),
                'auc': auc(cs, ps),
                'macro_f1': macro_f1(cs, ps, tau),
                'pareada': float(np.mean([sc[(s, '1d', d)] > sc[(s, 'control')] for s in snids])),
                'localizacion': float(np.mean([ac[(s, '1d', d)] for s in snids])),
            }

        dos_d = {}
        for d in DFS_2D:
            for dt in DTS:
                ps = np.array([sc[(s, '2d', d, dt)] for s in snids])
                dos_d[f'{d}|{dt}'] = {
                    'recall': float((ps > tau).mean()),
                    'auc': auc(cs, ps),
                    'macro_f1': macro_f1(cs, ps, tau),
                }

        rhos = [spearmanr(DFS_1D, [sc[(s, '1d', d)] for d in DFS_1D]).statistic for s in snids]
        salida['celdas'][forma] = {
            'tau': tau, 'monotonia': float(np.nanmedian(rhos)),
            'loc_control': float(np.mean([ac[(s, 'control')] for s in snids])),
            'una_d': una_d, 'dos_d': dos_d,
            'nll_control': {s: sc[(s, 'control')] for s in snids},
        }

        print(f'  tau={tau:.2f}  monotonia={salida["celdas"][forma]["monotonia"]:.3f}')
        for d in DFS_1D[:6]:
            u = una_d[str(d)]
            print(f'    df={d:>3d}  recall {u["recall"]:.2f}  macroF1 {u["macro_f1"]:.3f}  '
                  f'AUC {u["auc"]:.3f}  pareada {u["pareada"]:.2f}')

    destino = OUT / f'barrido_extendido{PRIOR_TAG}.json'
    with open(destino, 'w') as f:
        json.dump(salida, f, indent=1)
    print(f'\nescrito {destino}')


if __name__ == '__main__':
    main()
