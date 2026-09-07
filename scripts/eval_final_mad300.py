"""Evaluacion definitiva: mad300 sobre el dataset completo, sin fuga de datos.

Por que existe este script y no reusamos el anterior. Los 40 SNIa que veniamos
usando para evaluar salieron del manifiesto sintetico original, y al entrenar
sobre las 1.39M imagenes 37 de ellos quedaron dentro del conjunto de
entrenamiento. Sus versiones limpias fueron memorizadas, asi que el NLL del
control baja artificialmente, la brecha contra la perturbada se agranda sola y
el recall sale optimista. Los numeros no medirian deteccion sino memoria.

Aca se eligen objetos del fold de validacion, que ni el VQ-VAE ni el MaskGIT
vieron nunca. Sobre esos se inyecta el pico en la cola de la banda r y se mide
de verdad.

Ademas se cruzan las dos formas de mascara. Ya sabiamos por el experimento con
la codificacion antigua que lo que rinde es entrenar con mascara aleatoria y
puntuar con ventana contigua: al tapar un bloque entero el modelo no puede
reconstruir el pico copiandose de los tokens vecinos, que tambien estan
contaminados. Aca se confirma o se descarta sobre los modelos definitivos.

Uso: python scripts/eval_final_mad300.py [N_OBJETOS]
"""
import sys
sys.path.insert(0, '.')
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from scripts.eval_enc_cmp import (
    MIN_TAIL, TAIL_FRAC, TARGET_BAND, auc, col_band, col_flux, col_mjd,
    fig_params, mad, mejor_ckpt, render,
)
from scripts.eval_mascara_cruzada import FORMAS, nll_map, N_RONDAS
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

ENC = 'mad300'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
BASE_FILE = 'lc_SNIa-SALT2.parquet'
FOLDS = 'data/enc_cmp/mad300/folds_full.json'
VQ_CKPT = 'results/enc_cmp_full/mad300/vqvae/checkpoints/**/*.ckpt'
MODELOS = {
    'aleatoria': 'results/enc_cmp_full/mad300/maskgit/checkpoints/**/*.ckpt',
    'ventana': 'results/enc_cmp_full/mad300/maskgit_ventana/checkpoints/**/*.ckpt',
}
OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]
N_OBJ, CTRL_Q, SEED = 40, 0.95, 0


def candidatos(n_obj):
    """SNIa del fold de validacion con cola suficiente en la banda r."""
    with open(FOLDS) as f:
        val = json.load(f)['0']['val']
    en_val = set()
    for p in val:
        m = re.match(r'(\d+)', Path(p).stem)
        if m:
            en_val.add(m.group(1))

    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df['SNID'] = df['SNID'].astype(str)
    df = df[df['SNID'].isin(en_val)]
    print(f'SNIa en el fold de validacion: {df["SNID"].nunique()}', flush=True)

    rng = np.random.default_rng(SEED)
    snids = sorted(df['SNID'].unique())
    rng.shuffle(snids)

    curvas = {}
    for snid in snids:
        o = df[df['SNID'] == snid].copy()
        o[col_flux] = o[col_flux].astype('float64')
        sc = mad(o.loc[o[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(sc) or sc <= 0:
            continue
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        if len(sub) < MIN_TAIL:
            continue
        t0, t1 = float(o[col_mjd].min()), float(o[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])
        curvas[snid] = (o, idx, sc)
        if len(curvas) >= n_obj:
            break
    return curvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    r0, r1, c0, c1 = zona_objetivo()
    print(f'dispositivo: {dev}   zona objetivo: filas {r0}-{r1}, cols {c0}-{c1}', flush=True)

    curvas = candidatos(n_obj)
    print(f'objetos de evaluacion (ninguno visto en entrenamiento): {len(curvas)}', flush=True)
    mads = {s: v[2] for s, v in curvas.items()}

    vq_ck = mejor_ckpt(VQ_CKPT)
    print(f'vqvae: {vq_ck}', flush=True)
    vq = VQVAE.load_from_checkpoint(vq_ck, map_location=dev).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Las imagenes no dependen del MaskGIT, asi que se renderizan y tokenizan una
    # sola vez y se reusan en las cuatro celdas del cruce.
    imgs = {}
    for snid, (o, idx, sc) in curvas.items():
        imgs[(snid, 'control')] = render(o, ENC, fig, ax)
        for d in DFS:
            p = o.copy()
            p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
            imgs[(snid, d)] = render(p, ENC, fig, ax)
    print(f'imagenes renderizadas: {len(imgs)}', flush=True)

    res = {}
    for modelo, patron in MODELOS.items():
        mg_ck = mejor_ckpt(patron)
        print(f'\nmaskgit {modelo}: {mg_ck}', flush=True)
        mg = MaskGIT.load_from_checkpoint(mg_ck, map_location=dev, strict=False).eval().to(dev)

        for forma in FORMAS:
            g = torch.Generator(device=dev)
            g.manual_seed(SEED)
            ctrl, pert = {}, {d: {} for d in DFS}
            loc = {d: [] for d in DFS}
            loc_ctrl = []

            for snid in curvas:
                v, mapa = nll_map(vq, mg, imgs[(snid, 'control')], dev, forma, g)
                ctrl[snid] = v
                rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
                loc_ctrl.append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))
                for d in DFS:
                    v, mapa = nll_map(vq, mg, imgs[(snid, d)], dev, forma, g)
                    pert[d][snid] = v
                    rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
                    loc[d].append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))

            cs = np.array([ctrl[s] for s in curvas])
            tau = float(np.quantile(cs, CTRL_Q))
            m = {'tau': tau, 'loc_control': float(np.mean(loc_ctrl)), 'por_df': {}}
            for d in DFS:
                ps = np.array([pert[d][s] for s in curvas])
                m['por_df'][d] = {
                    'recall': float((ps > tau).mean()),
                    'auc': auc(cs, ps),
                    'pareada': float(np.mean([pert[d][s] > ctrl[s] for s in curvas])),
                    'localizacion': float(np.mean(loc[d])),
                }
            rhos = [spearmanr(DFS, [pert[d][s] for d in DFS]).statistic for s in curvas]
            m['monotonia'] = float(np.median([r for r in rhos if np.isfinite(r)]))
            m['nll_control'] = {s: float(ctrl[s]) for s in curvas}
            m['nll_pert'] = {str(d): {s: float(pert[d][s]) for s in curvas} for d in DFS}
            res[f'{modelo}|{forma}'] = m

            marca = '   <- coinciden' if modelo == forma else ''
            print(f'\n[entrenado {modelo} / puntuado {forma}]{marca}', flush=True)
            print(f'  monotonia={m["monotonia"]:+.3f}   loc control={m["loc_control"]:.2f}',
                  flush=True)
            for d in DFS:
                x = m['por_df'][d]
                print(f'    df{d:<4} recall={x["recall"]:.2f}  auc={x["auc"]:.3f}  '
                      f'pareada={x["pareada"]:.2f}  loc={x["localizacion"]:.2f}', flush=True)
        del mg
        torch.cuda.empty_cache()

    with open(OUT / 'final_mad300.json', 'w') as f:
        json.dump({'dfs': DFS, 'objetos': sorted(curvas), 'mads': mads,
                   'rondas': N_RONDAS, 'celdas': res}, f, indent=2)

    estilos = {
        'aleatoria|ventana': ('#31a354', '-', 'entrenado aleatorio, puntuado ventana'),
        'aleatoria|aleatoria': ('#8856a7', '-', 'entrenado aleatorio, puntuado aleatorio'),
        'ventana|ventana': ('#e6550d', '--', 'entrenado ventana, puntuado ventana'),
        'ventana|aleatoria': ('#e6550d', ':', 'entrenado ventana, puntuado aleatorio'),
    }
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, clave, tit in zip(axes, ('recall', 'auc', 'localizacion'),
                             ('Recall con tau = P95 de los controles',
                              'AUC control contra perturbada',
                              'El maximo del NLL cae donde metimos el pico')):
        for k, (col, ls, lab) in estilos.items():
            a.plot(DFS, [res[k]['por_df'][d][clave] for d in DFS], 'o', ls=ls, lw=2,
                   ms=4, color=col, label=lab)
        a.set_xscale('log')
        a.set_xlabel(r'$df$  [veces MAD$_r$ del objeto]')
        a.set_ylabel(clave)
        a.set_title(tit, fontsize=10)
        a.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig2.suptitle(f'mad300 sobre 1.39M, {len(curvas)} SNIa fuera del entrenamiento',
                  fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT / 'final_mad300.png', dpi=140)
    plt.close(fig2)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
