"""Cuanto reescribir: percentil 85 del protocolo contra alternativas mas parsimoniosas.

El diagnostico mostro que el contrafactual repara la zona del pico pero ensucia
el resto, y que el daño global viene de reescribir 154 tokens cuando la anomalia
ocupa unos 50. El barrido global sugeria que el percentil 95 (que enmascara
justo ~50) hace mucho menos daño, pero no sabiamos si tambien repara igual de
bien adentro. Eso es lo que decide si conviene cambiar el protocolo, asi que se
mide dentro y fuera de la zona por separado.

Uso: python scripts/contrafactual_percentil.py
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
import torch.nn.functional as F

from scripts.eval_enc_cmp import col_flux, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import (
    GRID, N_STEPS, PX, nll_por_token, reconstruir, seleccionar, zona_px,
)
from scripts.contrafactual_diagnostico import parte
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [100, 500]
PERCENTILES = [0.70, 0.85, 0.90, 0.95, 0.98]
N_OBJ, SEED = 40, 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    pz = zona_px(*zona)
    area_tot, area_z = PX * PX, (pz[1] - pz[0]) * (pz[3] - pz[2])
    area_f = area_tot - area_z

    def fuera(a, b):
        tot = float(F.mse_loss(a, b).item())
        return (tot * area_tot - parte(a, b, pz) * area_z) / area_f

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imgs = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        imgs[(s, 'limpia')] = render(o, ENC, fig, ax)
        for d in DFS:
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            imgs[(s, d)] = render(p, ENC, fig, ax)
    plt.close(fig)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def rec_de(img):
        x = tf(img).unsqueeze(0).to(dev)
        t = vq.encode_to_indices(x).view(1, -1).long().to(dev)
        return t, reconstruir(vq, t)

    limpias = {s: rec_de(imgs[(s, 'limpia')])[1] for s in snids}

    res = {}
    for d in DFS:
        res[str(d)] = {}
        print(f'\n{"=" * 68}\ndf={d}', flush=True)
        print(f'  {"percentil":>10}{"tokens":>8}{"MSE zona":>11}{"MSE fuera":>11}'
              f'{"repara":>9}{"dana":>8}')
        base_z = base_f = None
        for pc in PERCENTILES:
            zs, fs, ns = [], [], []
            bz, bf = [], []
            for s in snids:
                with torch.no_grad():
                    t, xr = rec_de(imgs[(s, d)])
                    nll = nll_por_token(mg, t, dev, g)
                    sel, _ = seleccionar(nll, pc)
                    m = torch.from_numpy(sel).view(1, -1).to(dev)
                    cf = mg.generate_counterfactual(t, m, n_steps=N_STEPS)
                    xc = reconstruir(vq, cf)
                xl = limpias[s]
                zs.append(parte(xc, xl, pz))
                fs.append(fuera(xc, xl))
                bz.append(parte(xr, xl, pz))
                bf.append(fuera(xr, xl))
                ns.append(int(sel.sum()))
            base_z, base_f = float(np.mean(bz)), float(np.mean(bf))
            mz, mf, mn = float(np.mean(zs)), float(np.mean(fs)), float(np.mean(ns))
            repara = 100 * (base_z - mz) / base_z
            dana = 100 * (mf - base_f) / base_f
            res[str(d)][str(pc)] = {'tokens': mn, 'mse_zona': mz, 'mse_fuera': mf,
                                    'repara_pct': repara, 'dana_pct': dana}
            marca = '  <- protocolo' if pc == 0.85 else ''
            print(f'  {f"p{int(pc * 100)}":>10}{mn:8.0f}{mz:11.5f}{mf:11.5f}'
                  f'{repara:8.1f}%{dana:7.1f}%{marca}', flush=True)
        res[str(d)]['sin_reescribir'] = {'mse_zona': base_z, 'mse_fuera': base_f}
        print(f'  {"sin tocar":>10}{0:8d}{base_z:11.5f}{base_f:11.5f}'
              f'{0.0:8.1f}%{0.0:7.1f}%')

    with open(OUT / 'contrafactual_percentil.json', 'w') as f:
        json.dump({'dfs': DFS, 'percentiles': PERCENTILES, 'resultados': res}, f,
                  indent=2)

    fig, axes = plt.subplots(1, len(DFS), figsize=(7.2 * len(DFS), 4.9))
    if len(DFS) == 1:
        axes = [axes]
    for a, d in zip(axes, DFS):
        r = res[str(d)]
        toks = [r[str(p)]['tokens'] for p in PERCENTILES]
        a.plot(toks, [r[str(p)]['repara_pct'] for p in PERCENTILES], 'o-',
               color='#31a354', label='repara dentro de la zona')
        a.plot(toks, [r[str(p)]['dana_pct'] for p in PERCENTILES], 's-',
               color='#e45756', label='dana fuera de la zona')
        a.axvline(50, color='#4c78a8', ls=':', lw=1.4,
                  label='tamano real de la zona (50 tokens)')
        a.axhline(0, color='#999999', lw=0.8)
        for p, t in zip(PERCENTILES, toks):
            a.annotate(f'p{int(p * 100)}', (t, r[str(p)]['repara_pct']),
                       textcoords='offset points', xytext=(0, 8), ha='center',
                       fontsize=7.5, color='#555555')
        a.set_xscale('log')
        a.set_xlabel('tokens reescritos')
        a.set_ylabel('% de cambio respecto de no reescribir')
        a.set_title(f'df={d}', fontsize=10)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle('Cuanto conviene reescribir: reparacion contra dano colateral\n'
                 '40 SNIa fuera del entrenamiento', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'contrafactual_percentil.png', dpi=140)
    plt.close(fig)
    print(f'\nGuardado en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
