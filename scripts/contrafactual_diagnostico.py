"""Diagnostico: donde actua el contrafactual, dentro o fuera de la zona.

La primera medicion dio un resultado contradictorio. Visualmente el contrafactual
aplana el pico inyectado, y la masa del cambio se concentra en la zona objetivo
(23% en un area que es 4.9% de la imagen). Pero el MSE global contra la curva
limpia EMPEORA, de 0.0274 a 0.0315. Las dos cosas pueden ser ciertas a la vez si
el contrafactual arregla la zona y a cambio ensucia el resto.

Este script separa las dos regiones y agrega dos controles que faltaban:

  posiciones al azar   reescribir la misma cantidad de tokens (~154) pero en
                       posiciones aleatorias en vez de las de mayor NLL. Si
                       apuntar por NLL sirve, el dirigido tiene que ganarle.
  sin reescribir       la perturbada tal cual, como linea de base.

Ademas se mide la distancia en tokens dentro y fuera de la zona, para entender
por que el 74% de los tokens difiere entre limpia y perturbada incluso en df=10:
si esa diferencia es pareja dentro y fuera, no viene de la anomalia sino de que
la normalizacion por objeto se recalcula y corre la imagen entera.

Uso: python scripts/contrafactual_diagnostico.py
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
    GRID, N_STEPS, PERCENTIL, PX, nll_por_token, reconstruir, seleccionar, zona_px,
)
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
N_OBJ, SEED = 40, 0


@torch.no_grad()
def variantes(vq, mg, img, dev, g):
    """Contrafactual dirigido por NLL y contrafactual en posiciones al azar."""
    x = tf(img).unsqueeze(0).to(dev)
    tokens = vq.encode_to_indices(x).view(1, -1).long().to(dev)

    nll = nll_por_token(mg, tokens, dev, g)
    sel, _ = seleccionar(nll, PERCENTIL)
    n_sel = int(sel.sum())

    m_dir = torch.from_numpy(sel).view(1, -1).to(dev)
    cf_dir = mg.generate_counterfactual(tokens, m_dir, n_steps=N_STEPS)

    T = tokens.shape[1]
    m_azar = torch.zeros(1, T, dtype=torch.bool, device=dev)
    m_azar[0, torch.randperm(T, device=dev, generator=g)[:n_sel]] = True
    cf_azar = mg.generate_counterfactual(tokens, m_azar, n_steps=N_STEPS)

    return {
        'tokens': tokens, 'cf_dir': cf_dir, 'cf_azar': cf_azar, 'n_sel': n_sel,
        'x_rec': reconstruir(vq, tokens),
        'x_dir': reconstruir(vq, cf_dir),
        'x_azar': reconstruir(vq, cf_azar),
        'sel': sel,
    }


def parte(a, b, region):
    """MSE entre dos reconstrucciones restringido a una region de pixeles."""
    p0, p1, q0, q1 = region
    return float(F.mse_loss(a[:, :, p0:p1, q0:q1], b[:, :, p0:p1, q0:q1]).item())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    r0, r1, c0, c1 = zona
    pz = zona_px(r0, r1, c0, c1)
    print(f'zona token filas {r0}-{r1} cols {c0}-{c1}   pixeles {pz}', flush=True)

    # Mascara booleana de la zona en la grilla de tokens.
    zt = np.zeros((GRID, GRID), bool)
    zt[r0:r1 + 1, c0:c1 + 1] = True
    zt = zt.reshape(-1)
    print(f'tokens en la zona: {zt.sum()} de {GRID * GRID}', flush=True)

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

    limpias = {s: variantes(vq, mg, imgs[(s, 'limpia')], dev, g) for s in snids}

    # Region complementaria: se calcula el MSE global y el de la zona, y el de
    # afuera se despeja por diferencia de sumas de cuadrados ponderadas por area.
    area_tot = PX * PX
    area_z = (pz[1] - pz[0]) * (pz[3] - pz[2])
    area_f = area_tot - area_z

    def fuera(a, b):
        tot = float(F.mse_loss(a, b).item())
        return (tot * area_tot - parte(a, b, pz) * area_z) / area_f

    res = {}
    for d in DFS:
        acc = []
        for s in snids:
            rp = variantes(vq, mg, imgs[(s, d)], dev, g)
            rl = limpias[s]
            xl = rl['x_rec']
            t_l = rl['tokens'][0].cpu().numpy()
            t_p = rp['tokens'][0].cpu().numpy()
            dif = t_p != t_l
            acc.append({
                'mse_z_pert': parte(rp['x_rec'], xl, pz),
                'mse_z_dir': parte(rp['x_dir'], xl, pz),
                'mse_z_azar': parte(rp['x_azar'], xl, pz),
                'mse_f_pert': fuera(rp['x_rec'], xl),
                'mse_f_dir': fuera(rp['x_dir'], xl),
                'mse_f_azar': fuera(rp['x_azar'], xl),
                'tokdif_zona': float(dif[zt].mean()),
                'tokdif_fuera': float(dif[~zt].mean()),
                'sel_en_zona': float(rp['sel'][zt].mean()),
                'sel_fuera': float(rp['sel'][~zt].mean()),
            })
        res[d] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        x = res[d]
        print(f'\n[df={d}]')
        print(f'  MSE contra limpia DENTRO de la zona: perturbada {x["mse_z_pert"]:.5f}'
              f' -> dirigido {x["mse_z_dir"]:.5f}   (azar {x["mse_z_azar"]:.5f})')
        print(f'  MSE contra limpia FUERA de la zona : perturbada {x["mse_f_pert"]:.5f}'
              f' -> dirigido {x["mse_f_dir"]:.5f}   (azar {x["mse_f_azar"]:.5f})')
        print(f'  tokens limpia!=perturbada: dentro {100 * x["tokdif_zona"]:.1f}%   '
              f'fuera {100 * x["tokdif_fuera"]:.1f}%')
        print(f'  el selector apunta a la zona: {100 * x["sel_en_zona"]:.1f}% de sus '
              f'tokens (fuera {100 * x["sel_fuera"]:.1f}%)', flush=True)

    with open(OUT / 'contrafactual_diagnostico.json', 'w') as f:
        json.dump({'dfs': DFS, 'zona_token': list(zona), 'resultados':
                   {str(d): res[d] for d in DFS}}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    a = axes[0]
    a.plot(DFS, [res[d]['mse_z_pert'] for d in DFS], 'o-', color='#e45756',
           label='perturbada (sin reescribir)')
    a.plot(DFS, [res[d]['mse_z_dir'] for d in DFS], 'o-', color='#4c78a8',
           label='contrafactual dirigido por NLL')
    a.plot(DFS, [res[d]['mse_z_azar'] for d in DFS], 's--', color='#8856a7',
           label='contrafactual en posiciones al azar')
    a.set_title('DENTRO de la zona del pico\nmenor es mejor: mas cerca de la limpia',
                fontsize=10)
    a = axes[1]
    a.plot(DFS, [res[d]['mse_f_pert'] for d in DFS], 'o-', color='#e45756',
           label='perturbada (sin reescribir)')
    a.plot(DFS, [res[d]['mse_f_dir'] for d in DFS], 'o-', color='#4c78a8',
           label='contrafactual dirigido por NLL')
    a.plot(DFS, [res[d]['mse_f_azar'] for d in DFS], 's--', color='#8856a7',
           label='contrafactual en posiciones al azar')
    a.set_title('FUERA de la zona\nel dano colateral de reescribir', fontsize=10)
    for a in axes:
        a.set_xscale('log')
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel('MSE contra la reconstruccion limpia')
        a.legend(frameon=False, fontsize=8)
    fig.suptitle('El contrafactual arregla la zona y ensucia el resto  '
                 '(40 SNIa fuera del entrenamiento)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / 'contrafactual_diagnostico.png', dpi=140)
    plt.close(fig)
    print(f'\nGuardado en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
