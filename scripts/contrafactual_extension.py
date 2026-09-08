"""Variar la DURACION de la anomalia para desconfundir la eleccion de metodo.

En toda la comparacion anterior la anomalia ocupa siempre la misma extension
(los puntos de banda r del ultimo 30% del tiempo, unos 50 tokens) y solo cambia
su amplitud. Eso deja un confundente serio: el percentil 95 enmascara justo ~50
tokens, o sea exactamente el tamano de nuestra zona. Puede estar ganando porque
le acertamos al presupuesto, no porque el criterio sea bueno.

El rectangulo, en cambio, elige su tamano a partir de los datos. Si la ventaja
de p95 es un artefacto, deberia evaporarse cuando la anomalia deja de medir 50
tokens; si es real, deberia aguantar.

Se fija df=300 y se varia dt, la ventana temporal perturbada. La zona de
referencia sigue siendo la fija (banda r, ultimo 30%): para dt chico la anomalia
llena solo una parte de ella, lo que diluye la reparacion medida, pero la
dilucion es la misma para todos los metodos, asi que la comparacion entre ellos
sigue siendo pareja.

Uso: python scripts/contrafactual_extension.py
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

from scripts.eval_enc_cmp import col_flux, col_mjd, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import GRID, PX, nll_por_token, reconstruir, zona_px
from scripts.contrafactual_diagnostico import parte
from scripts.contrafactual_comparacion import (
    metodo_base, metodo_rect, plausibilidad,
)
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DTS = [5, 15, 40, 90]
DF = 300
N_OBJ, SEED = 40, 0
METODOS = ['base p85', 'base p95', 'rect 1 pasa']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    pz = zona_px(*zona)
    area_tot = PX * PX
    area_z = (pz[1] - pz[0]) * (pz[3] - pz[2])
    area_f = area_tot - area_z

    def fuera(a, b):
        tot = float(F.mse_loss(a, b).item())
        return (tot * area_tot - parte(a, b, pz) * area_z) / area_f

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)

    # Ventanas temporales: puntos de banda r en los ultimos dt dias.
    ventanas, n_pts = {}, {dt: [] for dt in DTS}
    for s in snids:
        o, idx_cola, _ = curvas[s]
        sub = o.loc[idx_cola]
        t_fin = float(sub[col_mjd].max())
        for dt in DTS:
            idx = list(sub.index[sub[col_mjd] >= t_fin - dt])
            ventanas[(s, dt)] = idx
            n_pts[dt].append(len(idx))
    print('\n  dt[dias]   puntos perturbados (mediana, rango)')
    for dt in DTS:
        p = np.array(n_pts[dt])
        print(f'  {dt:6d}     {np.median(p):5.1f}  ({p.min()}-{p.max()})', flush=True)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imgs = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        imgs[(s, 'limpia')] = render(o, ENC, fig, ax)
        for dt in DTS:
            p = o.copy()
            idx = ventanas[(s, dt)]
            p.loc[idx, col_flux] = p.loc[idx, col_flux] + DF * sc
            imgs[(s, dt)] = render(p, ENC, fig, ax)
    plt.close(fig)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def codificar(img):
        x = tf(img).unsqueeze(0).to(dev)
        t = vq.encode_to_indices(x).view(1, -1).long().to(dev)
        return t, reconstruir(vq, t)

    limpias = {s: codificar(imgs[(s, 'limpia')]) for s in snids}

    res = {m: {} for m in METODOS}
    for dt in DTS:
        acc = {m: [] for m in METODOS}
        for s in snids:
            tp, xp = codificar(imgs[(s, dt)])
            tl, xl = limpias[s]
            base_z, base_f = parte(xp, xl, pz), fuera(xp, xl)
            salidas = {
                'base p85': metodo_base(mg, tp, dev, g, 0.85),
                'base p95': metodo_base(mg, tp, dev, g, 0.95),
                'rect 1 pasa': metodo_rect(mg, tp, dev, g),
            }
            for nombre, (cf, info) in salidas.items():
                xc = reconstruir(vq, cf)
                dmap = (xp - xc).abs().sum(dim=1)[0].cpu().numpy()
                masa = float(dmap.sum())
                pl = plausibilidad(xc)
                acc[nombre].append({
                    'repara': 100 * (base_z - parte(xc, xl, pz)) / base_z,
                    'dana': 100 * (fuera(xc, xl) - base_f) / base_f,
                    'n_tok': int((cf != tp).sum().item()),
                    'loc': (float(dmap[pz[0]:pz[1], pz[2]:pz[3]].sum()) / masa)
                           if masa else 0.0,
                    'rugosidad': pl['rugosidad'],
                })

        print(f'\n{"=" * 74}\ndt={dt} dias   df={DF}   '
              f'puntos perturbados (mediana) {np.median(n_pts[dt]):.0f}')
        print(f'  {"metodo":<13}{"repara":>8}{"dana":>8}{"tokens":>8}'
              f'{"loc":>8}{"rep/dano":>10}')
        for m in METODOS:
            a = acc[m]
            f_ = lambda k: float(np.nanmean([x[k] for x in a]))
            res[m][str(dt)] = {k: f_(k) for k in a[0]}
            rd = f_('repara') / f_('dana') if f_('dana') > 0 else np.inf
            print(f'  {m:<13}{f_("repara"):7.1f}%{f_("dana"):7.1f}%'
                  f'{f_("n_tok"):8.0f}{100 * f_("loc"):7.1f}%{rd:10.2f}', flush=True)

    with open(OUT / 'contrafactual_extension.json', 'w') as f:
        json.dump({'dts': DTS, 'df': DF,
                   'puntos_por_dt': {str(dt): n_pts[dt] for dt in DTS},
                   'resultados': res}, f, indent=2)

    COL = {'base p85': '#e45756', 'base p95': '#f2a640', 'rect 1 pasa': '#31a354'}
    MRK = {'base p85': 'o', 'base p95': 'v', 'rect 1 pasa': '^'}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for a, (k, ylab, tit) in zip(axes, [
            ('repara', 'repara dentro de la zona [%]', 'Reparacion'),
            ('dana', 'dana fuera de la zona [%]', 'Dano colateral'),
            ('n_tok', 'tokens cambiados', 'Cuanto se reescribe')]):
        for m in METODOS:
            a.plot(DTS, [res[m][str(dt)][k] for dt in DTS], MRK[m] + '-',
                   color=COL[m], label=m, ms=6)
        if k == 'n_tok':
            a.axhline(50, color='#4c78a8', ls=':', lw=1.3,
                      label='tamano de la zona fija (50 tokens)')
        a.set_xlabel('dt: duracion de la anomalia [dias]')
        a.set_ylabel(ylab)
        a.set_title(tit, fontsize=10)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Variando la duracion de la anomalia (df={DF}): el presupuesto fijo '
                 'de p95 contra el rectangulo adaptativo', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / 'contrafactual_extension.png', dpi=140)
    plt.close(fig)
    print(f'\nGuardado en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
