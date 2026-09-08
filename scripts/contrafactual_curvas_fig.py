"""Figura de tesis: contrafactuales presentados como curvas de luz.

La primera version superponia las seis bandas en escala symlog y quedaba
ilegible: el mensaje se perdia. Aca se separa en dos paneles por objeto.

  izquierda  la banda r, que es donde inyectamos, en escala lineal, con la curva
             limpia, la perturbada y los dos contrafactuales. La ventana de
             inyeccion va sombreada.
  derecha    el residuo contra la curva limpia verdadera para TODAS las bandas,
             con la banda del piso del inversor marcada. Muestra de una sola vez
             que el pico se deshace y que el resto de la curva no se toca: si el
             residuo del contrafactual cae dentro del piso, el cambio fuera de
             la zona es fisicamente indistinguible de cero.

Uso: python scripts/contrafactual_curvas_fig.py
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
from matplotlib.lines import Line2D

from scripts.eval_enc_cmp import (
    all_bands, col_band, col_flux, col_mjd, fig_params, mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import reconstruir
from scripts.contrafactual_comparacion import metodo_base, metodo_rect
from scripts.contrafactual_flujo import invertir
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DF, SEED = 300, 0
OBJETOS = 3
BANDA = 'r'


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(OUT / 'contrafactual_flujo.json') as f:
        piso = json.load(f)['piso_inversor_rmse']

    curvas = candidatos(40)
    snids = sorted(curvas)[:OBJETOS]
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)

    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    def stats(v):
        v = np.asarray(v, dtype='float64')
        m = np.median(v)
        s = np.median(np.abs(v - m))
        return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)

    fig, axes = plt.subplots(OBJETOS, 2, figsize=(15, 3.5 * OBJETOS),
                             gridspec_kw={'width_ratios': [1.15, 1]})

    for i, s in enumerate(snids):
        o, idx_cola, sc = curvas[s]
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        p = o.copy()
        p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + DF * sc
        med_p, esc_p = stats(p[col_flux].values)

        with torch.no_grad():
            x = tf(render(p, ENC, fr, ar)).unsqueeze(0).to(dev)
            tp = vq.encode_to_indices(x).view(1, -1).long().to(dev)
            c95, _ = metodo_base(mg, tp, dev, g, 0.95)
            cre, _ = metodo_rect(mg, tp, dev, g)
            f95 = invertir(reconstruir(vq, c95), med_p, esc_p, o, t0, rango)
            fre = invertir(reconstruir(vq, cre), med_p, esc_p, o, t0, rango)

        t = o[col_mjd].values - t0
        ftrue = o[col_flux].values.astype('float64')
        fpert = p[col_flux].values.astype('float64')
        mr = (o[col_band] == BANDA).values
        cola = np.zeros(len(o), bool)
        cola[o.index.get_indexer(idx_cola)] = True

        a = axes[i, 0]
        a.axvspan(t[cola].min(), t[cola].max(), color='#39ff14', alpha=0.13,
                  label='ventana de inyeccion' if i == 0 else None)
        a.plot(t[mr], fpert[mr], ':', color='#e45756', lw=1.6, marker='.', ms=6,
               label='perturbada' if i == 0 else None)
        a.plot(t[mr], ftrue[mr], '-', color='#222222', lw=1.5, marker='o', ms=4,
               label='limpia (verdad del parquet)' if i == 0 else None)
        a.plot(t[mr], f95[mr], '--', color='#f2a640', lw=1.5, marker='v', ms=5,
               label='contrafactual, base p95' if i == 0 else None)
        a.plot(t[mr], fre[mr], '--', color='#31a354', lw=1.5, marker='^', ms=5,
               label='contrafactual, region contigua' if i == 0 else None)
        a.set_ylabel(f'FLUXCAL, banda {BANDA}')
        a.set_title(f'SNID {s}: la banda donde inyectamos  (df={DF})', fontsize=10)
        if i == 0:
            a.legend(frameon=False, fontsize=8, loc='upper left')

        a = axes[i, 1]
        a.axhspan(-piso, piso, color='#4c78a8', alpha=0.16,
                  label=f'piso del inversor (+-{piso:.1f} FLUXCAL)' if i == 0 else None)
        a.axhline(0, color='#888888', lw=0.8)
        for band_key, j in all_bands.items():
            m = (o[col_band] == band_key).values
            if not m.any():
                continue
            c = fig_params['colors'][j]
            a.plot(t[m], f95[m] - ftrue[m], 'v', color=c, ms=4, alpha=0.85)
            a.plot(t[m], fre[m] - ftrue[m], '^', color=c, ms=4, alpha=0.55)
        a.set_yscale('symlog', linthresh=max(piso, 1.0))
        a.set_ylabel('residuo contra la limpia [FLUXCAL]')
        a.set_title('Todas las bandas: cuanto se aparta el contrafactual',
                    fontsize=10)
        if i == 0:
            a.legend(frameon=False, fontsize=8, loc='upper left')
    axes[-1, 0].set_xlabel('dias desde la primera observacion')
    axes[-1, 1].set_xlabel('dias desde la primera observacion')

    fig.suptitle('Contrafactuales en espacio de flujo: el pico se deshace y el resto '
                 'de la curva queda dentro del ruido del inversor', fontsize=12)
    fig.text(0.5, 0.005, 'triangulo hacia abajo: base p95   triangulo hacia arriba: '
             'region contigua   |   40 SNIa del fold de validacion, ninguno visto en '
             'entrenamiento', ha='center', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / 'contrafactual_curvas.png', dpi=140)
    plt.close(fig)
    plt.close(fr)
    print(f'Guardado en {OUT}/contrafactual_curvas.png', flush=True)


if __name__ == '__main__':
    main()
