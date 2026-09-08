"""Figura cualitativa final: los tres protocolos sobre los mismos objetos.

La version anterior de esta grilla usaba df=100 e incluia la variante con bucle,
que a ese nivel no se dispara: la columna salia identica a la perturbada y el
mapa de diferencia todo negro. Es un retrato honesto de esa limitacion, pero no
sirve para mostrar el metodo. Aca se usa df=500, donde los tres protocolos si
actuan, y se compara lo que de verdad esta en disputa: tokens dispersos (p85),
presupuesto chico disperso (p95) y region contigua adaptativa (rect).

Uso: python scripts/contrafactual_figura.py
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from scripts.eval_enc_cmp import col_flux, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import reconstruir, zona_px
from scripts.contrafactual_comparacion import metodo_base, metodo_rect
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DF, N_MUESTRA, SEED = 500, 5, 0


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pz = zona_px(*zona_objetivo())
    curvas = candidatos(40)
    snids = sorted(curvas)[:N_MUESTRA]

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
        p = o.copy()
        p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + DF * sc
        imgs[(s, 'pert')] = render(p, ENC, fig, ax)
    plt.close(fig)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def codificar(img):
        x = tf(img).unsqueeze(0).to(dev)
        t = vq.encode_to_indices(x).view(1, -1).long().to(dev)
        return t, reconstruir(vq, t)

    def a_img(t):
        return np.clip((t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)

    cols = ['curva limpia', f'con pico df={DF}', 'base p85\n(protocolo)',
            'base p95', 'region contigua', r'$|\hat{x}-\hat{x}\,^\prime|$  p85',
            r'$|\hat{x}-\hat{x}\,^\prime|$  contigua']
    fig, axes = plt.subplots(len(snids), 7, figsize=(21, 3.15 * len(snids)))

    for i, s in enumerate(snids):
        tp, xp = codificar(imgs[(s, 'pert')])
        _, xl = codificar(imgs[(s, 'limpia')])
        c85, _ = metodo_base(mg, tp, dev, g, 0.85)
        c95, _ = metodo_base(mg, tp, dev, g, 0.95)
        cr, ir = metodo_rect(mg, tp, dev, g)
        x85, x95, xr = (reconstruir(vq, c) for c in (c85, c95, cr))
        d85 = (xp - x85).abs().sum(dim=1)[0].cpu().numpy()
        dr = (xp - xr).abs().sum(dim=1)[0].cpu().numpy()

        paneles = [a_img(xl), a_img(xp), a_img(x85), a_img(x95), a_img(xr), d85, dr]
        pies = ['', '', f'{int((c85 != tp).sum())} tokens',
                f'{int((c95 != tp).sum())} tokens',
                f'{int((cr != tp).sum())} tokens, area {int(ir["area_rect"])}', '', '']
        for j, (im, pie) in enumerate(zip(paneles, pies)):
            a = axes[i, j]
            a.imshow(im, cmap=None if im.ndim == 3 else 'magma')
            a.add_patch(mpatches.Rectangle((pz[2], pz[0]), pz[3] - pz[2],
                                           pz[1] - pz[0], fill=False,
                                           edgecolor='#39ff14', lw=1.4))
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=9.5)
            if pie:
                a.set_xlabel(pie, fontsize=8)
        axes[i, 0].set_ylabel(f'SNID {s}', fontsize=8)

    fig.suptitle('Tres protocolos de contrafactual sobre los mismos objetos '
                 f'(df={DF}, codificacion mad300)\n'
                 'recuadro verde: zona donde inyectamos el pico', fontsize=12)
    fig.text(0.5, 0.004, '40 SNIa del fold de validacion, ninguno visto en '
             'entrenamiento por el VQ-VAE ni por el MaskGIT', ha='center',
             fontsize=8.5, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.015, 1, 0.94])
    fig.savefig(OUT / 'contrafactual_lado_a_lado.png', dpi=140)
    plt.close(fig)
    print(f'Guardado en {OUT}/contrafactual_lado_a_lado.png', flush=True)


if __name__ == '__main__':
    main()
