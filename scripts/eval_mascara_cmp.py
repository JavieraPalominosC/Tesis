"""Enmascarado aleatorio contra ventana contigua, en deteccion.

Los dos MaskGIT comparten VQ-VAE, imagenes y datos de entrenamiento; lo unico
que cambia es la forma de la mascara durante el entrenamiento. Sus val/loss NO
se pueden comparar (4.22 contra 3.63) porque reconstruir un bloque contiguo es
una tarea mas dificil por definicion: no quedan vecinos alrededor del hueco. La
comparacion honesta es esta, sobre la tarea que nos importa.

Ojo con una limitacion heredada: estos modelos se entrenaron sobre las 1.39M
imagenes, que incluyen a los 40 objetos de evaluacion en su version limpia. Eso
infla el desempeno de los dos por igual, asi que la comparacion entre ellos vale,
pero los valores absolutos no son un estimador honesto de generalizacion.

Uso: python scripts/eval_mascara_cmp.py
"""
import sys
sys.path.insert(0, '.')
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from scripts.eval_enc_cmp import GRID, auc, nll_max
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

VQVAE_CKPT = ('results/vqvae_waterfall_global_full/checkpoints/'
              'vqvae-epoch=26-train/loss=0.0597.ckpt')
MODELOS = {
    'aleatoria': ('results/maskgit_full/waterfall_global/checkpoints/'
                  'maskgit-epoch=20-val/loss=3.6256.ckpt'),
    'ventana': ('results/maskgit_window/waterfall_global/checkpoints/'
                'maskgit-window-epoch=20-val/loss=4.2217.ckpt'),
}
MANIFEST = 'data/synthetic_mad_tail_manifest.json'
OUT = Path('results/enc_cmp/eval')
DFS = list(range(10, 101, 10))
CTRL_Q = 0.95
COLORES = {'aleatoria': '#8856a7', 'ventana': '#e6550d'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r0, r1, c0, c1 = zona_objetivo()
    print(f'zona objetivo: filas {r0}-{r1}, columnas {c0}-{c1}', flush=True)

    with open(MANIFEST) as f:
        man = [r for r in json.load(f) if not r.get('frozen')]
    por_snid = collections.defaultdict(dict)
    for r in man:
        p = Path(r['path'])
        if p.exists():
            por_snid[r['snid']][r['tag']] = p
    snids = [s for s, d in por_snid.items()
             if 'control' in d and all(f'df{k}_unfrozen' in d for k in DFS)]
    print(f'objetos completos: {len(snids)}', flush=True)

    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev).eval().to(dev)

    res = {}
    for nombre, ckpt in MODELOS.items():
        mg = MaskGIT.load_from_checkpoint(ckpt, map_location=dev, strict=False).eval().to(dev)
        print(f'\n[{nombre}]', flush=True)

        ctrl, pert = {}, {d: {} for d in DFS}
        loc = {d: [] for d in DFS}
        loc_ctrl = []
        for s in snids:
            v, mapa = nll_max(vq, mg, Image.open(por_snid[s]['control']).convert('RGB'), dev)
            ctrl[s] = v
            rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
            loc_ctrl.append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))
            for d in DFS:
                v, mapa = nll_max(
                    vq, mg, Image.open(por_snid[s][f'df{d}_unfrozen']).convert('RGB'), dev)
                pert[d][s] = v
                rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
                loc[d].append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))

        cs = np.array([ctrl[s] for s in snids])
        tau = float(np.quantile(cs, CTRL_Q))
        m = {'tau': tau, 'loc_control': float(np.mean(loc_ctrl)), 'por_df': {}}
        for d in DFS:
            ps = np.array([pert[d][s] for s in snids])
            m['por_df'][d] = {
                'recall': float((ps > tau).mean()),
                'auc': auc(cs, ps),
                'pareada': float(np.mean([pert[d][s] > ctrl[s] for s in snids])),
                'localizacion': float(np.mean(loc[d])),
            }
        rhos = [spearmanr(DFS, [pert[d][s] for d in DFS]).statistic for s in snids]
        rhos = [r for r in rhos if np.isfinite(r)]
        m['monotonia'] = float(np.median(rhos))
        res[nombre] = m

        print(f'  tau={tau:.3f}  monotonia={m["monotonia"]:+.3f}  '
              f'localizacion control={m["loc_control"]:.2f}', flush=True)
        for d in DFS:
            x = m['por_df'][d]
            print(f'    df{d:<4} recall={x["recall"]:.2f}  auc={x["auc"]:.3f}  '
                  f'pareada={x["pareada"]:.2f}  loc={x["localizacion"]:.2f}', flush=True)
        del mg
        torch.cuda.empty_cache()

    with open(OUT / 'mascara_cmp.json', 'w') as f:
        json.dump({'dfs': DFS, 'n_objetos': len(snids), 'ramas': res}, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, clave, tit in zip(axes, ('recall', 'auc', 'localizacion'),
                             ('Recall con tau = P95 de los controles',
                              'AUC control contra perturbada',
                              'El maximo del NLL cae donde metimos el pico')):
        for nombre, m in res.items():
            a.plot(DFS, [m['por_df'][d][clave] for d in DFS], 'o-', lw=2,
                   color=COLORES[nombre], label=f'mascara {nombre}')
        if clave == 'localizacion':
            for nombre, m in res.items():
                a.axhline(m['loc_control'], color=COLORES[nombre], ls=':', lw=1.4)
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel(clave)
        a.set_title(tit, fontsize=10)
        a.set_ylim(0, 1.02)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle('Enmascarado aleatorio contra ventana contigua  '
                 '(mismo VQ-VAE, mismas imagenes, codificacion antigua)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'mascara_cmp.png', dpi=140)
    plt.close(fig)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
