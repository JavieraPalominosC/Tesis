"""Ejemplos del prior MaskGIT sobre una clase excluida.

Muestra la imagen 2grid y el mapa de NLL por token de 10 objetos de la clase
elegida y de 10 normales, con el mismo rango de color, para ver donde se
concentra la sorpresa del prior. El puntaje de cada objeto es la NLL promedio
sobre la grilla, que es el estadistico que mejor funciona en estas clases.
"""

import os
import sys
import json
import random
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from scripts.eval_enc_cmp import mejor_ckpt
from scripts.eval_mascara_cruzada import mascara_ventana
from src.models.maskgit import MaskGIT

RAMA = os.environ.get('RAMA', 'mad300')
CLASE = os.environ.get('CLASE', 'SLSN-I_no_host')
N = int(os.environ.get('N_EJ', 10))
ROOT = f'data/enc_cmp/{RAMA}'
OUT = Path('results/enc_cmp_heldout/eval' + ('' if RAMA == 'mad300' else f'_{RAMA}'))
SALIDA = Path(os.environ.get('SALIDA_FIG', 'imagenes'))
N_RONDAS, RATIO, SEED = 24, 0.5, 0


def nll_mapa(mg, t, dev, g):
    """NLL por token promediada sobre N_RONDAS de ventanas contiguas."""
    B, T = t.shape
    n = int(T * RATIO)
    total = torch.zeros(B, T, device=dev)
    veces = torch.zeros(B, T, device=dev)
    for _ in range(N_RONDAS):
        mask = torch.cat([mascara_ventana(T, n, dev, g) for _ in range(B)])
        masked = t.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        total += (-lp.gather(2, t.unsqueeze(-1)).squeeze(-1)) * mask.float()
        veces += mask.float()
    return total / veces.clamp(min=1.0)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tabla = pd.read_csv(OUT / 'rasgos_curva.csv', dtype={'snid': str})
    puntajes = json.load(open(OUT / 'puntajes_prior_media_ven.json'))

    normales = [s for s in tabla.loc[tabla['grupo'] == 'normal', 'snid'] if s in puntajes]
    tau = float(np.percentile([puntajes[s] for s in normales], 95))

    anom = [s for s in tabla.loc[tabla['grupo'] == CLASE, 'snid'] if s in puntajes]
    if not anom:
        raise SystemExit(f'sin objetos de {CLASE}; hay {sorted(set(tabla["grupo"]))}')
    rng = random.Random(SEED)
    sel = {CLASE: rng.sample(anom, min(N, len(anom))),
           'normal': rng.sample(normales, min(N, len(normales)))}
    for k in sel:
        sel[k].sort(key=lambda s: -puntajes[s])
    print(f'tau (p95 normales) = {tau:.3f}')

    with open(f'{ROOT}/paths_heldout.txt') as f:
        paths = [l.strip() for l in f]
    tokens = np.load(f'{ROOT}/tokens_heldout.npy', mmap_mode='r')
    por_snid = {}
    for i, p in enumerate(paths):
        m = re.match(r'(\d+)', Path(p).stem)
        if m:
            por_snid.setdefault(m.group(1), (i, p))

    mg = MaskGIT.load_from_checkpoint(
        mejor_ckpt(f'results/enc_cmp_heldout/{RAMA}/maskgit/checkpoints/**/*.ckpt'),
        map_location=dev).to(dev).eval()
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    mapas, imgs = {}, {}
    with torch.no_grad():
        for grupo, snids in sel.items():
            idx = [por_snid[s][0] for s in snids]
            t = torch.from_numpy(np.asarray(tokens[idx])).long().to(dev)
            m = nll_mapa(mg, t, dev, g).cpu().numpy()
            lado = int(round(np.sqrt(m.shape[1])))
            for s, fila in zip(snids, m):
                mapas[s] = fila.reshape(lado, lado)
                imgs[s] = Image.open(por_snid[s][1]).convert('RGB')

    vmin, vmax = (np.percentile(np.stack(list(mapas.values())), q) for q in (2, 98))
    fig, axes = plt.subplots(4, N, figsize=(1.55 * N, 8.4))
    filas = [(CLASE, 0), ('normal', 2)]
    for grupo, r0 in filas:
        for c, s in enumerate(sel[grupo]):
            axes[r0, c].imshow(imgs[s])
            det = puntajes[s] > tau
            axes[r0, c].set_title(f'{puntajes[s]:.2f}  {"si" if det else "no"}',
                                  fontsize=8, color='#c0392b' if det else '#555555')
            im = axes[r0 + 1, c].imshow(mapas[s], cmap='magma', vmin=vmin, vmax=vmax)
            for r in (r0, r0 + 1):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        etiqueta = 'SLSN-I sin host' if grupo == CLASE else 'SNIa normal'
        axes[r0, 0].set_ylabel(f'{etiqueta}\nimagen', fontsize=9)
        axes[r0 + 1, 0].set_ylabel('sorpresa\ndel prior', fontsize=9)

    fig.suptitle(f'Prior MaskGIT sobre {CLASE}: NLL promedio por objeto, mascara de '
                 f'ventana\nel titulo es el puntaje y si supera el umbral '
                 f'tau = {tau:.2f} (percentil 95 de los normales)', fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93], h_pad=1.6)
    cax = fig.add_axes([0.25, 0.025, 0.5, 0.018])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='NLL por token')
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / f'maskgit_ejemplos_{CLASE}.png'
    fig.savefig(destino, dpi=150, bbox_inches='tight')
    print(f'escrito {destino}')


if __name__ == '__main__':
    main()
