"""Cuanto del mapa de NLL es dificultad comun y cuanto distingue a la clase.

El mapa de sorpresa de un objeto cualquiera esta moteado de tokens caros, pero
ese moteado se repite en objetos normales y anomalos: son posiciones de la
grilla que el prior predice mal siempre. Esta figura separa las dos cosas.
Calcula la mediana por token sobre los normales, que es el patron de dificultad
basal, y resta ese patron a cada objeto. Lo que queda es la sorpresa atribuible
al objeto y no a la posicion.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.eval_enc_cmp import mejor_ckpt
from scripts.fig_maskgit_ejemplos import nll_mapa
from src.models.maskgit import MaskGIT

RAMA = os.environ.get('RAMA', 'mad300')
CLASE = os.environ.get('CLASE', 'SLSN-I_no_host')
N_POB = int(os.environ.get('N_POB', 200))
ROOT = f'data/enc_cmp/{RAMA}'
OUT = Path('results/enc_cmp_heldout/eval' + ('' if RAMA == 'mad300' else f'_{RAMA}'))
SALIDA = Path(os.environ.get('SALIDA_FIG', 'imagenes'))
LOTE, SEED = 64, 0


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tabla = pd.read_csv(OUT / 'rasgos_curva.csv', dtype={'snid': str})

    with open(f'{ROOT}/paths_heldout.txt') as f:
        paths = [l.strip() for l in f]
    tokens = np.load(f'{ROOT}/tokens_heldout.npy', mmap_mode='r')
    por_snid = {}
    for i, p in enumerate(paths):
        m = re.match(r'(\d+)', Path(p).stem)
        if m:
            por_snid.setdefault(m.group(1), i)

    rng = random.Random(SEED)
    grupos = {}
    for gr in ('normal', CLASE):
        s = [x for x in tabla.loc[tabla['grupo'] == gr, 'snid'] if x in por_snid]
        grupos[gr] = rng.sample(s, min(N_POB, len(s)))

    mg = MaskGIT.load_from_checkpoint(
        mejor_ckpt(f'results/enc_cmp_heldout/{RAMA}/maskgit/checkpoints/**/*.ckpt'),
        map_location=dev).to(dev).eval()
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    pila = {}
    with torch.no_grad():
        for gr, snids in grupos.items():
            trozos = []
            for a in range(0, len(snids), LOTE):
                idx = [por_snid[s] for s in snids[a:a + LOTE]]
                t = torch.from_numpy(np.asarray(tokens[idx])).long().to(dev)
                trozos.append(nll_mapa(mg, t, dev, g).cpu().numpy())
            m = np.concatenate(trozos)
            lado = int(round(np.sqrt(m.shape[1])))
            pila[gr] = m.reshape(-1, lado, lado)
            print(f'{gr:20s} {len(m):4d} objetos')

    basal = np.median(pila['normal'], axis=0)
    med_an = np.median(pila[CLASE], axis=0)
    dif = med_an - basal

    # Cuanta de la variacion espacial de un objeto es el patron basal y no el objeto.
    plano = pila[CLASE].reshape(len(pila[CLASE]), -1)
    r = np.array([np.corrcoef(f, basal.ravel())[0, 1] for f in plano])
    plano_n = pila['normal'].reshape(len(pila['normal']), -1)
    r_n = np.array([np.corrcoef(f, basal.ravel())[0, 1] for f in plano_n])
    print(f'corr con el patron basal: {CLASE} {r.mean():.3f}, normales {r_n.mean():.3f}')
    print(f'amplitud del basal {basal.std():.3f}, de la diferencia {dif.std():.3f}')

    # Fila de token -> banda. El renderizador apila la banda k centrada en
    # k*STEP + AMP/2 sobre un eje que va de -0.2 a (N_BANDS-1)*STEP + AMP + 0.2,
    # y la imagen se lee de arriba hacia abajo, asi que u queda abajo.
    lado = basal.shape[0]
    amp, paso, bandas = 1.8, 2.0, ['u', 'g', 'r', 'i', 'z', 'Y']
    y0, y1 = -0.2, (len(bandas) - 1) * paso + amp + 0.2
    filas = [(y1 - (k * paso + amp / 2)) / (y1 - y0) * lado for k in range(len(bandas))]
    perfil = dif.mean(1)
    k_max = int(np.argmin([abs(f - np.argmax(perfil)) for f in filas]))
    print(f'fila con mayor diferencia: {np.argmax(perfil)}, banda {bandas[k_max]}')
    print('perfil por banda: ' + ', '.join(
        f'{b} {perfil[int(round(f)) % lado]:+.2f}' for b, f in zip(bandas, filas)))

    def ejes(ax):
        ax.set_xticks([])
        ax.set_yticks(filas)
        ax.set_yticklabels(bandas, fontsize=8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    vb = (min(basal.min(), med_an.min()), max(basal.max(), med_an.max()))
    for ax, (dat, tit) in zip(axes[:2], [
            (basal, f'mediana de {len(pila["normal"])} normales'),
            (med_an, f'mediana de {len(pila[CLASE])} {CLASE}')]):
        im = ax.imshow(dat, cmap='magma', vmin=vb[0], vmax=vb[1])
        ax.set_title(tit + '\nel moteado individual desaparece', fontsize=10)
        ejes(ax)
    fig.colorbar(im, ax=axes[:2], fraction=0.03, label='NLL por token')

    v = float(np.percentile(np.abs(dif), 99))
    im2 = axes[2].imshow(dif, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[2].set_title(f'diferencia\nsigma {dif.std():.2f} contra {basal.std():.2f} del basal',
                      fontsize=10)
    ejes(axes[2])
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].hist(r_n, bins=30, alpha=0.6, color='#2c3e50', label='normales')
    axes[3].hist(r, bins=30, alpha=0.6, color='#c0392b', label=CLASE)
    axes[3].set_xlabel('correlacion del mapa del objeto con el patron basal')
    axes[3].set_ylabel('objetos')
    axes[3].set_title('la forma del mapa ya separa las clases', fontsize=10)
    axes[3].legend(fontsize=8)

    fig.suptitle('El moteado de los mapas individuales no es el patron: se promedia y '
                 'desaparece.\nLo que distingue a la clase es una diferencia suave y '
                 'localizada en banda, mucho mas chica que el moteado.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / f'maskgit_residual_{CLASE}.png'
    fig.savefig(destino, dpi=150, bbox_inches='tight')
    print(f'escrito {destino}')


if __name__ == '__main__':
    main()
