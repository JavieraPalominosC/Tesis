"""Que miran los MaskGIT del subconjunto: localizacion de la anomalia.

Las metricas globales (recall, AUC) dicen si el NLL sube, pero no si sube en el
lugar correcto. Un modelo que se enciende entero cuando la imagen cambia da buen
recall sin haber localizado nada, y eso no sirve para explicarle a un astronomo
por que un objeto es raro.

Aca se mide si el maximo del mapa de NLL cae dentro de la zona donde realmente
metimos el pico: carril de la banda r, ultimo 30% del eje temporal.

La referencia que importa no es el azar sino el CONTROL. Si el maximo de la
imagen sin perturbar ya cae en esa zona, entonces acertar no significa haber
detectado el pico: significa que esa region es intrinsecamente la mas dificil de
predecir para esa codificacion. Por eso el control se mide igual que las
perturbadas y se reporta como linea base.

Uso: python scripts/analizar_maskgits.py [N_OBJETOS]
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

from scripts.eval_enc_cmp import (
    AMP, BASE_FILE, GRID, MANIFEST, EXCLUIR, MIN_TAIL, N_BANDS, RAW_DIR, STEP,
    TAIL_FRAC, TARGET_BAND, all_bands, col_band, col_flux, col_mjd, fig_params,
    mad, mejor_ckpt, nll_max, render,
)
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

RAMAS = ('actual', 'mad100', 'mad300')
DFS = [10, 50, 100, 200, 500]
OUT = Path('results/enc_cmp/eval')
PX = 256

NOMBRES = {'actual': 'actual (p01-p99 + clip)', 'mad100': 'mad100 (asinh/MAD, U=100)',
           'mad300': 'mad300 (asinh/MAD, U=300)'}
COLORES = {'actual': '#d7191c', 'mad100': '#2c7bb6', 'mad300': '#4C9F70'}


def zona_objetivo():
    """Rectangulo en coordenadas de token donde inyectamos el pico."""
    y_lo, y_hi = -0.2, (N_BANDS - 1) * STEP + AMP + 0.2
    j = all_bands[TARGET_BAND]

    def fila(y):
        return (y_hi - y) / (y_hi - y_lo) * PX

    r0 = int(fila(j * STEP + AMP) // (PX // GRID))
    r1 = int(fila(j * STEP) // (PX // GRID))

    def col(t):
        return (t + 0.05) / 1.10 * PX

    c0 = int(col(1.0 - TAIL_FRAC) // (PX // GRID))
    c1 = int(col(1.0) // (PX // GRID))
    return max(r0, 0), min(r1, GRID - 1), max(c0, 0), min(c1, GRID - 1)


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)

    r0, r1, c0, c1 = zona_objetivo()
    n_zona = (r1 - r0 + 1) * (c1 - c0 + 1)
    azar = n_zona / (GRID * GRID)
    print(f'zona objetivo: filas {r0}-{r1}, columnas {c0}-{c1}  '
          f'({n_zona} de {GRID * GRID} tokens, azar {azar:.1%})', flush=True)

    with open(MANIFEST) as f:
        man = json.load(f)
    snids = [r['snid'] for r in man if r.get('tag') == 'control']
    snids = [s for s in dict.fromkeys(snids) if s not in EXCLUIR]
    if lim:
        snids = snids[:lim]

    df_all = pd.read_parquet(f'{RAW_DIR}/{BASE_FILE}')
    df_all['SNID'] = df_all['SNID'].astype(str)
    df_all = df_all[df_all['SNID'].isin(snids)]

    curvas = {}
    for s in snids:
        o = df_all[df_all['SNID'] == s].copy()
        o[col_flux] = o[col_flux].astype('float64')
        sc = mad(o.loc[o[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(sc) or sc <= 0:
            continue
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        t0, t1 = float(o[col_mjd].min()), float(o[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])
        curvas[s] = (o, idx, sc)
    print(f'objetos: {len(curvas)}', flush=True)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    acierto = {r: {} for r in RAMAS}
    contraste = {r: {} for r in RAMAS}
    base = {r: {} for r in RAMAS}
    ejemplos = {}
    snid_demo = list(curvas)[0]

    def medir(mapa):
        rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
        dentro = mapa[r0:r1 + 1, c0:c1 + 1]
        fuera = np.delete(mapa.reshape(-1),
                          np.ravel_multi_index(
                              np.meshgrid(np.arange(r0, r1 + 1),
                                          np.arange(c0, c1 + 1), indexing='ij'),
                              mapa.shape).reshape(-1))
        return (bool(r0 <= rr <= r1 and c0 <= cc <= c1),
                float(dentro.max() - np.percentile(fuera, 99)))

    for rama in RAMAS:
        vq = VQVAE.load_from_checkpoint(
            mejor_ckpt(f'results/enc_cmp/{rama}/vqvae/checkpoints/**/*.ckpt'),
            map_location=dev).eval().to(dev)
        mg = MaskGIT.load_from_checkpoint(
            mejor_ckpt(f'results/enc_cmp/{rama}/maskgit/checkpoints/**/*.ckpt'),
            map_location=dev, strict=False).eval().to(dev)
        print(f'\n[{rama}]', flush=True)

        hits, contr = [], []
        for s, (o, idx, sc) in curvas.items():
            _, mapa = nll_max(vq, mg, render(o, rama, fig, ax), dev)
            h, c = medir(mapa)
            hits.append(h)
            contr.append(c)
            if s == snid_demo:
                ejemplos[(rama, 0)] = mapa
        base[rama] = {'acierto': float(np.mean(hits)),
                      'contraste': float(np.median(contr))}
        print(f'  control  acierto={base[rama]["acierto"]:.2f}  '
              f'contraste={base[rama]["contraste"]:+.2f}   <- linea base', flush=True)

        for d in DFS:
            hits, contr = [], []
            for s, (o, idx, sc) in curvas.items():
                p = o.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
                _, mapa = nll_max(vq, mg, render(p, rama, fig, ax), dev)
                h, c = medir(mapa)
                hits.append(h)
                contr.append(c)
                if s == snid_demo and d in (50, 500):
                    ejemplos[(rama, d)] = mapa
            acierto[rama][d] = float(np.mean(hits))
            contraste[rama][d] = float(np.median(contr))
            print(f'  df{d:<4}    acierto={acierto[rama][d]:.2f}  '
                  f'contraste={contraste[rama][d]:+.2f}  '
                  f'(gana {acierto[rama][d] - base[rama]["acierto"]:+.2f} sobre el control)',
                  flush=True)

        del vq, mg
        torch.cuda.empty_cache()

    plt.close(fig)

    with open(OUT / 'localizacion.json', 'w') as f:
        json.dump({'zona': [r0, r1, c0, c1], 'azar': azar, 'dfs': DFS,
                   'n_objetos': len(curvas), 'acierto': acierto,
                   'contraste': contraste, 'control': base}, f, indent=2)

    graficar(acierto, contraste, azar, base)
    mapas(ejemplos, (r0, r1, c0, c1), snid_demo)
    print(f'\nListo en {OUT}/', flush=True)


def graficar(acierto, contraste, azar, base):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for rama in RAMAS:
        a1.plot(DFS, [acierto[rama][d] for d in DFS], 'o-', lw=2,
                color=COLORES[rama], label=NOMBRES[rama])
        a2.plot(DFS, [contraste[rama][d] for d in DFS], 'o-', lw=2,
                color=COLORES[rama], label=NOMBRES[rama])
        # Sin esta referencia el acierto no dice nada: si el control ya cae en la
        # zona, la codificacion no esta localizando el pico sino su propio ruido.
        a1.axhline(base[rama]['acierto'], color=COLORES[rama], ls=':', lw=1.4)
        a2.axhline(base[rama]['contraste'], color=COLORES[rama], ls=':', lw=1.4)
    a1.axhline(azar, color='gray', ls='--', lw=1)
    a1.text(DFS[0], azar + 0.02, f'azar ({azar:.0%})', fontsize=8, color='gray')
    a1.set_ylim(0, 1.02)
    a1.set_ylabel('fraccion de objetos')
    a1.set_title('El maximo del NLL cae donde metimos el pico\n'
                 'punteado: mismo objeto SIN perturbar', fontsize=10)
    a2.axhline(0, color='gray', ls=':', lw=1)
    a2.set_ylabel('NLL max en la zona  -  P99 fuera de la zona')
    a2.set_title('Cuanto sobresale la zona del resto de la imagen\n'
                 'punteado: mismo objeto SIN perturbar', fontsize=10)
    for a in (a1, a2):
        a.set_xscale('log')
        a.set_xticks(DFS, [str(d) for d in DFS])
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'localizacion.png', dpi=140)
    plt.close(fig)


def mapas(ej, zona, snid):
    r0, r1, c0, c1 = zona
    fig, axes = plt.subplots(len(RAMAS), 3, figsize=(10.5, 9.5))
    for i, rama in enumerate(RAMAS):
        vals = [ej[(rama, k)] for k in (0, 50, 500) if (rama, k) in ej]
        vmin, vmax = min(v.min() for v in vals), max(v.max() for v in vals)
        for j, (k, tit) in enumerate([(0, 'control'), (50, 'df=50'), (500, 'df=500')]):
            a = axes[i, j]
            m = ej.get((rama, k))
            if m is None:
                a.axis('off')
                continue
            im = a.imshow(m, cmap='inferno', vmin=vmin, vmax=vmax)
            a.add_patch(mpatches.Rectangle((c0 - .5, r0 - .5), c1 - c0 + 1, r1 - r0 + 1,
                                           fill=False, edgecolor='cyan', lw=1.6))
            rr, cc = np.unravel_index(np.argmax(m), m.shape)
            a.plot(cc, rr, 'x', color='white', ms=9, mew=2)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f'{tit}   max={m.max():.1f}', fontsize=9)
            if j == 0:
                a.set_ylabel(NOMBRES[rama], fontsize=8)
            plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f'Mapas de NLL del objeto {snid}\n'
                 'recuadro celeste: zona con el pico inyectado    x blanca: maximo del mapa',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'mapas_nll.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
