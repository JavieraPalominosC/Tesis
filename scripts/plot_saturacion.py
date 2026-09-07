"""
Por que la deteccion no llega al 100% aunque el pico crezca sin limite.

La respuesta no esta en el MaskGIT sino antes, en como se dibuja la imagen.
El flujo se normaliza con p01-p99 y se recorta:

    flux_norm = ((flux - p01) / rng).clip(0.0, 1.0)

  - Escala CONGELADA: p01-p99 se fijan en el control. Cualquier flujo por
    encima de p99 se recorta a 1.0, o sea al mismo pixel. Subir el pico de
    20 a 200 MAD da una imagen IDENTICA.

  - Escala LIBRE: p01-p99 se recalculan sobre la curva perturbada, asi que
    el pico siempre queda arriba del panel y lo que cambia es que el resto
    se aplana. Ese aplanamiento se agota rapido.

En los dos casos la imagen deja de cambiar mucho antes de que el flujo deje
de crecer. Por eso la deteccion se estanca.

Uso: python scripts/plot_saturacion.py
     PROFE_FIG_OUT=/tmp/imagenes python scripts/plot_saturacion.py
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAIL_DIR = Path('data/images/elasticc_1/synthetic_mad_tail')
DT_DIR = Path('data/images/elasticc_1/synthetic_mad_dt_wfg')
TAIL_SUM = Path('results/synthetic_eval/mad_tail/summary.json')
DT_SUM = Path('results/synthetic_eval/mad_dt_wfg/summary.json')
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
DFS = list(range(10, 101, 10))
DFS_FROZEN = [1, 2, 5, 10, 20, 50, 100, 200]
DT_REF = 30
UMBRAL_GRIS = 8


def grey(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32)


def frac_dist(a, b):
    return float((np.abs(a - b) > UMBRAL_GRIS).mean() * 100.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tail = json.load(open(TAIL_SUM))
    snids = sorted({r['snid'] for r in tail['per_image'] if r['tag'] == 'control'})
    recall = np.array([tail['per_tag'][f'df{d}_unfrozen']['max_detector']['recall']
                       for d in DFS])

    # --- escala libre: cuanto cambia la imagen respecto del control ---------
    curvas = []
    for s in snids:
        c = TAIL_DIR / f'{s}__control.png'
        paths = [TAIL_DIR / f'{s}__df{d}_unfrozen.png' for d in DFS]
        if not c.exists() or not all(p.exists() for p in paths):
            continue
        base = grey(c)
        curvas.append([frac_dist(grey(p), base) for p in paths])
    curvas = np.array(curvas)
    med = np.median(curvas, axis=0)
    q1, q3 = np.percentile(curvas, [25, 75], axis=0)
    print(f'escala libre: {len(curvas)} objetos', flush=True)
    for d, m in zip(DFS, med):
        print(f'  df={d:3d}  pixeles cambiados vs control: {m:5.2f}%', flush=True)

    # --- escala congelada: pares de niveles con imagen identica -------------
    cong, identicos, n_obj_f = [], 0, 0
    for s in snids:
        c = DT_DIR / f'{s}__control.png'
        paths = [DT_DIR / f'{s}__df{d}_dt{DT_REF}.png' for d in DFS_FROZEN]
        if not c.exists() or not all(p.exists() for p in paths):
            continue
        base = grey(c)
        ims = [grey(p) for p in paths]
        cong.append([frac_dist(im, base) for im in ims])
        n_obj_f += 1
        identicos += sum(np.array_equal(ims[i - 1], ims[i]) for i in range(1, len(ims)))
    cong = np.array(cong) if cong else None
    if cong is not None:
        n_pares = n_obj_f * (len(DFS_FROZEN) - 1)
        print(f'escala congelada: {n_obj_f} objetos, '
              f'{identicos}/{n_pares} pares de niveles vecinos dan imagen identica',
              flush=True)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.9))

    ax.plot(DFS, med, 'o-', color='#2c7bb6', lw=2, label='imagen: pixeles que cambian')
    ax.fill_between(DFS, q1, q3, color='#2c7bb6', alpha=0.18)
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_ylabel('% de pixeles distintos del control', color='#2c7bb6')
    ax.tick_params(axis='y', labelcolor='#2c7bb6')
    ax.set_ylim(0, max(q3) * 1.25)
    axr = ax.twinx()
    axr.plot(DFS, recall, 's--', color='darkorange', lw=2, label='recall')
    axr.set_ylabel('recall del detector', color='darkorange')
    axr.tick_params(axis='y', labelcolor='darkorange')
    axr.set_ylim(0, 1.05)
    ax.set_title('Escala libre: la imagen se satura y el recall con ella\n'
                 f'mediana sobre {len(curvas)} SNIa', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='lower right')

    if cong is not None:
        medc = np.median(cong, axis=0)
        ax2.plot(DFS_FROZEN, medc, 'o-', color='#d7191c', lw=2)
        ax2.fill_between(DFS_FROZEN, *np.percentile(cong, [25, 75], axis=0),
                         color='#d7191c', alpha=0.18)
        ax2.set_xscale('log')
        ax2.set_xticks(DFS_FROZEN, [str(d) for d in DFS_FROZEN])
        ax2.set_ylim(0, max(medc) * 2.2)
        ax2.set_xlabel(r'$df$  [veces MAD$_r$],  $\Delta t$ = 30 d')
        ax2.set_ylabel('% de pixeles distintos del control')
        ax2.set_title('Escala congelada: el clip a 1.0 borra todo\n'
                      f'{identicos} de {n_obj_f * (len(DFS_FROZEN) - 1)} pares vecinos '
                      'dan imagenes identicas', fontsize=10)
        ax2.annotate('de aca en adelante\nla imagen ya no cambia',
                     xy=(20, medc[DFS_FROZEN.index(20)]),
                     xytext=(30, max(medc) * 1.6), fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='#555555'))

    fig.tight_layout()
    fig.savefig(OUT / 'saturacion_imagen.png', dpi=140)
    plt.close(fig)

    meta = {
        'umbral_gris': UMBRAL_GRIS,
        'escala_libre_pct_pixeles': dict(zip(map(str, DFS), map(float, med))),
        'recall': dict(zip(map(str, DFS), map(float, recall))),
        'escala_congelada_pct_pixeles': (dict(zip(map(str, DFS_FROZEN),
                                                  map(float, np.median(cong, axis=0))))
                                         if cong is not None else None),
        'pares_identicos_congelado': identicos,
    }
    with open(OUT / 'saturacion.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Figura en {OUT}/saturacion_imagen.png', flush=True)


if __name__ == '__main__':
    main()
