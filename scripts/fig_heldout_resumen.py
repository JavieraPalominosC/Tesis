"""Figura resumen de la deteccion sobre clases excluidas.

Cubre los puntos 2 y 5 de la reunion, que comparten montaje: puntuar las seis
clases que quedaron fuera del entrenamiento y comparar el prior contra
alternativas mas simples.

Panel A: AUROC por clase y por metodo, con la linea de azar.
Panel B: efecto de renderizar con eje temporal absoluto, por clase.

Uso: python scripts/fig_heldout_resumen.py
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('results/enc_cmp_heldout')
OUT = Path('imagenes')

METODOS = [
    ('mad_max', 'Desviación máxima en MAD', '#bdbdbd'),
    ('iforest', 'Isolation Forest', '#9ecae1'),
    ('recon', 'Error de reconstrucción (VQ-VAE)', '#fdae6b'),
    ('prior_media_ven', 'Prior MaskGIT (el método)', '#31a354'),
]
# Nombres cortos para que quepan en el eje.
CORTO = {
    'SLSN-I_no_host': 'SLSN-I\nsin host',
    'SLSN-I+host': 'SLSN-I\ncon host',
    'PISN': 'PISN',
    'SNIa-91bg': 'SNIa-91bg',
    'SNIax': 'SNIax',
    'SNIIb+HostXT_V19': 'SNIIb',
}


def main():
    a = json.load(open(BASE / 'eval' / 'baselines_auroc.json'))
    b = json.load(open(BASE / 'eval_mad300t' / 'baselines_auroc.json'))

    # Orden por desempeno del metodo, que es el que hace legible la division
    # entre clases exoticas y clases sutiles.
    clases = sorted(a['por_clase'],
                    key=lambda c: -a['por_clase'][c]['prior_media_ven'])
    x = np.arange(len(clases))
    ancho = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    for k, (clave, etiqueta, color) in enumerate(METODOS):
        v = [a['por_clase'][c][clave] for c in clases]
        ax1.bar(x + (k - 1.5) * ancho, v, ancho, label=etiqueta, color=color,
                edgecolor='white', linewidth=0.6)
    ax1.axhline(0.5, color='k', ls='--', lw=1.2)
    ax1.text(len(clases) - 0.45, 0.512, 'azar', fontsize=9, ha='right')
    ax1.axvspan(-0.5, 2.5, color='#fff3e0', zorder=0)
    ax1.text(1.0, 0.95, 'morfológicamente exóticas', fontsize=9, ha='center',
             style='italic', color='#8c6d31')
    ax1.text(4.0, 0.95, 'confundibles con las vistas', fontsize=9, ha='center',
             style='italic', color='#777777')
    ax1.set_xticks(x, [CORTO[c] for c in clases], fontsize=9)
    ax1.set_ylabel('AUROC contra las clases vistas')
    ax1.set_ylim(0.25, 1.0)
    ax1.set_xlim(-0.5, len(clases) - 0.5)
    ax1.set_title('A. Detección de las seis clases excluidas del entrenamiento',
                  fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, -0.1))

    va = [a['por_clase'][c]['prior_media_ven'] for c in clases]
    vb = [b['por_clase'][c]['prior_media_ven'] for c in clases]
    ax2.hlines(x, va, vb, color='#999999', lw=1.6, zorder=1)
    ax2.scatter(va, x, s=62, color='#31a354', zorder=2, label='eje de tiempo por objeto')
    ax2.scatter(vb, x, s=62, color='#08519c', zorder=2, label='eje de tiempo absoluto')
    for i, (p, q) in enumerate(zip(va, vb)):
        if abs(q - p) > 0.015:
            ax2.annotate(f'{q - p:+.3f}', (max(p, q) + 0.014, i),
                         va='center', fontsize=9, color='#08519c', weight='bold')
    # La duracion en dias, sola, era el baseline que superaba al pipeline con el
    # eje viejo. Con el eje absoluto el pipeline la pasa, que es el punto.
    ax2.plot([0.895, 0.895], [-0.34, 0.34], color='#d7191c', lw=1.8, zorder=3)
    ax2.annotate('0.895: duración en días,\nsin modelo alguno',
                 xy=(0.895, 0.36), xytext=(0.70, 1.15), fontsize=8.5,
                 color='#d7191c', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#d7191c', lw=1))
    ax2.axvline(0.5, color='k', ls='--', lw=1.2)
    ax2.set_yticks(x, [CORTO[c].replace('\n', ' ') for c in clases], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('AUROC')
    ax2.set_xlim(0.40, 1.02)
    ax2.set_ylim(len(clases) - 0.4, -0.75)
    ax2.set_title('B. Efecto de conservar la duración en la imagen', fontsize=11)
    ax2.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, -0.1))

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'heldout_resumen.png', dpi=150)
    plt.close(fig)

    print('promedio prior: %.3f -> %.3f' % (a['promedio']['prior_media_ven'],
                                            b['promedio']['prior_media_ven']))
    print('Listo:', OUT / 'heldout_resumen.png')


if __name__ == '__main__':
    main()
