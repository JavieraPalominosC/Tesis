"""Figura de la calibracion del extractor de trazas y sus consecuencias.

Lee imagenes/contrafactual_extractor_resumen.json, producido por
scripts/contrafactual_extractor.py, y no recalcula nada.
"""

import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, 'imagenes', 'contrafactual_extractor_resumen.json')
SALIDA = os.path.join(RAIZ, 'imagenes', 'contrafactual_extractor.png')

DFS = [10, 50, 100, 300, 500]
METODOS = [
    ('base p95', 'base p95', '#c0392b', 'o'),
    ('rect 1 pasa', 'region contigua', '#1f77b4', 's'),
    ('azar', 'azar (control)', '#7f8c8d', '^'),
]


def serie(d, bloque, metodo, campo, umbral=None):
    ys = []
    for df in DFS:
        clave = f'{df}|{metodo}' if umbral is None else f'{umbral}|{df}|{metodo}'
        v = d[bloque].get(clave)
        ys.append(np.nan if v is None else v[campo])
    return np.array(ys, float)


def panel_calibracion(ax, d):
    cal = d['calibracion']
    claves = sorted(d['barrido'], key=float)
    ths = [float(k) for k in claves]
    borr = [100 * d['barrido'][k]['borrado'] for k in claves]

    ax.plot(ths, borr, 'o-', color='#2c3e50', lw=1.6, ms=4)
    ax.axvline(cal['min'], color='#27ae60', ls='--', lw=1.4)
    ax.axvline(d['umbral'], color='#c0392b', ls='-', lw=1.8)
    ax.axvline(d['umbral_viejo'], color='#e67e22', ls=':', lw=1.8)

    ax.annotate(f"minimo limpio\n{cal['min']:.3f}", xy=(cal['min'], 0.5),
                xytext=(cal['min'] + 0.12, 3.0), fontsize=7.5, color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=0.9))
    ax.text(d['umbral'], 9.5, f"calibrado\n{d['umbral']}", fontsize=7.5,
            color='#c0392b', ha='center', va='top', fontweight='bold')
    ax.text(d['umbral_viejo'], 9.5, f"previo\n{d['umbral_viejo']}", fontsize=7.5,
            color='#e67e22', ha='center', va='top')

    ax.set_xlabel('umbral de tinta')
    ax.set_ylabel('borrado falso sobre curvas limpias (%)')
    ax.set_title('A. Calibracion: el umbral mas alto sin borrado falso', fontsize=9.5)
    ax.set_ylim(-0.4, 11)
    ax.grid(alpha=0.25)


def panel_tinta(ax, d):
    cal = d['calibracion']
    for metodo, etiq, color, marca in METODOS:
        ax.plot(DFS, serie(d, 'tinta', metodo, 'mediana'), marker=marca,
                color=color, lw=1.6, ms=5, label=etiq)

    pert = serie(d, 'tinta', 'base p95', 'pert_mediana')
    ax.plot(DFS, pert, ls='--', color='#8e44ad', lw=1.3, label='perturbada')
    ax.axhline(cal['mediana'], color='#27ae60', lw=1.3, ls='-.')
    ax.axhline(cal['min'], color='#c0392b', lw=1.3, ls='--')
    ax.text(520, cal['mediana'], ' mediana limpia', fontsize=7.5, color='#27ae60',
            va='center')
    ax.text(520, cal['min'], ' traza real\n mas debil', fontsize=7.5, color='#c0392b',
            va='center')

    ax.set_xscale('log')
    ax.set_xlim(8.5, 1300)
    ax.set_ylim(0.35, 2.55)
    ax.set_xlabel('df (amplitud de la anomalia, en MAD)')
    ax.set_ylabel('tinta mediana en la epoca reescrita')
    ax.set_title('B. Cuanta traza deja el contrafactual', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(alpha=0.25)


def panel_bajo_piso(ax, d):
    x = np.arange(len(DFS))
    ancho = 0.26
    for i, (metodo, etiq, color, _) in enumerate(METODOS):
        y = 100 * serie(d, 'tinta', metodo, 'bajo_min')
        ax.bar(x + (i - 1) * ancho, y, ancho, color=color, label=etiq, alpha=0.9)

    ax.axhline(100 * d['confusion']['300|base p95']['limpia'], color='#27ae60',
               lw=1.2, ls='-.')
    ax.set_xticks(x)
    ax.set_xticklabels(DFS)
    ax.set_xlabel('df')
    ax.set_ylabel('epocas bajo el piso limpio (%)')
    ax.set_title('C. Sin umbral: fraccion mas tenue que cualquier dato real', fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis='y')


def panel_reduccion(ax, d):
    for metodo, etiq, color, marca in METODOS:
        ax.plot(DFS, serie(d, 'tabla', metodo, 'red_conserv', umbral=d['umbral']),
                marker=marca, color=color, lw=1.8, ms=5, label=etiq)
        ax.plot(DFS, serie(d, 'tabla', metodo, 'red_conserv', umbral=d['umbral_viejo']),
                marker=marca, color=color, lw=1.0, ms=3, alpha=0.4, ls=':')

    ax.set_xscale('log')
    ax.set_xlabel('df')
    ax.set_ylabel('reduccion del residuo en FLUXCAL (%)')
    ax.set_title('D. Reparacion, criterio conservador\nsolido: umbral 0.73   punteado: 0.2',
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.25)


def panel_borrado(ax, d):
    for metodo, etiq, color, marca in METODOS:
        ax.plot(DFS, 100 * serie(d, 'tabla', metodo, 'borrado', umbral=d['umbral']),
                marker=marca, color=color, lw=1.8, ms=5, label=etiq)
    ax.plot(DFS, 100 * serie(d, 'tabla', 'base p95', 'borrado_pert', umbral=d['umbral']),
            ls='--', color='#8e44ad', lw=1.3, label='perturbada')

    ax.set_xscale('log')
    ax.set_xlabel('df')
    ax.set_ylabel('epocas borradas (%)')
    ax.set_title('E. Tasa de borrado con el extractor calibrado', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.25)


def panel_confusion(ax, d):
    dfs_c = [100, 300]
    x = np.arange(len(dfs_c))
    ancho = 0.26
    for i, (metodo, etiq, color, _) in enumerate(METODOS):
        y = [100 * d['confusion'][f'{df}|{metodo}']['cf'] for df in dfs_c]
        b = ax.bar(x + (i - 1) * ancho, y, ancho, color=color, label=etiq, alpha=0.9)
        ax.bar_label(b, fmt='%.0f', fontsize=7.5, padding=1)

    limpia = 100 * d['confusion']['300|base p95']['limpia']
    ax.axhline(limpia, color='#27ae60', lw=1.4, ls='-.')
    ax.set_xticks(x)
    ax.set_xticklabels([f'df={df}' for df in dfs_c])
    ax.set_ylabel('epocas con color de otra banda (%)')
    ax.set_title('F. Confusion de banda en la region reescrita\n'
                 f'linea verde: reconstruccion limpia, {limpia:.1f}%', fontsize=9.5)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.25, axis='y')


def main():
    with open(ENTRADA) as fh:
        d = json.load(fh)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    panel_calibracion(axes[0, 0], d)
    panel_tinta(axes[0, 1], d)
    panel_bajo_piso(axes[0, 2], d)
    panel_reduccion(axes[1, 0], d)
    panel_borrado(axes[1, 1], d)
    panel_confusion(axes[1, 2], d)

    cal = d['calibracion']
    fig.suptitle(
        'Calibracion del extractor de trazas y su efecto sobre las cifras de contrafactuales\n'
        f"umbral {d['umbral']} fijado sobre {cal['n']:,} epocas de {cal['n_cal']} objetos limpios "
        'disjuntos del conjunto de evaluacion',
        fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(SALIDA, dpi=150)
    print('escrito', SALIDA)


if __name__ == '__main__':
    main()
