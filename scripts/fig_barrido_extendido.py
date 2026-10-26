"""Figuras del barrido extendido en amplitud y duracion.

Cubre los puntos 3 y 4 de la reunion del 28 de septiembre: bajar el rango de
Delta f hasta dejar ver el piso de sensibilidad, y subir el de Delta t hasta
cubrir una fraccion apreciable de la curva.

Uso: python scripts/fig_barrido_extendido.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRIOR_TAG = os.environ.get('PRIOR_TAG', '')
ENTRADA = f'results/enc_cmp_full/eval/barrido_extendido{PRIOR_TAG}.json'
SALIDA = Path(os.environ.get('SALIDA_FIG', 'results/figs_profe'))


def fig_amplitud(d, claves=('recall', 'macro_f1', 'auc')):
    x = d['dfs_1d']
    piso_f1 = piso_macro_f1(len(d['objetos']), 0.95)
    solo = len(claves) == 1
    fig, axes = plt.subplots(1, len(claves), figsize=(6.6 if solo else 4.5 * len(claves), 4.3),
                             squeeze=False)
    estilo = {'ventana': ('#c0392b', '-', 'o'), 'aleatoria': ('#95a5a6', ':', 's')}
    # Referencia horizontal de cada metrica: para el macro F1 lo informativo no
    # es 0.5 sino el valor que da un detector que no encuentra nada.
    refs = {'recall': (0.5, None), 'auc': (0.5, 'azar'),
            'macro_f1': (piso_f1, 'piso: recall 0')}
    letras = 'ABC'

    for i, clave in enumerate(claves):
        ax = axes[0][i]
        nombre = METRICAS[clave][0]
        for forma, (c, ls, mk) in estilo.items():
            y = [d['celdas'][forma]['una_d'][str(v)][clave] for v in x]
            ax.plot(x, y, ls, marker=mk, color=c, lw=2 if forma == 'ventana' else 1.3,
                    ms=4.5, label=f'puntuado con {forma}')
        ax.set_xscale('log')
        ax.set_xlabel('$\\Delta f$ (en MAD del objeto)')
        ax.set_ylabel(nombre)
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.03)
        # Region que antes no se media: el barrido anterior arrancaba en 10.
        ax.axvspan(min(x), 10, color='#f39c12', alpha=0.10)
        ax.legend(fontsize=8, loc='lower right')

        valor, etiqueta = refs[clave]
        ax.axhline(valor, color='gray', ls='--', lw=0.9)
        if etiqueta:
            ax.text(min(x) * 1.05, valor + 0.02, etiqueta, fontsize=8, color='gray')
        ax.set_title(nombre if solo else f'{letras[i]}. {nombre}', fontsize=10)

    if claves == ('macro_f1',):
        c = d['celdas']['ventana']['una_d']
        cruce = next((v for v in x if c[str(v)]['macro_f1'] >= 0.8), None)
        fig.suptitle('Macro F1 contra amplitud. La franja naranja es el rango que el barrido '
                     'anterior no cubria.\n'
                     f'Arranca en {piso_f1:.2f} y no en 0, porque con recall nulo la clase normal '
                     f'se sigue acertando; supera 0.8 sobre $\\Delta f \\approx {cruce}$.',
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.87])
    else:
        piso = d['celdas']['ventana']['una_d'][str(min(x))]['auc']
        cruce = next((v for v in x if d['celdas']['ventana']['una_d'][str(v)]['auc'] >= 0.7), None)
        fig.suptitle('Piso de sensibilidad en amplitud. La franja naranja es el rango que el '
                     'barrido anterior no cubria, porque arrancaba en $\\Delta f = 10$.\n'
                     f'Con $\\Delta f = {min(x)}$ el detector esta en azar (AUROC {piso:.3f}) y '
                     f'recien sobre $\\Delta f \\approx {cruce}$ supera 0.7.', fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def piso_macro_f1(n_ctrl, q):
    """Macro F1 cuando el recall es 0, que es el minimo alcanzable.

    No es 0: con el umbral en el percentil q de los controles, la clase normal
    se sigue acertando casi entera, asi que su F1 arrastra el promedio. Sirve
    para que la escala de color cubra el rango posible y no uno vacio.
    """
    fp = round(n_ctrl * (1 - q))
    tn, fn = n_ctrl - fp, n_ctrl
    p, r = tn / (tn + fn), tn / (tn + fp)
    return 0.5 * (2 * p * r / (p + r))


METRICAS = {
    'recall': ('recall', 0.0),
    'macro_f1': ('macro F1', None),   # el piso depende del numero de objetos
    'auc': ('AUROC', 0.5),
}


def fig_duracion(d, claves=('recall', 'macro_f1', 'auc')):
    dfs, dts = d['dfs_2d'], d['dts']
    c = d['celdas']['ventana']
    piso_f1 = piso_macro_f1(len(d['objetos']), 0.95)
    ancho = 6.2 if len(claves) == 1 else 5.85 * len(claves)
    fig, axes = plt.subplots(1, len(claves), figsize=(ancho, 4.6), squeeze=False)

    paneles = [(axes[0][i], k, METRICAS[k][0],
                round(piso_f1, 2) if METRICAS[k][1] is None else METRICAS[k][1])
               for i, k in enumerate(claves)]

    for ax, clave, nombre, vmin in paneles:
        M = np.array([[c['dos_d'][f'{df}|{dt}'][clave] for dt in dts] for df in dfs])
        im = ax.imshow(M, cmap='RdYlGn', vmin=vmin, vmax=1,
                       aspect='auto', origin='lower')
        for i in range(len(dfs)):
            for j in range(len(dts)):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                        color='black')
        ax.set_xticks(range(len(dts)))
        ax.set_xticklabels([f'{dt}\n({100 * d["cobertura_dt"][str(dt)]:.0f}%)' for dt in dts],
                           fontsize=7.5)
        ax.set_yticks(range(len(dfs)))
        ax.set_yticklabels(dfs, fontsize=8)
        ax.set_xlabel('$\\Delta t$ en dias, y fraccion de la banda $r$ que abarca', fontsize=9)
        ax.set_ylabel('$\\Delta f$ (en MAD)', fontsize=9)
        ax.set_title(nombre, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045)

    nota_f1 = (f'El macro F1 arranca en {piso_f1:.2f} y no en 0: con recall nulo la clase normal '
               'se sigue acertando y su F1 arrastra el promedio.')
    if claves == ('macro_f1',):
        fig.suptitle('Macro F1 en el plano amplitud-duracion, puntuando con mascara de ventana.\n'
                     'Umbral en el percentil 95 de los controles, o sea 2 falsos positivos de 40 '
                     'fijos por construccion.\n' + nota_f1, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
    else:
        fig.suptitle('Amplitud contra duracion, puntuando con mascara de ventana. El barrido '
                     'anterior llegaba a $\\Delta t = 80$ dias, un tercio de los puntos; aca a '
                     '320 y un 62%. ' + nota_f1, fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    d = json.load(open(ENTRADA))
    fig_amplitud(d).savefig(SALIDA / f'barrido_amplitud{PRIOR_TAG}.png', dpi=150,
                            bbox_inches='tight')
    fig_amplitud(d, ('macro_f1',)).savefig(SALIDA / f'barrido_amplitud_f1{PRIOR_TAG}.png',
                                           dpi=150, bbox_inches='tight')
    fig_duracion(d).savefig(SALIDA / f'barrido_duracion{PRIOR_TAG}.png', dpi=150,
                            bbox_inches='tight')
    # Version de un solo panel, para cuando solo interesa el macro F1.
    fig_duracion(d, ('macro_f1',)).savefig(SALIDA / f'barrido_duracion_f1{PRIOR_TAG}.png',
                                           dpi=150, bbox_inches='tight')
    print('figuras en', SALIDA)


if __name__ == '__main__':
    main()
