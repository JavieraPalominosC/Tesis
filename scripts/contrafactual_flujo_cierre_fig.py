"""Figura del cierre en flujo, dibujada desde el JSON ya calculado.

Separada del calculo porque el promedio del neto no era el estadistico que
contaba la historia: sale positivo en todo dt y esconde el umbral. El umbral
aparece al mirar en que fraccion de objetos el neto es positivo, que es la
mediana disfrazada de porcentaje.

Uso: python scripts/contrafactual_flujo_cierre_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
DTS = [5, 10, 15, 20, 30, 45, 60, 90]
DFS_DT = [100, 300]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
COL = {'base p95': '#e08214', 'rect 1 pasa': '#1b7837', 'azar': '#8856a7'}
MRK = {'base p95': 'v', 'rect 1 pasa': '^', 'azar': 's'}


def main():
    with open(OUT / 'contrafactual_flujo_cierre.json') as f:
        d = json.load(f)
    piso, U, A = d['piso_inversor'], d['umbral_flujo'], d['amplificacion']
    pts = [np.mean(d['puntos_por_dt'][str(dt)]) for dt in DTS]
    g = lambda df, m, k: [U[str(df)][m][str(dt)][k] for dt in DTS]

    fig, ax = plt.subplots(1, 4, figsize=(21, 5.1))

    a = ax[0]
    for df, ls in zip(DFS_DT, [':', '-']):
        for m in METODOS:
            a.plot(DTS, g(df, m, 'neto'), MRK[m] + ls, color=COL[m], ms=5,
                   label=f'{m}, df={df}' if ls == '-' else None)
    a.set_yscale('log')
    a.set_ylabel('neto medio = ganancia - perdida\n[FLUXCAL$^2$ sumado]')
    a.set_title('El promedio es positivo en toda duracion\n'
                'punteado df=100, continuo df=300', fontsize=10)

    a = ax[1]
    for df, ls in zip(DFS_DT, [':', '-']):
        for m in METODOS:
            a.plot(DTS, [100 * x for x in g(df, m, 'frac_neto_pos')], MRK[m] + ls,
                   color=COL[m], ms=5, label=f'{m}, df={df}' if ls == '-' else None)
    a.axhline(50, color='#333333', lw=1.2)
    a.axvspan(0, 40, color='#cccccc', alpha=0.35, zorder=0)
    a.text(20, 95, 'menos de 4 puntos\nfotometricos', ha='center', fontsize=8,
           color='#555555')
    a.set_ylim(0, 102)
    a.set_ylabel('% de objetos con neto > 0')
    a.set_title('Pero el objeto tipico solo gana sobre ~40 dias\n'
                'bajo la linea, el contrafactual mediano empeora', fontsize=10)

    a = ax[2]
    a.plot(DTS, g(300, 'base p95', 'vent_pert'), 'o-', color='#b2182b', ms=5,
           label='perturbada (sin reparar)')
    for m in METODOS:
        a.plot(DTS, g(300, m, 'vent'), MRK[m] + '-', color=COL[m], ms=5, label=m)
    a.plot(DTS, g(300, 'base p95', 'resto_pert'), 'o--', color='#777777', ms=4,
           label='fuera de la ventana')
    a.axhline(piso, color='#333333', lw=1.4, ls='-.')
    a.text(6, piso * 0.72, f'piso del inversor  {piso:.1f} FLUXCAL', fontsize=8)
    a.set_yscale('log')
    a.set_ylim(5, 30000)
    a.set_ylabel('residuo absoluto en flujo  [FLUXCAL, RMSE]')
    a.set_title('df=300: la reparacion es real, el dano no\n'
                'el residuo fuera de la ventana ya esta en el piso', fontsize=10)

    for a in ax[:3]:
        a.set_xlabel('dt: duracion de la anomalia [dias]')
        a.legend(frameon=False, fontsize=8, ncol=2 if a is ax[2] else 1,
                 loc={0: 'lower center', 1: 'lower right', 2: 'upper center'}[
                     list(ax[:3]).index(a)])
        b = a.twiny()
        b.set_xlim(a.get_xlim())
        b.set_xticks(DTS)
        b.set_xticklabels([f'{p:.1f}' for p in pts], fontsize=7)
        b.set_xlabel('puntos fotometricos reales en la ventana', fontsize=8)

    a = ax[3]
    xs, w = np.arange(len(DFS)), 0.26
    for i, m in enumerate(METODOS):
        v = [100 * A[m][str(df)] for df in DFS]
        a.bar(xs + (i - 1) * w, v, w, color=COL[m], label=m)
        for x, y in zip(xs + (i - 1) * w, v):
            a.text(x, y + 0.8, f'{y:.0f}', ha='center', fontsize=7)
    a.set_xticks(xs)
    a.set_xticklabels([str(df) for df in DFS])
    a.set_xlabel(r'$df$ [veces MAD$_r$],  anomalia de cola completa')
    a.set_ylabel('% de objetos')
    a.set_ylim(0, 46)
    a.legend(frameon=False, fontsize=8)
    a.set_title('Tasa de amplificacion: el contrafactual deja\n'
                'el residuo en flujo PEOR que la perturbada', fontsize=10)

    s = d['saturacion']['500']
    fig.suptitle('Cierre experimental en flujo: umbral de tamano minimo y tasa de fallo. '
                 '40 SNIa del fold de validacion, nunca vistas en entrenamiento.',
                 fontsize=12)
    fig.text(0.5, 0.005,
             'Ganancia y perdida son sumas de residuo cuadratico sobre las epocas '
             'realmente inyectadas y sobre el resto. Solo df=500 satura la '
             f'codificacion mad300 ({100 * s["frac_recortada"]:.0f}% de epocas '
             'recortadas en 300 MAD); df<=300 no recorta.',
             ha='center', fontsize=8.5, color='#444444')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(OUT / 'contrafactual_flujo_cierre.png', dpi=140)
    print('listo')


if __name__ == '__main__':
    main()
