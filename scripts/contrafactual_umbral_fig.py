"""Figura del umbral afinado, dibujada desde el analisis ya calculado.

Separada del computo para poder iterar el diseno sin repetir la GPU.

La curva principal es la convencion "medible": entre los casos donde el punto
fotometrico sobrevive a la reescritura, en que fraccion mejora el flujo. La
convencion "estricta", que ademas exige no borrar ninguna epoca, va como linea
tenue de referencia, pero no sirve como umbral porque penaliza mecanicamente las
ventanas largas: con una tasa de borrado por epoca de p, la probabilidad de no
borrar ninguna de n epocas es (1-p)^n y cae sola al crecer la ventana. El
borrado se reporta aparte, que es donde de verdad se ve.

Uso: python scripts/contrafactual_umbral_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
DTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
COL = {'base p95': '#e08214', 'rect 1 pasa': '#1b7837', 'azar': '#8856a7'}
MRK = {'base p95': 'v', 'rect 1 pasa': '^', 'azar': 's'}


def logis(x50, beta, xs):
    b0 = -beta * np.log(max(x50, 1e-12))
    return 1 / (1 + np.exp(-np.clip(b0 + beta * np.log(xs), -30, 30)))


def panel(a, d, df, eje):
    lo_x, hi_x = (4.2, 96) if eje == 'dias' else (0.92, 12.4)
    xs = np.geomspace(lo_x, hi_x, 200)
    clave = 'por_dt' if eje == 'dias' else 'por_puntos'
    notas = []
    for m in METODOS:
        c = d['curvas'][str(df)][m]['estricto'][clave]
        ks = sorted(c, key=float)
        a.plot([float(k) for k in ks], [100 * c[k]['p'] for k in ks], ':',
               color=COL[m], lw=1.1, alpha=0.55)

        c = d['curvas'][str(df)][m]['medible'][clave]
        ks = sorted(c, key=float)
        x = np.array([float(k) for k in ks])
        p = np.array([c[k]['p'] for k in ks])
        yerr = 100 * np.vstack([p - [c[k]['lo'] for k in ks],
                                [c[k]['hi'] for k in ks] - p])
        a.errorbar(x, 100 * p, yerr=yerr, fmt=MRK[m], color=COL[m], ms=5, lw=0,
                   elinewidth=1.1, capsize=2, alpha=0.9)
        u = d['umbrales'][str(df)][m]['medible'][eje]
        a.plot(xs, 100 * logis(u['x50'], u['beta'], xs), '-', color=COL[m], lw=1.9,
               label=m)
        if u['frac_dentro_grilla'] > 0.8:
            a.axvspan(u['lo'], u['hi'], color=COL[m], alpha=0.16, lw=0)
            a.plot([u['x50']], [50], 'o', color=COL[m], ms=10, mfc='white', mew=2.2)
            notas.append(f'{m}: {u["x50"]:.2f}  [{u["lo"]:.2f}, {u["hi"]:.2f}]')
        else:
            notas.append(f'{m}: sin cruce en el rango')
    a.axhline(50, color='#333333', lw=1.2)
    a.set_xscale('log')
    a.set_ylim(0, 100)
    a.set_xlim(lo_x, hi_x)
    a.set_ylabel('% de objetos con neto positivo en flujo')
    a.set_xlabel('duracion de la anomalia [dias]' if eje == 'dias'
                 else 'puntos fotometricos perturbados')
    t = (DTS if eje == 'dias' else list(range(1, 13)))
    a.set_xticks(t)
    a.set_xticklabels([str(v) for v in t], fontsize=7)
    a.text(0.03, 0.03, 'cruce del 50% e IC95:\n' + '\n'.join(notas),
           transform=a.transAxes, va='bottom', fontsize=7.2, family='monospace')
    a.legend(frameon=False, fontsize=8, loc='upper left')
    a.set_title(f'df={df},  contra {eje}', fontsize=10)


def barras(a, d, clave, titulo, ylab):
    xs, w = np.arange(len(DFS)), 0.26
    for i, m in enumerate(METODOS):
        v = [100 * d['amplificacion'][str(df)][m][clave] for df in DFS]
        kw = {}
        if clave == 'p':
            kw['yerr'] = [[100 * (d['amplificacion'][str(df)][m]['p'] -
                                  d['amplificacion'][str(df)][m]['lo']) for df in DFS],
                          [100 * (d['amplificacion'][str(df)][m]['hi'] -
                                  d['amplificacion'][str(df)][m]['p']) for df in DFS]]
            kw['error_kw'] = {'elinewidth': 1, 'ecolor': '#333333'}
        a.bar(xs + (i - 1) * w, v, w, color=COL[m], label=m, **kw)
        for x, y in zip(xs + (i - 1) * w, v):
            a.text(x, y + 0.7, f'{y:.0f}', ha='center', fontsize=6.5)
    a.set_xticks(xs)
    a.set_xticklabels([str(x) for x in DFS])
    a.set_xlabel(r'$df$ [veces MAD$_r$],  anomalia de cola completa')
    a.set_ylabel(ylab)
    a.set_title(titulo, fontsize=10)
    a.legend(frameon=False, fontsize=8)


def main():
    with open(OUT / 'contrafactual_umbral_analisis.json') as f:
        d = json.load(f)
    fig, ax = plt.subplots(2, 3, figsize=(18.5, 10.5))
    for i, df in enumerate([300, 100]):
        panel(ax[i][0], d, df, 'dias')
        panel(ax[i][1], d, df, 'puntos')
    barras(ax[0][2], d, 'p', 'Tasa de amplificacion\nel residuo en flujo queda '
           'PEOR que el de la perturbada', '% de objetos')
    barras(ax[1][2], d, 'borrado', 'Tasa de borrado\nel contrafactual elimina el '
           'punto en vez de corregirlo', '% de epocas inyectadas')

    fig.suptitle(f'Umbral de tamano minimo del contrafactual en espacio de flujo. '
                 f'{d["n_objetos"]} SNIa del fold de validacion, nunca vistas en '
                 f'entrenamiento.', fontsize=12.5)
    fig.tight_layout(rect=[0, 0.032, 1, 0.955])
    fig.text(0.5, 0.016,
             'Simbolos con barra: fraccion observada e IC95 de Wilson entre los casos '
             'donde el punto sobrevive a la reescritura. Linea continua: ajuste '
             'logistico; circulo blanco y banda: cruce del 50% con IC95 por bootstrap '
             'de 2000 remuestras de objetos.', ha='center', fontsize=8.5, color='#444444')
    fig.text(0.5, 0.002,
             'Linea punteada tenue: convencion estricta, que ademas exige no borrar '
             'ninguna epoca. No sirve como umbral porque decae sola con el largo de la '
             f'ventana. Piso del inversor {d["piso_inversor"]:.1f} FLUXCAL.',
             ha='center', fontsize=8.5, color='#444444')
    fig.savefig(OUT / 'contrafactual_umbral.png', dpi=140)
    print('listo')


if __name__ == '__main__':
    main()
