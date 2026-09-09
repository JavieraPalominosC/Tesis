"""Figura de la ablacion: que factor manda de verdad.

Tres paneles. El primero es el resultado: el dano depende de cuanto de la
anomalia cubre la mascara, y de nada mas. El segundo muestra que las dos
palancas que se sospechaban, el prior y la temperatura de muestreo, casi no
mueven la aguja. El tercero traduce lo anterior a la metrica de flujo.

Uso: python scripts/contrafactual_ablacion_fig.py
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('results/enc_cmp_full/eval')
IMG = Path('imagenes')
COL = {'actual': '#c44e52', 'topk10': '#4c72b0', 'oraculo': '#55a868'}


def cargar(n):
    p = OUT / n
    return json.load(open(p)) if p.exists() else None


def main():
    IMG.mkdir(exist_ok=True)
    a2 = cargar('contrafactual_ablacion2.json')
    r1 = cargar('contrafactual_ablacion_resumen.json')
    fl = cargar('contrafactual_flujo_ablacion_resumen.json')
    if a2 is None:
        print('falta contrafactual_ablacion2.json')
        return
    r2 = cargar('contrafactual_ablacion2_resumen.json')['res']

    cob = {}
    for x in a2['geo']:
        cob.setdefault(x['forma'], []).append(x['cobertura'])
    cob = {k: 100 * float(np.mean(v)) for k, v in cob.items()}

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # --- 1. cobertura contra dano -------------------------------------------
    for dec in ('actual', 'topk10'):
        xs, ys, es, nm = [], [], [], []
        for cfg, e in r2.items():
            pr, f, d = cfg.split('|')
            if pr != 'aleatoria' or d != dec or f == 'oraculo':
                continue
            xs.append(cob[f])
            ys.append(100 * e['bajo_piso'])
            es.append(100 * e['ee'])
            nm.append(f)
        o = np.argsort(xs)
        xs, ys, es = np.array(xs)[o], np.array(ys)[o], np.array(es)[o]
        ax[0].errorbar(xs, ys, yerr=es, marker='o', ms=6, lw=1.6, capsize=3,
                       color=COL[dec], label=f'decodificación {dec}')
        for i in o:
            if dec == 'topk10':
                ax[0].annotate(nm[i], (cob[nm[i]], 100 * r2[f'aleatoria|{nm[i]}|{dec}']
                                       ['bajo_piso']), fontsize=7.5,
                               xytext=(4, -9), textcoords='offset points')
    for dec in ('actual', 'topk10'):
        e = r2.get(f'aleatoria|oraculo|{dec}')
        if e:
            ax[0].errorbar([100], [100 * e['bajo_piso']], yerr=[100 * e['ee']],
                           marker='*', ms=15, color=COL['oraculo'], capsize=3,
                           ls='none', label='oráculo' if dec == 'actual' else None)
    ax[0].set_xlabel('cobertura de la zona real de inyección (%)')
    ax[0].set_ylabel('épocas bajo el piso de tinta limpio (%)')
    ax[0].set_title('Lo que manda es cuánto de la anomalía se borra')
    ax[0].legend(fontsize=8.5, loc='upper right')
    ax[0].grid(alpha=.3)

    # --- 2. las palancas que se sospechaban ---------------------------------
    if r1:
        r = r1['res']
        decs = ['actual', 'greedy', 'topk3', 'topk10']
        barras = [('p95', 'aleatoria', '#b0b0b0', ''),
                  ('p95', 'ventana', '#b0b0b0', '//'),
                  ('rect', 'aleatoria', '#4c72b0', ''),
                  ('rect', 'ventana', '#4c72b0', '//')]
        w, x0 = 0.2, np.arange(len(decs))
        for k, (f, pr, c, hat) in enumerate(barras):
            v = [100 * r[f'{pr}|{f}|{d}|10']['bajo_piso']
                 if f'{pr}|{f}|{d}|10' in r else np.nan for d in decs]
            e = [100 * r[f'{pr}|{f}|{d}|10']['ee']
                 if f'{pr}|{f}|{d}|10' in r else np.nan for d in decs]
            ax[1].bar(x0 + (k - 1.5) * w, v, w, yerr=e, hatch=hat, color=c,
                      edgecolor='white', linewidth=.6, capsize=2,
                      label=f'máscara {f}, prior {pr}')
        ax[1].set_xticks(x0)
        ax[1].set_xticklabels(decs)
        ax[1].set_xlabel('decodificación')
        ax[1].set_ylabel('épocas bajo el piso de tinta limpio (%)')
        ax[1].set_title('Prior y temperatura casi no importan')
        ax[1].legend(fontsize=7.5, ncol=1, loc='center left')
        ax[1].set_ylim(0, 68)
        ax[1].grid(alpha=.3, axis='y')

    # --- 3. traduccion a flujo ----------------------------------------------
    if fl:
        t = fl['tabla']
        nombres = [n for n in fl['configs'] if f'300|{n}' in t]
        v = [t[f'300|{n}']['red_conserv'] for n in nombres]
        o = np.argsort(v)
        c = ['#55a868' if 'oraculo' in nombres[i] else
             '#bbbbbb' if 'azar' in nombres[i] else
             '#c44e52' if nombres[i] == 'referencia' else '#4c72b0' for i in o]
        ax[2].barh(range(len(o)), [v[i] for i in o], color=c, edgecolor='white')
        ax[2].set_yticks(range(len(o)))
        ax[2].set_yticklabels([nombres[i] for i in o], fontsize=8.5)
        ax[2].axvline(t['300|referencia']['red_conserv'], color='#c44e52', ls='--',
                      lw=1.2)
        ax[2].set_xlabel('reducción del residuo en FLUXCAL (%)')
        ax[2].set_title('En flujo, criterio conservador, $\\Delta F$=300')
        ax[2].grid(alpha=.3, axis='x')
    else:
        ax[2].axis('off')
        ax[2].text(.5, .5, 'falta la corrida de flujo', ha='center', va='center')

    fig.suptitle('Ablación del generador de contrafactuales: qué mueve la aguja '
                 'y qué no', y=1.0, fontsize=12.5)
    fig.text(.5, -.03, 'Diagnóstico sobre 300 SNIa del fold de validación, '
             '$\\Delta F$=300 MAD. El oráculo enmascara la zona de inyección '
             'exacta y marca el techo alcanzable con localización perfecta.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(IMG / 'contrafactual_ablacion.png', dpi=180, bbox_inches='tight')
    print(f'imagenes/contrafactual_ablacion.png')


if __name__ == '__main__':
    main()
