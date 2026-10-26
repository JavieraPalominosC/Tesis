"""Tres figuras del entrenamiento final de los priors MaskGIT sobre mad300.

Produce lo pedido para la reunion:
  1. curvas de entrenamiento (train y val) de los dos priors
  2. matriz de confusion de la deteccion en el punto de operacion
  3. macro F1 en funcion de la amplitud de la anomalia

El macro F1 y la matriz de confusion salen de la tarea de deteccion, que es
binaria: cada objeto del fold de validacion aporta un control (normal) y una
version perturbada (anomala), y se predice anomala cuando el score supera el
umbral tau, fijado en el percentil 95 de los controles de esa celda.

Uso: python scripts/fig_maskgit_final.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Con PRIOR_TAG=_folds se usan los priors reentrenados sobre la misma particion
# que el VQ-VAE, y la evaluacion sin los objetos que el prior habia visto.
TAG = os.environ.get('PRIOR_TAG', '')
EVAL = f'results/enc_cmp_full/eval/final_mad300{TAG}.json'
SALIDA = Path(os.environ.get('SALIDA_FIG', 'results/figs_profe'))

CORRIDAS = {
    'aleatoria': f'results/enc_cmp_full/mad300/maskgit{TAG}/logs/wandb',
    'ventana': f'results/enc_cmp_full/mad300/maskgit_ventana{TAG}/logs/wandb',
}

# Celda destacada: prior entrenado con mascara aleatoria y puntuado con ventana.
# Es la combinacion recomendada, porque alcanza el mejor recall sin reentrenar.
CELDA = 'aleatoria|ventana'
DF_MATRIZ = ['30', '100', '300']

COLOR = {'aleatoria': '#2c3e50', 'ventana': '#c0392b'}


def historial(dir_wandb):
    """Lee train/loss_epoch y val/loss del datastore binario de wandb."""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    archivos = sorted(Path(dir_wandb).glob('run-*/run-*.wandb'))
    if not archivos:
        raise SystemExit(f'sin archivo .wandb en {dir_wandb}')

    ds = datastore.DataStore()
    ds.open_for_scan(str(archivos[-1]))

    tr, va = {}, {}
    while True:
        d = ds.scan_data()
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof('record_type') != 'history':
            continue
        fila = {}
        for i in r.history.item:
            k = i.key or '/'.join(i.nested_key)
            try:
                fila[k] = json.loads(i.value_json)
            except (ValueError, TypeError):
                pass
        ep = fila.get('epoch')
        if ep is None:
            continue
        if 'train/loss_epoch' in fila:
            tr[int(ep)] = float(fila['train/loss_epoch'])
        if 'val/loss' in fila:
            va[int(ep)] = float(fila['val/loss'])
    return tr, va


def metricas(neg, pos, tau):
    """Matriz de confusion y F1 por clase para un umbral dado.

    neg son los scores de los controles y pos los de las perturbadas, de modo
    que la clase positiva es "anomala".
    """
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    tp = int((pos > tau).sum())
    fn = int(len(pos) - tp)
    fp = int((neg > tau).sum())
    tn = int(len(neg) - fp)

    def f1(vp, vfp, vfn):
        p = vp / (vp + vfp) if (vp + vfp) else 0.0
        r = vp / (vp + vfn) if (vp + vfn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    f1_anom = f1(tp, fp, fn)
    f1_norm = f1(tn, fn, fp)
    return {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'f1_anomala': f1_anom, 'f1_normal': f1_norm,
        'macro_f1': 0.5 * (f1_anom + f1_norm),
        'recall': tp / len(pos) if len(pos) else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) else 0.0,
    }


def fig_curvas(hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    for nombre, (tr, va) in hist.items():
        c = COLOR[nombre]
        ep_t = sorted(tr)
        ep_v = sorted(va)
        axes[0].plot(ep_t, [tr[e] for e in ep_t], '-o', color=c, ms=3.2, lw=1.5,
                     label=f'{nombre}, entrenamiento')
        axes[0].plot(ep_v, [va[e] for e in ep_v], '--s', color=c, ms=3.2, lw=1.5,
                     alpha=0.65, label=f'{nombre}, validacion')
        if va:
            mejor = min(va, key=va.get)
            axes[0].plot([mejor], [va[mejor]], '*', color=c, ms=15,
                         markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            # Si el mejor checkpoint cae cerca del final, la etiqueta se sale por
            # la derecha, asi que en ese caso se ancla hacia adentro.
            tarde = mejor > 0.75 * max(ep_v)
            axes[0].annotate(f'epoca {mejor}\n{va[mejor]:.3f}',
                             xy=(mejor, va[mejor]),
                             xytext=(-8, 12) if tarde else (8, 10),
                             textcoords='offset points', fontsize=7.5, color=c,
                             ha='right' if tarde else 'left')

        brecha = [va[e] - tr[e] for e in ep_v if e in tr]
        axes[1].plot([e for e in ep_v if e in tr], brecha, '-o', color=c,
                     ms=3.2, lw=1.5, label=nombre)

    axes[0].set_xlabel('epoca')
    axes[0].set_ylabel('perdida (entropia cruzada por token)')
    axes[0].set_title('A. Curvas de entrenamiento de los dos priors', fontsize=10)
    axes[0].legend(fontsize=7.5, framealpha=0.9)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color='gray', lw=0.8, ls=':')
    axes[1].set_xlabel('epoca')
    axes[1].set_ylabel('validacion menos entrenamiento')
    axes[1].set_title('B. Brecha entre validacion y entrenamiento', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle('Prior MaskGIT sobre mad300 (1.39M imagenes). Las dos perdidas no son '
                 'comparables entre si:\ncada prior se evalua con la forma de mascara con '
                 'que fue entrenado, y completar un rectangulo es mas dificil.',
                 fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    return fig


def fig_matriz(celda, res):
    n = len(DF_MATRIZ)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6))
    if n == 1:
        axes = [axes]

    for ax, df in zip(axes, DF_MATRIZ):
        m = res[df]
        M = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]], float)
        total = M.sum(axis=1, keepdims=True)
        ax.imshow(M / total, cmap='Blues', vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                frac = M[i, j] / total[i, 0]
                ax.text(j, i, f'{int(M[i, j])}\n{frac:.0%}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if frac > 0.5 else '#2c3e50')

        ax.set_xticks([0, 1]); ax.set_xticklabels(['normal', 'anomala'], fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['normal', 'anomala'], fontsize=8.5)
        ax.set_xlabel('prediccion', fontsize=9)
        ax.set_title(f'$\\Delta f = {df}$\nmacro F1 = {m["macro_f1"]:.2f}', fontsize=9.5)

    axes[0].set_ylabel('clase real', fontsize=9)
    fig.suptitle(f'Matriz de confusion de la deteccion, celda {celda.replace("|", " / ")}.'
                 f' 40 controles y 40 perturbadas del fold de validacion,\n'
                 f'umbral en el percentil 95 de los controles, que fija por construccion '
                 f'cerca de un 5% de falsos positivos.', fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    return fig


def fig_f1(por_celda, dfs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    x = [float(d) for d in dfs]

    estilo = {
        'aleatoria|aleatoria': ('#95a5a6', ':', 'o'),
        'aleatoria|ventana': ('#c0392b', '-', 'o'),
        'ventana|aleatoria': ('#7f8c8d', ':', 's'),
        'ventana|ventana': ('#2c3e50', '-', 's'),
    }

    for celda, res in por_celda.items():
        c, ls, mk = estilo[celda]
        grosor = 2.2 if celda == CELDA else 1.3
        axes[0].plot(x, [res[d]['macro_f1'] for d in dfs], ls, marker=mk, color=c,
                     lw=grosor, ms=4, label=celda.replace('|', ' / '))
        axes[1].plot(x, [res[d]['recall'] for d in dfs], ls, marker=mk, color=c,
                     lw=grosor, ms=4, label=celda.replace('|', ' / '))

    for ax, nombre in ((axes[0], 'macro F1'), (axes[1], 'recall de la clase anomala')):
        ax.set_xscale('log')
        ax.set_xlabel('$\\Delta f$ (en unidades del MAD del objeto)')
        ax.set_ylabel(nombre)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, title='entrenamiento / puntuacion',
                  title_fontsize=7.5, loc='lower right')

    axes[0].axhline(0.5, color='gray', lw=0.9, ls='--')
    axes[0].text(x[0], 0.515, 'azar', fontsize=7.5, color='gray')
    axes[0].set_title('A. Macro F1 sobre las clases normal y anomala', fontsize=10)
    axes[1].set_title('B. Recall, es decir la fraccion de anomalias detectadas', fontsize=10)

    fig.suptitle('Deteccion sobre 40 SNIa del fold de validacion, en el cruce entre la mascara '
                 'de entrenamiento y la de puntuacion.\nPuntuar con ventana domina, sin importar '
                 'con que mascara se entreno el prior.', fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    return fig


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    d = json.load(open(EVAL))
    dfs = d['dfs'] if isinstance(d['dfs'][0], str) else [str(x) for x in d['dfs']]

    por_celda = {}
    for celda, c in d['celdas'].items():
        neg = list(c['nll_control'].values())
        por_celda[celda] = {
            df: metricas(neg, list(c['nll_pert'][df].values()), c['tau'])
            for df in dfs
        }

    hist = {n: historial(p) for n, p in CORRIDAS.items()}

    fig_curvas(hist).savefig(SALIDA / f'maskgit_curvas{TAG}.png', dpi=160, bbox_inches='tight')
    fig_matriz(CELDA, por_celda[CELDA]).savefig(
        SALIDA / f'maskgit_confusion{TAG}.png', dpi=160, bbox_inches='tight')
    fig_f1(por_celda, dfs).savefig(SALIDA / f'maskgit_f1{TAG}.png', dpi=160, bbox_inches='tight')

    resumen = {
        'celda_destacada': CELDA,
        'n_control': len(next(iter(d['celdas'].values()))['nll_control']),
        'umbral': {c: v['tau'] for c, v in d['celdas'].items()},
        'metricas': por_celda,
        'epocas': {n: {'train': tr, 'val': va} for n, (tr, va) in hist.items()},
        'mejor_epoca': {n: (min(va, key=va.get) if va else None) for n, (_, va) in hist.items()},
    }
    with open(SALIDA / f'maskgit_final_resumen{TAG}.json', 'w') as f:
        json.dump(resumen, f, indent=1)

    print('figuras en', SALIDA)
    for n, (tr, va) in hist.items():
        mejor = min(va, key=va.get) if va else None
        print(f'  {n:10s} epocas {len(tr):2d} | mejor val {va[mejor]:.4f} en epoca {mejor}')
    print()
    for df in DF_MATRIZ:
        m = por_celda[CELDA][df]
        print(f'  df={df:>3s}  macro F1 {m["macro_f1"]:.3f} | recall {m["recall"]:.2f} | '
              f'precision {m["precision"]:.2f} | tp {m["tp"]} fp {m["fp"]}')


if __name__ == '__main__':
    main()
