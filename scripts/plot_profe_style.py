"""
Figuras pedidas en la reunion, con el eje en FLUJO ABSOLUTO (FLUXCAL).

  1) f1_recall_vs_flux.png  F1 y recall vs Delta F  (el grafico de siempre,
                            antes en veces MAD)
  2) dist_max_nll.png       densidades de max S_ij, control vs perturbacion
  3) roc_deltaf.png         ROC por nivel de Delta F
  4) contour_dt_dF.png      isolineas de recall en el plano (Delta F, Delta t)

Sobre el eje en flujo absoluto: la perturbacion se aplico como
df * MAD_r del objeto, asi que cada supernova recibio un Delta F distinto.
Lo que se grafica es <Delta F> = df * <MAD_r> sobre las 40 SNIa, y el
MAD_r individual va de 1.4 a 9.7 FLUXCAL. Por eso cada punto del eje es un
promedio, no un valor fijo por objeto.

Los 100 Jy / 10^6 Jy de la pizarra son esquematicos, no se usan aca.

Uso: python scripts/plot_profe_style.py
     PROFE_FIG_OUT=/tmp/figs python scripts/plot_profe_style.py
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAIL_CANDIDATES = [
    Path('results/synthetic_eval/mad_tail/summary.json'),
    Path('reunion_profe_2026-08-24/mad_dt/tail/summary.json'),
]
DT_CANDIDATES = [
    Path('results/synthetic_eval/mad_dt_wfg/summary.json'),
    Path('reunion_profe_2026-08-24/mad_dt/summary.json'),
]
MANIFEST_CANDIDATES = [
    Path('data/synthetic_mad_tail_manifest.json'),
]
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
DFS = list(range(10, 101, 10))
HIGHLIGHT = [10, 40, 70, 100]
COLORS = ['#444444', '#4C78A8', '#F58518', '#54A24B', '#E45756']


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def roc_xy(neg, pos):
    """ROC barriendo el umbral sobre max NLL: control = negativo."""
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
    order = np.argsort(-s)
    y = y[order]
    tpr = np.r_[0.0, np.cumsum(y) / max(y.sum(), 1.0), 1.0]
    fpr = np.r_[0.0, np.cumsum(1.0 - y) / max((1.0 - y).sum(), 1.0), 1.0]
    trapz = getattr(np, 'trapezoid', np.trapz)
    return fpr, tpr, float(trapz(tpr, fpr))


def kde_curve(x, grid):
    """KDE gaussiana con ancho de banda de Silverman."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros_like(grid)
    std = np.std(x)
    h = 1.06 * std * (len(x) ** -0.2) if std > 1e-6 else 0.3
    z = (grid[None, :] - x[:, None]) / h
    return np.exp(-0.5 * z ** 2).mean(axis=0) / (h * np.sqrt(2 * np.pi))


def load_mads(snids):
    """MAD_r por objeto. El manifest de la cola es la fuente autoritativa;
    el summary del grid df x dt tiene los mismos valores como respaldo."""
    man = first_existing(MANIFEST_CANDIDATES)
    if man is not None:
        rows = json.load(open(man))
        mads = {r['snid']: float(r['mad']) for r in rows if r.get('mad')}
        if all(s in mads for s in snids):
            return mads, man.name
    dt_path = first_existing(DT_CANDIDATES)
    mads = {}
    if dt_path is not None:
        for r in json.load(open(dt_path)).get('per_image', []):
            if r.get('mad') and r['snid'] not in mads:
                mads[r['snid']] = float(r['mad'])
    return mads, (dt_path.name if dt_path else 'ninguna')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tail_path = first_existing(TAIL_CANDIDATES)
    if tail_path is None:
        raise SystemExit('no encuentro el summary de la cola MAD')
    tail = json.load(open(tail_path))

    by = {}
    for r in tail['per_image']:
        by.setdefault(r['snid'], {})[r['tag']] = r
    snids = sorted(s for s in by if 'control' in by[s])
    ctrl = np.array([by[s]['control']['max_nll'] for s in snids])
    tau = float(tail['tau_max'])

    mads, mad_src = load_mads(snids)
    mad_vals = np.array([mads[s] for s in snids if s in mads])
    mean_mad = float(mad_vals.mean())
    print(f'MAD_r desde {mad_src}: n={len(mad_vals)}/{len(snids)}  '
          f'media={mean_mad:.3f}  rango={mad_vals.min():.2f}-{mad_vals.max():.2f}',
          flush=True)

    tags = [f'df{k}_unfrozen' for k in DFS]
    tags = [t for t in tags if t in tail['per_tag']]
    dfs = [int(t[2:].split('_')[0]) for t in tags]
    fluxes = np.array([d * mean_mad for d in dfs])
    f1s = [tail['per_tag'][t]['max_detector']['f1'] for t in tags]
    recs = [tail['per_tag'][t]['max_detector']['recall'] for t in tags]

    sub = (f'$\\Delta F = df \\times \\mathrm{{MAD}}_r$ por objeto; el eje es el '
           f'promedio sobre {len(snids)} SNIa '
           f'($\\mathrm{{MAD}}_r$ entre {mad_vals.min():.1f} y {mad_vals.max():.1f})')

    # --- 1. F1 y recall vs flujo absoluto -----------------------------------
    x = np.arange(len(dfs))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, f1s, width=0.4, color='steelblue', label='F1')
    ax.bar(x + 0.2, recs, width=0.4, color='darkorange', label='recall')
    for xi, v in zip(x, f1s):
        ax.text(xi - 0.2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x, recs):
        ax.text(xi + 0.2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, [f'{f:.0f}' for f in fluxes])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
    ax.set_ylabel('F1 / recall')
    ax.set_title(f'Deteccion por el maximo NLL, cola de la curva, escala libre\n{sub}',
                 fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'f1_recall_vs_flux.png', dpi=140)
    plt.close(fig)

    # --- 2. densidades de max S_ij ------------------------------------------
    series = [(ctrl, 'sin perturbacion')]
    for k in HIGHLIGHT:
        t = f'df{k}_unfrozen'
        if t not in tail['per_tag']:
            continue
        arr = np.array([by[s][t]['max_nll'] for s in snids if t in by[s]])
        series.append((arr, f'{k}$\\times$MAD  ($\\langle\\Delta F\\rangle$={k * mean_mad:.0f})'))

    lo = min(a.min() for a, _ in series) - 1.0
    hi = max(a.max() for a, _ in series) + 1.0
    grid = np.linspace(lo, hi, 500)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for (arr, lab), c in zip(series, COLORS):
        dens = kde_curve(arr, grid)
        ax.plot(grid, dens, color=c, lw=2, label=lab)
        ax.fill_between(grid, dens, color=c, alpha=0.12)
    ax.axvline(tau, color='crimson', ls='--', lw=1.2,
               label=f'$\\tau$ = P95 control = {tau:.2f}')
    ax.set_xlabel(r'$\max_{ij}\, S_{ij}$,   $S_{ij} = -\log p(s_{ij}\,|\,\mathrm{resto})$')
    ax.set_ylabel('densidad')
    ax.set_title(f'Distribucion del maximo NLL sobre el mapa 32$\\times$32\n'
                 f'KDE sobre {len(snids)} SNIa', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'dist_max_nll.png', dpi=140)
    plt.close(fig)

    # --- 3. ROC --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color='gray', lw=0.8, ls='--', label='azar')
    for k, c in zip(HIGHLIGHT, COLORS[1:]):
        t = f'df{k}_unfrozen'
        if t not in tail['per_tag']:
            continue
        pos = np.array([by[s][t]['max_nll'] for s in snids if t in by[s]])
        fpr, tpr, auc = roc_xy(ctrl, pos)
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f'{k}$\\times$MAD  ($\\langle\\Delta F\\rangle$={k * mean_mad:.0f}, AUC={auc:.2f})')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR (recall)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ROC del detector por maximo NLL', fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'roc_deltaf.png', dpi=140)
    plt.close(fig)

    # --- 4. isolineas de recall en (Delta F, Delta t) -----------------------
    dt_path = first_existing(DT_CANDIDATES)
    if dt_path is not None:
        dt_sum = json.load(open(dt_path))
        dfs_dt = np.array(dt_sum['dfs'], float)
        dts = np.array(dt_sum['dts'], float)
        rec2 = np.array(dt_sum['recall_max'], float)
        x_flux = dfs_dt * mean_mad
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        im = ax.pcolormesh(x_flux, dts, rec2, shading='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, label='recall')
        lo_r, hi_r = float(rec2.min()), float(rec2.max())
        levels = [lv for lv in np.arange(0.05, 1.0, 0.05) if lo_r < lv < hi_r]
        if levels:
            cs = ax.contour(x_flux, dts, rec2, levels=levels,
                            colors='#1f3d7a', linewidths=1.4)
            ax.clabel(cs, fmt=lambda v: f'{100 * v:.0f}%', fontsize=8)
        ax.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
        ax.set_ylabel(r'$\Delta t$  [dias MJD]')
        ax.set_title('Isolineas de recall en $(\\Delta F,\\,\\Delta t)$, escala CONGELADA\n'
                     f'el recall se queda entre {100 * lo_r:.0f}% y {100 * hi_r:.0f}%: '
                     'no aparecen las isolineas 0-100%', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / 'contour_dt_dF.png', dpi=140)
        plt.close(fig)

    meta = {
        'n_objetos': len(snids),
        'mad_r': {'media': mean_mad, 'min': float(mad_vals.min()),
                  'max': float(mad_vals.max()), 'fuente': mad_src},
        'tau_p95_control': tau,
        'delta_f_medio_por_df': {str(d): d * mean_mad for d in dfs},
        'f1_por_df': dict(zip(map(str, dfs), f1s)),
        'recall_por_df': dict(zip(map(str, dfs), recs)),
    }
    with open(OUT / 'flux_axis.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Figuras en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
