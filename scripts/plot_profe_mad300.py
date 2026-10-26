"""Las cuatro figuras del profe, rehechas con mad300 y los modelos definitivos.

Reemplazan a las que genero plot_profe_style.py, que estaban hechas con la
codificacion antigua y por lo tanto mostraban el problema (la deteccion que se
estanca) en vez de la solucion.

Que cambio respecto de aquellas:

  - Codificacion mad300 (asinh sobre mediana y MAD del objeto) en vez de la
    normalizacion por percentiles, que saturaba.
  - VQ-VAE y MaskGIT entrenados sobre las 1.39M imagenes completas.
  - Se puntua con mascara de ventana contigua, que fue el otro hallazgo: tapar
    un bloque entero impide que el modelo reconstruya el pico copiandose de sus
    vecinos, que tambien estan contaminados.
  - Los 40 SNIa de evaluacion salen del fold de validacion. En la version
    anterior 37 de los 40 habian quedado dentro del entrenamiento y sus
    controles estaban memorizados, lo que inflaba la deteccion.

Sobre el eje en flujo absoluto. Cada objeto recibio df veces SU propio MAD_r, y
los MAD_r van de 2.2 a 9.7 FLUXCAL, asi que un mismo df NO corresponde a un
unico Delta F. Donde la metrica agrupa a todos los objetos se grafica el
promedio <Delta F> y se dice en la etiqueta; en el panel derecho de la figura 1
se usa el Delta F individual de cada objeto, sin promediar.

Uso: python scripts/plot_profe_mad300.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
# Con PRIOR_TAG=_folds se leen los resultados de los priors que comparten
# particion con el VQ-VAE, y el contorno pasa a salir del barrido extendido,
# que llega a 320 dias en vez de 80 y usa los objetos sin fuga.
TAG = os.environ.get('PRIOR_TAG', '')
EVAL = OUT / f'final_mad300{TAG}.json'
BARRIDO = OUT / (f'barrido_extendido{TAG}.json' if TAG else 'barrido_dt_df.json')
GANADORA, REFERENCIA = 'aleatoria|ventana', 'aleatoria|aleatoria'
DESTACADOS = [10, 50, 150, 500]
COLORS = ['#444444', '#4C78A8', '#F58518', '#54A24B', '#E45756']
PIE = '40 SNIa del fold de validacion: ningun modelo los vio en entrenamiento'


def roc_xy(neg, pos):
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
    y = y[np.argsort(-s)]
    tpr = np.r_[0.0, np.cumsum(y) / max(y.sum(), 1.0), 1.0]
    fpr = np.r_[0.0, np.cumsum(1.0 - y) / max((1.0 - y).sum(), 1.0), 1.0]
    trapz = getattr(np, 'trapezoid', np.trapz)
    return fpr, tpr, float(trapz(tpr, fpr))


def kde(x, grid):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros_like(grid)
    std = np.std(x)
    h = 1.06 * std * (len(x) ** -0.2) if std > 1e-6 else 0.3
    z = (grid[None, :] - x[:, None]) / h
    return np.exp(-0.5 * z ** 2).mean(axis=0) / (h * np.sqrt(2 * np.pi))


def prf(ctrl, pert, tau):
    """Precision, recall y F1 tratando los controles como los negativos."""
    tp = float((pert > tau).sum())
    fn = float((pert <= tau).sum())
    fp = float((ctrl > tau).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(EVAL))
    dfs = d['dfs']
    snids = sorted(d['objetos'])
    mads = {s: float(d['mads'][s]) for s in snids}
    mad_v = np.array([mads[s] for s in snids])
    mad_med = float(mad_v.mean())

    cel = d['celdas'][GANADORA]
    ref = d['celdas'][REFERENCIA]
    tau, tau_ref = cel['tau'], ref['tau']
    ctrl = np.array([cel['nll_control'][s] for s in snids])
    ctrl_ref = np.array([ref['nll_control'][s] for s in snids])
    pert = {k: np.array([cel['nll_pert'][str(k)][s] for s in snids]) for k in dfs}
    pert_ref = {k: np.array([ref['nll_pert'][str(k)][s] for s in snids]) for k in dfs}

    print(f'MAD_r: media={mad_med:.2f}  rango={mad_v.min():.2f}-{mad_v.max():.2f}')
    print(f'tau (P95 control) ganadora={tau:.3f}  referencia={tau_ref:.3f}')

    sub = (f'$\\Delta F = df \\times \\mathrm{{MAD}}_r$ de cada objeto; el eje es el '
           f'promedio sobre {len(snids)} SNIa '
           f'($\\mathrm{{MAD}}_r$ entre {mad_v.min():.1f} y {mad_v.max():.1f} FLUXCAL)')

    met = {k: prf(ctrl, pert[k], tau) for k in dfs}
    met_ref = {k: prf(ctrl_ref, pert_ref[k], tau_ref) for k in dfs}
    fluxes = np.array([k * mad_med for k in dfs])

    # --- 1. F1 y recall contra flujo absoluto -------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.0))
    x = np.arange(len(dfs))
    a1.bar(x - 0.2, [met[k][2] for k in dfs], width=0.4, color='steelblue', label='F1')
    a1.bar(x + 0.2, [met[k][1] for k in dfs], width=0.4, color='darkorange', label='recall')
    a1.plot(x, [met_ref[k][1] for k in dfs], 'o--', color='#8856a7', ms=4, lw=1.4,
            label='recall, mascara aleatoria al puntuar')
    for xi, k in zip(x, dfs):
        a1.text(xi - 0.2, met[k][2] + 0.02, f'{met[k][2]:.2f}', ha='center',
                va='bottom', fontsize=7.5)
        a1.text(xi + 0.2, met[k][1] + 0.02, f'{met[k][1]:.2f}', ha='center',
                va='bottom', fontsize=7.5)
    a1.set_xticks(x)
    a1.set_xticklabels([f'{f:.0f}' for f in fluxes], fontsize=8)
    a1.set_ylim(0, 1.18)
    a1.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
    a1.set_ylabel('F1 / recall')
    a1.set_title('Promediando los objetos en cada nivel', fontsize=10)
    a1.legend(frameon=False, fontsize=8, loc='upper left')

    # Panel derecho: sin promediar. Cada par (objeto, df) aporta su propio
    # Delta F = df * MAD_r del objeto, y se agrupa por Delta F real.
    df_i, det_i = [], []
    for k in dfs:
        for j, s in enumerate(snids):
            df_i.append(k * mads[s])
            det_i.append(pert[k][j] > tau)
    df_i, det_i = np.array(df_i), np.array(det_i, float)
    bordes = np.logspace(np.log10(df_i.min() * 0.99), np.log10(df_i.max() * 1.01), 13)
    cent, frac, nn = [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (df_i >= lo) & (df_i < hi)
        if m.sum() >= 5:
            cent.append(np.sqrt(lo * hi))
            frac.append(det_i[m].mean())
            nn.append(int(m.sum()))
    a2.plot(cent, frac, 'o-', color='#31a354', lw=2, ms=5)
    for xc, yc, n in zip(cent, frac, nn):
        a2.annotate(f'n={n}', (xc, yc), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.5, color='#555555')
    a2.axhline(0.05, color='crimson', ls=':', lw=1.2, label='tasa de falsos positivos (5%)')
    a2.set_xscale('log')
    a2.set_ylim(0, 1.05)
    a2.set_xlabel(r'$\Delta F$ real del objeto  [FLUXCAL]')
    a2.set_ylabel('fraccion detectada')
    a2.set_title(r'Sin promediar: cada objeto en su propio $\Delta F$', fontsize=10)
    a2.legend(frameon=False, fontsize=8, loc='upper left')

    fig.suptitle('Deteccion por el maximo NLL, codificacion mad300, puntuado con '
                 f'ventana contigua\n{sub}', fontsize=10)
    fig.text(0.5, 0.015, PIE, ha='center', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    fig.savefig(OUT / 'f1_recall_vs_flux.png', dpi=140)
    plt.close(fig)

    # --- 2. distribucion del maximo NLL --------------------------------------
    series = [(ctrl, 'sin perturbacion')]
    for k in DESTACADOS:
        series.append((pert[k], f'{k}$\\times$MAD  '
                                f'($\\langle\\Delta F\\rangle$={k * mad_med:.0f})'))
    lo = min(a.min() for a, _ in series) - 1.0
    hi = max(a.max() for a, _ in series) + 1.0
    grid = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for (arr, lab), c in zip(series, COLORS):
        dens = kde(arr, grid)
        ax.plot(grid, dens, color=c, lw=2, label=lab)
        ax.fill_between(grid, dens, color=c, alpha=0.12)
    ax.axvline(tau, color='crimson', ls='--', lw=1.2,
               label=f'$\\tau$ = P95 control = {tau:.2f}')
    ax.set_xlabel(r'$\max_{ij}\, S_{ij}$,   $S_{ij} = -\log p(s_{ij}\,|\,\mathrm{resto})$')
    ax.set_ylabel('densidad')
    ax.set_title('Distribucion del maximo NLL sobre el mapa 32$\\times$32\n'
                 'mad300, puntuado con ventana contigua', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.015, PIE, ha='center', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / 'dist_max_nll.png', dpi=140)
    plt.close(fig)

    # --- 3. ROC por nivel de Delta F -----------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 1], [0, 1], color='gray', lw=0.8, ls='--', label='azar')
    aucs = {}
    for k, c in zip(DESTACADOS, COLORS[1:]):
        fpr, tpr, a = roc_xy(ctrl, pert[k])
        aucs[k] = a
        ax.plot(fpr, tpr, color=c, lw=2.2,
                label=f'{k}$\\times$MAD  ($\\langle\\Delta F\\rangle$={k * mad_med:.0f}, '
                      f'AUC={a:.2f})')
    for k, c in zip(DESTACADOS, COLORS[1:]):
        fpr, tpr, a = roc_xy(ctrl_ref, pert_ref[k])
        ax.plot(fpr, tpr, color=c, lw=1.0, ls=':', alpha=0.75)
    ax.plot([], [], color='gray', lw=1.0, ls=':', label='mismo nivel, mascara aleatoria')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR (recall)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ROC del detector por maximo NLL\n'
                 'llena: ventana contigua   punteada: mascara aleatoria', fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.text(0.5, 0.015, PIE, ha='center', fontsize=7.5, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / 'roc_deltaf.png', dpi=140)
    plt.close(fig)

    meta = {
        'n_objetos': len(snids),
        'celda': GANADORA,
        'mad_r': {'media': mad_med, 'min': float(mad_v.min()), 'max': float(mad_v.max())},
        'tau_p95_control': tau,
        'delta_f_medio_por_df': {str(k): k * mad_med for k in dfs},
        'f1_por_df': {str(k): met[k][2] for k in dfs},
        'recall_por_df': {str(k): met[k][1] for k in dfs},
        'precision_por_df': {str(k): met[k][0] for k in dfs},
        'auc_destacados': {str(k): aucs[k] for k in DESTACADOS},
        'recall_referencia_por_df': {str(k): met_ref[k][1] for k in dfs},
    }

    # --- 4. contorno de recall en (Delta F, Delta t) -------------------------
    if BARRIDO.exists():
        b = json.load(open(BARRIDO))
        if TAG:
            # El barrido extendido indexa por 'df|dt' y nombra las celdas solo
            # por la forma de puntuacion, porque el prior es siempre el aleatorio.
            dts = [int(t) for t in b['dts']]
            dfs_b = [int(v) for v in b['dfs_2d']]
            mad_b = float(np.mean([float(v) for v in b['mads'].values()]))
            etiq_y = {str(t): f'{100 * b["cobertura_dt"][str(t)]:.0f}%' for t in dts}
            recalls = {
                c: np.array([[b['celdas'][f.split('|')[1]]['dos_d'][f'{d}|{t}']['recall']
                              for d in dfs_b] for t in dts], float)
                for c, f in ((GANADORA, GANADORA), (REFERENCIA, REFERENCIA))
            }
        else:
            dts = [int(t) for t in b['dts']]
            dfs_b = b['dfs']
            mad_b = b['mad_medio']
            etiq_y = {k: f'{int(np.median(v))} pts' for k, v in b['puntos_por_dt'].items()}
            recalls = {c: np.array(b['celdas'][c]['recall'], float)
                       for c in (GANADORA, REFERENCIA)}
        x_flux = np.array(dfs_b, float) * mad_b

        # Los df y los dt no estan igualmente espaciados, y dibujar el mapa en sus
        # coordenadas reales aplasta las filas de abajo hasta hacerlas ilegibles.
        # Se usa una grilla categorica y las etiquetas llevan el valor real.
        ix, iy = np.arange(len(x_flux)), np.arange(len(dts))
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            gaussian_filter = None

        paneles = [(GANADORA, 'Puntuado con ventana contigua'),
                   (REFERENCIA, 'Puntuado con mascara aleatoria (referencia)')]
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
        rec_g = None
        for ax, (clave, titulo) in zip(axes, paneles):
            rec = recalls[clave]
            if clave == GANADORA:
                rec_g = rec
            im = ax.pcolormesh(ix, iy, rec, shading='nearest', cmap='Blues',
                               vmin=0, vmax=1)
            for i in range(len(dts)):
                for j in range(len(x_flux)):
                    v = rec[i, j]
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if v > 0.6 else '#22303c')
            # Las isolineas van sobre una version suavizada: con 40 objetos el
            # recall tiene pasos de 0.025 y las curvas crudas salen dentadas.
            if gaussian_filter is not None:
                suave = gaussian_filter(rec, sigma=0.7, mode='nearest')
                niveles = [lv for lv in (0.5, 0.75, 0.9)
                           if suave.min() < lv < suave.max()]
                if niveles:
                    cs = ax.contour(ix, iy, suave, levels=niveles, colors='#0b2545',
                                    linewidths=1.6)
                    ax.clabel(cs, fmt=lambda v: f'{100 * v:.0f}%', fontsize=8)
            ax.set_xticks(ix)
            ax.set_xticklabels([f'{f:.0f}' for f in x_flux], fontsize=8)
            ax.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
            ax.set_title(f'{titulo}\nrecall {100 * rec.min():.0f}% a '
                         f'{100 * rec.max():.0f}%', fontsize=10)
        axes[0].set_yticks(iy)
        axes[0].set_yticklabels([f'{t}  ({etiq_y[str(t)]})' for t in dts], fontsize=8)
        axes[0].set_ylabel(r'$\Delta t$ de la ventana perturbada  [dias]')
        fig.colorbar(im, ax=axes, label='recall', fraction=0.03, pad=0.02)
        fig.suptitle('Recall en el plano $(\\Delta F,\\,\\Delta t)$, mad300 sobre 1.39M   '
                     '(isolineas suavizadas; los numeros son el recall medido)',
                     fontsize=10)
        fig.text(0.5, 0.015, PIE, ha='center', fontsize=8, style='italic',
                 color='#555555')
        fig.savefig(OUT / 'contour_dt_dF.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        meta['contorno'] = {'dts': dts, 'dfs': dfs_b,
                            'delta_f_medio': x_flux.tolist(),
                            'recall_ganadora': rec_g.tolist(),
                            'recall_referencia': recalls[REFERENCIA].tolist(),
                            'etiqueta_por_dt': etiq_y}
        print('contorno listo')
    else:
        print(f'sin {BARRIDO}: el contorno queda pendiente')

    with open(OUT / 'flux_axis.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Figuras en {OUT}/')


if __name__ == '__main__':
    main()
