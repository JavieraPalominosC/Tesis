"""
Analisis POR SUPERNOVA del experimento de cola MAD.

El detector global compara el max NLL de cada objeto contra un umbral
poblacional (P95 de los 40 controles). Eso mezcla dos cosas: cuanto subio
el NLL por la perturbacion, y cuan alto era el basal de ese objeto. Aca se
separan.

  1) matriz_deteccion.png    40 objetos x 10 niveles, detectado si o no,
                             ordenados por su max NLL de control
  2) deteccion_pareada.png   fraccion de objetos que suben respecto de SU
                             control, comparada con el recall del umbral global
  3) umbral_por_objeto.png   minimo Delta F detectable por objeto vs su MAD_r

Nada de esto necesita GPU: sale del summary.json que ya existe.

Uso: python scripts/plot_per_object.py
     PROFE_FIG_OUT=/tmp/imagenes python scripts/plot_per_object.py
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

TAIL_CANDIDATES = [
    Path('results/synthetic_eval/mad_tail/summary.json'),
    Path('reunion_profe_2026-08-24/mad_dt/tail/summary.json'),
]
MANIFEST_CANDIDATES = [Path('data/synthetic_mad_tail_manifest.json')]
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
DFS = list(range(10, 101, 10))


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tail = json.load(open(first_existing(TAIL_CANDIDATES)))
    tau = float(tail['tau_max'])

    by = {}
    for r in tail['per_image']:
        by.setdefault(r['snid'], {})[r['tag']] = r

    man = first_existing(MANIFEST_CANDIDATES)
    mads, npts = {}, {}
    if man is not None:
        for r in json.load(open(man)):
            if r.get('mad'):
                mads.setdefault(r['snid'], float(r['mad']))
            if r.get('n_pts'):
                npts.setdefault(r['snid'], int(r['n_pts']))

    dfs = [d for d in DFS if f'df{d}_unfrozen' in tail['per_tag']]
    snids = sorted(s for s in by
                   if 'control' in by[s] and all(f'df{d}_unfrozen' in by[s] for d in dfs))
    ctrl = np.array([by[s]['control']['max_nll'] for s in snids])
    # M[i, j] = max NLL del objeto i al nivel df_j
    M = np.array([[by[s][f'df{d}_unfrozen']['max_nll'] for d in dfs] for s in snids])
    mean_mad = float(np.mean([mads[s] for s in snids])) if mads else 4.06
    fluxes = np.array([d * mean_mad for d in dfs])

    order = np.argsort(ctrl)
    print(f'{len(snids)} objetos, {len(dfs)} niveles', flush=True)
    print(f'max NLL de control: {ctrl.min():.2f} a {ctrl.max():.2f} '
          f'(mediana {np.median(ctrl):.2f}), tau={tau:.2f}', flush=True)

    # --- 1. matriz de deteccion -------------------------------------------
    det = (M > tau).astype(int)
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.imshow(det[order], aspect='auto', cmap=ListedColormap(['#f0f0f0', '#2c7bb6']),
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(dfs)), [f'{d}' for d in dfs])
    ax.set_yticks(range(len(snids)),
                  [f'{snids[i]}  (ctrl {ctrl[i]:.1f})' for i in order], fontsize=6.5)
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_title('Deteccion objeto por objeto con el umbral global\n'
                 f'azul = max NLL > $\\tau$ = {tau:.2f}. '
                 'Ordenados por su max NLL de control (arriba, mas bajo)',
                 fontsize=10)
    for k in range(len(dfs) + 1):
        ax.axvline(k - 0.5, color='white', lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / 'matriz_deteccion.png', dpi=140)
    plt.close(fig)

    n_nunca = int((det.sum(1) == 0).sum())
    n_siempre = int((det.sum(1) == len(dfs)).sum())
    print(f'objetos nunca detectados: {n_nunca}/{len(snids)}  '
          f'detectados en todos los niveles: {n_siempre}', flush=True)

    # --- 2. detector pareado vs detector global ---------------------------
    # pareado: la perturbacion subio el max NLL respecto del control del MISMO objeto
    subio = (M > ctrl[:, None]).mean(0)
    recall_glob = np.array([tail['per_tag'][f'df{d}_unfrozen']['max_detector']['recall']
                            for d in dfs])
    dmax = M - ctrl[:, None]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax.plot(fluxes, subio, 'o-', color='#2c7bb6', lw=2,
            label='pareado: max NLL > su propio control')
    ax.plot(fluxes, recall_glob, 's--', color='darkorange', lw=2,
            label=f'global: max NLL > $\\tau$={tau:.2f}')
    # El score usa 10 rondas de enmascaramiento al azar, asi que re-puntuar la
    # misma imagen ya da valores distintos: sin perturbacion el criterio
    # pareado daria ~0.5. Ese es el piso, no 0.
    ax.axhline(0.5, color='gray', lw=1.0, ls=':')
    ax.text(fluxes[0], 0.52, 'piso esperado sin perturbacion (~0.5)',
            fontsize=7.5, color='gray')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
    ax.set_ylabel('fraccion de objetos detectados')
    ax.set_title('Comparar cada supernova consigo misma\n'
                 'el piso pareado es 0.5, falta calibrarlo', fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax2.boxplot([dmax[:, j] for j in range(len(dfs))],
                tick_labels=[f'{f:.0f}' for f in fluxes], showfliers=False)
    ax2.axhline(0, color='crimson', lw=1, ls='--')
    ax2.set_xlabel(r'$\langle\Delta F\rangle$  [FLUXCAL]')
    ax2.set_ylabel(r'$\Delta$max NLL  (perturbada $-$ control)')
    ax2.set_title('Cuanto sube el max NLL respecto de su control', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'deteccion_pareada.png', dpi=140)
    plt.close(fig)

    for d, a, b in zip(dfs, subio, recall_glob):
        print(f'  df={d:3d}  pareado={a:.2f}  global={b:.2f}', flush=True)

    # --- 3. umbral minimo por objeto --------------------------------------
    if mads:
        mad_i = np.array([mads[s] for s in snids])
        n_i = np.array([npts.get(s, 0) for s in snids])
        # Cada objeto recibio df * SU propio MAD, no el promedio.
        primero = []
        for i in range(len(snids)):
            hit = np.where(det[i] == 1)[0]
            primero.append(dfs[hit[0]] * mad_i[i] if len(hit) else np.nan)
        primero = np.array(primero, float)

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        ok = np.isfinite(primero)
        ax.scatter(mad_i[ok], primero[ok], c=n_i[ok], cmap='viridis', s=45)
        techo = float(np.nanmax(primero)) * 1.1 if ok.any() else 1.0
        for x0 in mad_i[~ok]:
            ax.scatter([x0], [techo], marker='^', color='crimson', s=40)
        ax.set_yscale('log')
        ax.set_xlabel(r'MAD$_r$ del objeto  [FLUXCAL]')
        ax.set_ylabel(r'primer $\Delta F$ detectado del objeto  [FLUXCAL]')
        ax.set_title(f'Umbral de deteccion por objeto\n'
                     f'triangulos rojos = nunca detectados ({int((~ok).sum())})',
                     fontsize=10)
        sc = ax.collections[0]
        fig.colorbar(sc, ax=ax, label='puntos perturbados en la cola')

        ax2.scatter(mad_i, ctrl, s=45, color='#444444')
        ax2.axhline(tau, color='crimson', ls='--', lw=1, label=f'$\\tau$={tau:.2f}')
        ax2.set_xlabel(r'MAD$_r$ del objeto  [FLUXCAL]')
        ax2.set_ylabel('max NLL del control')
        ax2.set_title('El basal ya varia mucho entre objetos\n'
                      'por eso un umbral unico pierde señal', fontsize=10)
        ax2.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / 'umbral_por_objeto.png', dpi=140)
        plt.close(fig)

        r = np.corrcoef(mad_i[ok], primero[ok])[0, 1] if ok.sum() > 2 else np.nan
        print(f'corr(MAD, primer DeltaF detectado) = {r:.2f}', flush=True)

    meta = {
        'n_objetos': len(snids),
        'tau_global': tau,
        'control_max_nll': {'min': float(ctrl.min()), 'max': float(ctrl.max()),
                            'mediana': float(np.median(ctrl))},
        'nunca_detectados': n_nunca,
        'deteccion_pareada': dict(zip(map(str, dfs), map(float, subio))),
        'recall_global': dict(zip(map(str, dfs), map(float, recall_glob))),
    }
    with open(OUT / 'per_object.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Figuras en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
