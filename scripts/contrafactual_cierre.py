"""Cierra los dos huecos de medicion que quedaron inferidos y no medidos.

HUECO 1: el tamano minimo de anomalia. En la corrida anterior solo se guardaron
metricas relativas, y con anomalias de uno o dos puntos el denominador colapsa y
las razones explotan hasta 1000% sin que eso signifique que el dano absoluto
crecio. Aca se guardan los MSE absolutos dentro y fuera de la zona, y se
convierte "hace mas dano que bien" en algo verificable:

    ganancia = (MSE_zona_perturbada - MSE_zona_contrafactual) * area_zona
    perdida  = (MSE_fuera_contrafactual - MSE_fuera_perturbada) * area_fuera
    neto     = ganancia - perdida

en unidades de error cuadratico total contra la reconstruccion de la curva
limpia. El neto pesa cada pixel igual, y la zona es solo el 4.9% de la imagen,
asi que ese criterio es exigente; se reportan tambien los dos terminos por
separado para que se pueda pesar distinto. Se registra ademas cuantos puntos
fotometricos reales cae en cada ventana de dt.

HUECO 2: la referencia de plausibilidad. Estaba construida sobre 40 curvas y la
desviacion estandar de la correlacion entre bandas era 0.155 sobre una media de
0.260, demasiado ancha para sostener nada. Se amplia a varios cientos de curvas
limpias del fold de validacion, que es barato porque solo hay que renderizar,
reconstruir y medir, sin pasar por el prior. Con la banda mas firme se vuelve a
preguntar si el sobresuavizado sigue ahi y si es significativo.

Se agrega un control que faltaba: la rugosidad de la imagen RENDERIZADA antes de
pasar por el VQ-VAE. Sirve para saber cuanto del suavizado lo pone el
autoencoder y cuanto el contrafactual. La comparacion principal ya es
reconstruccion contra reconstruccion, asi que el efecto del VQ-VAE esta
controlado, pero conviene saber su tamano.

Uso: python scripts/contrafactual_cierre.py
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_enc_cmp import col_flux, col_mjd, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import GRID, PX, reconstruir, zona_px
from scripts.contrafactual_diagnostico import parte
from scripts.contrafactual_comparacion import metodo_base, metodo_rect, plausibilidad
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
N_REF = 400
DTS = [5, 10, 15, 20, 30, 45, 60, 90]
DF_DT = 300
DFS_PLAUS = [100, 500]
METODOS = ['base p85', 'base p95', 'rect 1 pasa']
SEED = 0


def welch(a, b):
    """t de Welch y p normal aproximada. n grande, no hace falta scipy."""
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(t), float(p), float(d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    pz = zona_px(*zona)
    area_tot = PX * PX
    area_z = (pz[1] - pz[0]) * (pz[3] - pz[2])
    area_f = area_tot - area_z
    print(f'area zona {area_z} px, area fuera {area_f} px '
          f'({100 * area_z / area_tot:.1f}% / {100 * area_f / area_tot:.1f}%)',
          flush=True)

    def fuera(a, b):
        tot = float(F.mse_loss(a, b).item())
        return (tot * area_tot - parte(a, b, pz) * area_z) / area_f

    curvas = candidatos(N_REF)
    todos = list(curvas)
    with open(OUT / 'contrafactual_comparacion.json') as f:
        los40 = json.load(f)['objetos']
    faltan = [s for s in los40 if s not in curvas]
    assert not faltan, f'los 40 originales no estan en los {N_REF}: {faltan[:5]}'
    print(f'referencia: {len(todos)} curvas limpias; '
          f'metodos evaluados sobre los mismos {len(los40)} objetos de antes',
          flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    @torch.no_grad()
    def codificar(img):
        x = tf(img).unsqueeze(0).to(dev)
        t = vq.encode_to_indices(x).view(1, -1).long().to(dev)
        return t, reconstruir(vq, t)

    # ---------------------------------------------------- HUECO 2: referencia
    print(f'\n{"=" * 78}\nREFERENCIA DE PLAUSIBILIDAD SOBRE {N_REF} CURVAS LIMPIAS',
          flush=True)
    ref_rec, ref_raw = [], []
    limpias40 = {}
    for i, s in enumerate(todos):
        o, _, _ = curvas[s]
        img = render(o, ENC, fig, ax)
        t, xr = codificar(img)
        ref_rec.append(plausibilidad(xr))
        ref_raw.append(plausibilidad(tf(img).unsqueeze(0).to(dev)))
        if s in set(los40):
            limpias40[s] = (t, xr)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(todos)}', flush=True)

    R = {k: np.array([r[k] for r in ref_rec]) for k in ('rugosidad', 'corr_bandas')}
    Q = {k: np.array([r[k] for r in ref_raw]) for k in ('rugosidad', 'corr_bandas')}
    for k in R:
        R[k] = R[k][np.isfinite(R[k])]
        Q[k] = Q[k][np.isfinite(Q[k])]
    ee = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    print(f'\n  reconstruidas (n={len(R["rugosidad"])}):')
    print(f'    rugosidad          {R["rugosidad"].mean():.4f} +- {R["rugosidad"].std(ddof=1):.4f}'
          f'   (error estandar {ee(R["rugosidad"]):.4f})')
    print(f'    corr entre bandas  {R["corr_bandas"].mean():.4f} +- {R["corr_bandas"].std(ddof=1):.4f}'
          f'   (error estandar {ee(R["corr_bandas"]):.4f})')
    print(f'  renderizadas sin pasar por el VQ-VAE (n={len(Q["rugosidad"])}):')
    print(f'    rugosidad          {Q["rugosidad"].mean():.4f} +- {Q["rugosidad"].std(ddof=1):.4f}')
    print(f'    corr entre bandas  {Q["corr_bandas"].mean():.4f} +- {Q["corr_bandas"].std(ddof=1):.4f}')
    print(f'  -> el VQ-VAE por si solo cambia la rugosidad en '
          f'{100 * (R["rugosidad"].mean() - Q["rugosidad"].mean()) / Q["rugosidad"].mean():+.1f}%',
          flush=True)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    print(f'\n  plausibilidad de cada metodo contra esa referencia:')
    print(f'  {"df":>5} {"metodo":<13}{"rugosidad":>11}{"z":>7}{"d Cohen":>9}{"p":>10}'
          f'{"corr":>8}{"z":>7}')
    plaus = {}
    for d in DFS_PLAUS:
        for m in METODOS:
            plaus[f'{d}|{m}'] = {'rug': [], 'corr': []}
        for s in los40:
            o, idx_cola, sc = curvas[s]
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            tp, _ = codificar(render(p, ENC, fig, ax))
            salidas = {'base p85': metodo_base(mg, tp, dev, g, 0.85),
                       'base p95': metodo_base(mg, tp, dev, g, 0.95),
                       'rect 1 pasa': metodo_rect(mg, tp, dev, g)}
            for m, (cf, _) in salidas.items():
                pl = plausibilidad(reconstruir(vq, cf))
                plaus[f'{d}|{m}']['rug'].append(pl['rugosidad'])
                plaus[f'{d}|{m}']['corr'].append(pl['corr_bandas'])
        for m in METODOS:
            a = plaus[f'{d}|{m}']
            t_, p_, dc = welch(a['rug'], R['rugosidad'])
            zr = (np.nanmean(a['rug']) - R['rugosidad'].mean()) / R['rugosidad'].std(ddof=1)
            zc = (np.nanmean(a['corr']) - R['corr_bandas'].mean()) / R['corr_bandas'].std(ddof=1)
            plaus[f'{d}|{m}'].update({'z_rug': zr, 'z_corr': zc, 't': t_, 'p': p_,
                                      'd_cohen': dc})
            print(f'  {d:>5} {m:<13}{np.nanmean(a["rug"]):11.4f}{zr:7.2f}{dc:9.2f}'
                  f'{p_:10.2e}{np.nanmean(a["corr"]):8.3f}{zc:7.2f}', flush=True)
    print(f'  {"":>5} {"reales limpias":<13}{R["rugosidad"].mean():11.4f}'
          f'{0.0:7.2f}{"":>9}{"":>10}{R["corr_bandas"].mean():8.3f}{0.0:7.2f}')

    # ------------------------------------------------------- HUECO 1: umbral
    print(f'\n{"=" * 78}\nUMBRAL DE TAMANO: MSE ABSOLUTOS, df={DF_DT}', flush=True)
    ventanas, n_pts = {}, {}
    for s in los40:
        o, idx_cola, _ = curvas[s]
        sub = o.loc[idx_cola]
        t_fin = float(sub[col_mjd].max())
        for dt in DTS:
            idx = list(sub.index[sub[col_mjd] >= t_fin - dt])
            ventanas[(s, dt)] = idx
    print('\n  puntos fotometricos reales por ventana (banda r):')
    print(f'  {"dt[dias]":>9}{"mediana":>9}{"media":>8}{"min":>6}{"max":>6}{"total":>8}')
    for dt in DTS:
        c = np.array([len(ventanas[(s, dt)]) for s in los40])
        n_pts[dt] = c.tolist()
        print(f'  {dt:9d}{np.median(c):9.1f}{c.mean():8.2f}{c.min():6d}{c.max():6d}'
              f'{c.sum():8d}', flush=True)

    res = {m: {} for m in METODOS}
    for dt in DTS:
        acc = {m: [] for m in METODOS}
        for s in los40:
            o, _, sc = curvas[s]
            p = o.copy()
            p.loc[ventanas[(s, dt)], col_flux] = \
                p.loc[ventanas[(s, dt)], col_flux] + DF_DT * sc
            tp, xp = codificar(render(p, ENC, fig, ax))
            _, xl = limpias40[s]
            bz, bf = parte(xp, xl, pz), fuera(xp, xl)
            salidas = {'base p85': metodo_base(mg, tp, dev, g, 0.85),
                       'base p95': metodo_base(mg, tp, dev, g, 0.95),
                       'rect 1 pasa': metodo_rect(mg, tp, dev, g)}
            for m, (cf, _) in salidas.items():
                xc = reconstruir(vq, cf)
                mz, mf = parte(xc, xl, pz), fuera(xc, xl)
                acc[m].append({
                    'mse_z_pert': bz, 'mse_f_pert': bf, 'mse_z_cf': mz, 'mse_f_cf': mf,
                    'ganancia': (bz - mz) * area_z,
                    'perdida': (mf - bf) * area_f,
                    'neto': (bz - mz) * area_z - (mf - bf) * area_f,
                })

        print(f'\n  dt={dt}d   {np.mean(n_pts[dt]):.1f} puntos perturbados en promedio')
        print(f'  {"metodo":<13}{"MSE zona":>19}{"MSE fuera":>21}'
              f'{"ganancia":>10}{"perdida":>10}{"neto":>9}')
        print(f'  {"":13}{"pert -> cf":>19}{"pert -> cf":>21}')
        for m in METODOS:
            a = acc[m]
            f_ = lambda k: float(np.mean([x[k] for x in a]))
            res[m][str(dt)] = {k: f_(k) for k in a[0]}
            res[m][str(dt)]['neto_positivo'] = float(np.mean([x['neto'] > 0 for x in a]))
            print(f'  {m:<13}{f_("mse_z_pert"):9.5f}->{f_("mse_z_cf"):8.5f}'
                  f'{f_("mse_f_pert"):11.5f}->{f_("mse_f_cf"):8.5f}'
                  f'{f_("ganancia"):10.1f}{f_("perdida"):10.1f}{f_("neto"):9.1f}',
                  flush=True)
    plt.close(fig)

    salida = {'n_ref': len(todos), 'dts': DTS, 'df_dt': DF_DT,
              'area_zona_px': area_z, 'area_fuera_px': area_f,
              'puntos_por_dt': n_pts,
              'referencia': {
                  'reconstruidas': {k: {'media': float(v.mean()),
                                        'sd': float(v.std(ddof=1)),
                                        'ee': float(ee(v)), 'n': int(len(v))}
                                    for k, v in R.items()},
                  'renderizadas': {k: {'media': float(v.mean()),
                                       'sd': float(v.std(ddof=1))}
                                   for k, v in Q.items()}},
              'plausibilidad_metodos': {
                  k: {kk: (vv if not isinstance(vv, list) else
                           [float(x) for x in vv]) for kk, vv in v.items()}
                  for k, v in plaus.items()},
              'umbral_absoluto': res}
    with open(OUT / 'contrafactual_cierre.json', 'w') as f:
        json.dump(salida, f, indent=2)

    figura(res, n_pts, R, Q, plaus)
    print(f'\nGuardado en {OUT}/', flush=True)


def figura(res, n_pts, R, Q, plaus):
    COL = {'base p85': '#e45756', 'base p95': '#f2a640', 'rect 1 pasa': '#31a354'}
    MRK = {'base p85': 'o', 'base p95': 'v', 'rect 1 pasa': '^'}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    a = axes[0]
    for m in METODOS:
        a.plot(DTS, [res[m][str(dt)]['ganancia'] for dt in DTS], MRK[m] + '-',
               color=COL[m], label=f'{m}: gana en la zona', ms=6)
        a.plot(DTS, [res[m][str(dt)]['perdida'] for dt in DTS], MRK[m] + '--',
               color=COL[m], alpha=0.45, label=f'{m}: pierde afuera', ms=5)
    a.set_xlabel('dt: duracion de la anomalia [dias]')
    a.set_ylabel('error cuadratico total')
    a.set_title('Los dos terminos por separado', fontsize=10)
    a.legend(frameon=False, fontsize=6.5, ncol=1)

    a = axes[1]
    for m in METODOS:
        a.plot(DTS, [res[m][str(dt)]['neto'] for dt in DTS], MRK[m] + '-',
               color=COL[m], label=m, ms=6)
    a.axhline(0, color='#333333', lw=1.2)
    a.set_xlabel('dt: duracion de la anomalia [dias]')
    a.set_ylabel('neto = ganancia - perdida')
    a.set_title('Neto en error cuadratico total\nsobre cero, repara mas de lo que rompe',
                fontsize=10)
    a.legend(frameon=False, fontsize=8)
    b = a.twiny()
    b.set_xlim(a.get_xlim())
    b.set_xticks(DTS)
    b.set_xticklabels([f'{np.mean(n_pts[dt]):.1f}' for dt in DTS], fontsize=7)
    b.set_xlabel('puntos fotometricos perturbados (promedio)', fontsize=8)

    a = axes[2]
    lo, hi = R['rugosidad'].mean() - R['rugosidad'].std(ddof=1), \
        R['rugosidad'].mean() + R['rugosidad'].std(ddof=1)
    a.axhspan(lo, hi, color='#333333', alpha=0.12,
              label=f'reales limpias, +-1 sd (n={len(R["rugosidad"])})')
    a.axhline(R['rugosidad'].mean(), color='#333333', ls='--', lw=1.3)
    for i, d in enumerate(DFS_PLAUS):
        for m in METODOS:
            v = plaus[f'{d}|{m}']['rug']
            a.errorbar(i + 0.22 * (METODOS.index(m) - 1), np.nanmean(v),
                       yerr=np.nanstd(v) / np.sqrt(len(v)), fmt=MRK[m],
                       color=COL[m], ms=7, capsize=3,
                       label=m if i == 0 else None)
    a.set_xticks(range(len(DFS_PLAUS)))
    a.set_xticklabels([f'df={d}' for d in DFS_PLAUS])
    a.set_ylabel('rugosidad de las trazas')
    a.set_title('Sobresuavizado contra la referencia ampliada', fontsize=10)
    a.legend(frameon=False, fontsize=7.5)

    fig.suptitle('Cierre: umbral de tamano en unidades absolutas y referencia de '
                 'plausibilidad ampliada', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / 'contrafactual_cierre.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
