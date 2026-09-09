"""Etapa FLUXCAL de la ablacion: solo las configuraciones que sobrevivieron.

El criterio de evaluacion NO cambia respecto de contrafactual_extractor.py, que
es lo unico que hace comparable el resultado contra el 29.2% de referencia:

  - mismo extractor unico, umbral calibrado 0.73, aplicado identico a la limpia,
    la perturbada y el contrafactual;
  - misma imputacion conservadora, la epoca borrada se cuenta con el flujo de la
    perturbada, o sea como no reparada;
  - mismo conjunto de objetos, candidatos(300) del fold de validacion;
  - misma epoca base, solo entra si la perturbada es legible.

La unica diferencia es que el contrafactual se genera con (prior, forma de
mascara, decodificacion, n_steps) variables en vez de la combinacion fija
aleatoria / rect / muestreo recocido / 10 pasos.

Uso: python scripts/contrafactual_flujo_ablacion.py [N_OBJ]
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

from scripts.eval_enc_cmp import col_flux, col_mjd, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import PX, reconstruir
from scripts.contrafactual_comparacion import (
    GRID, R_LOC, nll_por_token, rect_max_exceso, seleccionar,
)
from scripts.contrafactual_flujo import col_de
from scripts.contrafactual_extractor import (
    DFS, J_R, banda_dominante, invertir_u, pesos, rmse, stats,
)
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
U = 0.73            # umbral calibrado, se hereda tal cual, no se recalibra
N_OBJ, SEED, N_AZAR = 300, 0, 148

# nombre -> (prior, forma, modo de decodificacion, n_steps).
# La etapa 2 dejo claro que lo que manda es cuanto de la zona real de inyeccion
# cubre la mascara, no el prior ni la temperatura, asi que a flujo pasan las
# formas de mayor cobertura, la referencia para poder comparar, el oraculo como
# techo y dos controles de posicion al azar.
CONFIGS = {
    'referencia': ('aleatoria', 'rect', 'actual', 10),
    'base p95 ref': ('aleatoria', 'p95', 'actual', 10),
    'rect m1 topk10': ('aleatoria', 'rect m1', 'topk10', 10),
    'rect m2 actual': ('aleatoria', 'rect m2', 'actual', 10),
    'rect m2 topk10': ('aleatoria', 'rect m2', 'topk10', 10),
    'oraculo topk10': ('aleatoria', 'oraculo', 'topk10', 10),
    'azar pareado m2': ('aleatoria', 'azar m2', 'topk10', 10),
    'azar referencia': ('aleatoria', 'azar', 'actual', 10),
}
DECOD = {'actual': dict(modo='muestreo', temperatura=True),
         'greedy': dict(modo='greedy'),
         'topk3': dict(modo='topk', k=3),
         'topk10': dict(modo='topk', k=10)}


def azar(T, n, dev, g):
    m = torch.zeros(1, T, dtype=torch.bool, device=dev)
    m[0, torch.randperm(T, device=dev, generator=g)[:max(1, n)]] = True
    return m


@torch.no_grad()
def mascaras(mg, tp, dev, g, zona):
    """Todas las mascaras del objeto, de un solo mapa de NLL."""
    T = tp.shape[1]
    nll = nll_por_token(mg, tp, dev, g, rondas=R_LOC)
    sel, _ = seleccionar(nll, 0.95)
    r0, r1, c0, c1, _, _ = rect_max_exceso(sel.reshape(GRID, GRID))
    out = {'p95': sel.reshape(GRID, GRID), 'oraculo': zona}
    for mg_ in (0, 1, 2):
        z = np.zeros((GRID, GRID), bool)
        z[max(0, r0 - mg_):min(GRID, r1 + 1 + mg_),
          max(0, c0 - mg_):min(GRID, c1 + 1 + mg_)] = True
        out[{0: 'rect', 1: 'rect m1', 2: 'rect m2'}[mg_]] = z
    d = {k: torch.from_numpy(v.reshape(1, -1)).to(dev) for k, v in out.items()}
    # El control al azar se aparea en numero de tokens con la mascara que
    # controla: sin eso la diferencia se confunde con cuanto se reescribio.
    d['azar m2'] = azar(T, int(out['rect m2'].sum()), dev, g)
    d['azar'] = azar(T, N_AZAR, dev, g)
    return d


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    priors = sorted({c[0] for c in CONFIGS.values()})
    mgs = {p: MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS[p]), map_location=dev,
                                           strict=False).eval().to(dev) for p in priors}
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    curvas = candidatos(n_obj)
    snids = sorted(curvas)
    zr0, zr1, zc0, zc1 = zona_objetivo()
    zona = np.zeros((GRID, GRID), bool)
    zona[zr0:zr1 + 1, zc0:zc1 + 1] = True
    print(f'objetos {len(snids)}   umbral heredado {U}   configs {len(CONFIGS)}',
          flush=True)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def tok(img):
        return vq.encode_to_indices(tf(img).unsqueeze(0).to(dev)).view(1, -1).long().to(dev)

    filas, color = [], []
    for df in DFS:
        for s in snids:
            o, idx, sc = curvas[s]
            t0 = float(o[col_mjd].min())
            rango = max(float(o[col_mjd].max()) - t0, 1.0)
            med_c, esc_c = stats(o[col_flux].values)
            p = o.copy()
            p.loc[idx, col_flux] = p.loc[idx, col_flux] + df * sc
            med_p, esc_p = stats(p[col_flux].values)
            with torch.no_grad():
                xc = reconstruir(vq, tok(render(o, ENC, fr, ar)))
                tp = tok(render(p, ENC, fr, ar))
                xp = reconstruir(vq, tp)
                cfs, ms = {}, {}
                for pr in priors:
                    ms[pr] = mascaras(mgs[pr], tp, dev, g, zona)
                for nom, (pr, forma, dec, ns) in CONFIGS.items():
                    cf = mgs[pr].generate_counterfactual(
                        tp, ms[pr][forma], n_steps=ns, **DECOD[dec])
                    cfs[nom] = reconstruir(vq, cf)
                img = {'ctrl': (xc, med_c, esc_c), 'pert': (xp, med_p, esc_p)}
                img.update({m: (v, med_p, esc_p) for m, v in cfs.items()})
                inv = {k: (invertir_u(v[0], v[1], v[2], o, t0, rango, 0.0),
                           pesos(o, v[0], t0, rango)) for k, v in img.items()}

            pos = o.index.get_indexer(idx)
            ft = o[col_flux].values.astype('float64')
            c0s = np.clip(np.round(col_de((o[col_mjd].values[pos] - t0) / rango)
                                   ).astype(int), 0, PX - 1)
            for nom in CONFIGS:
                for k, i in enumerate(pos):
                    fin = lambda v: None if not np.isfinite(v) else float(v)
                    filas.append({
                        'df': df, 'cfg': nom, 'f_true': float(ft[i]),
                        'f_pert': fin(inv['pert'][0][i]),
                        'w_pert': float(inv['pert'][1][i]),
                        'f_cf': fin(inv[nom][0][i]), 'w_cf': float(inv[nom][1][i]),
                        'snid': s})
                    if df in (100, 300):
                        bc = banda_dominante(cfs[nom], J_R, int(c0s[k]))
                        color.append({'df': df, 'cfg': nom,
                                      'cf': None if bc is None else int(bc)})
        print(f'  df={df} listo ({len(filas)} epocas)', flush=True)
    plt.close(fr)

    salida = {'umbral': U, 'filas': filas, 'color': color, 'n_obj': len(snids)}
    with open(OUT / 'contrafactual_flujo_ablacion.json', 'w') as f:
        json.dump(salida, f)
    analizar(salida)


def analizar(d):
    filas, tabla = d['filas'], {}
    print(f'\n{"=" * 108}\nFLUXCAL, extractor unico {d["umbral"]}, criterio conservador'
          f'   ({d["n_obj"]} objetos)')
    print('  La configuracion "referencia" es la que produjo el 29.2% de df=300.')
    print(f'  {"df":>5} {"configuracion":<22}{"pert":>9}{"conserv":>10}'
          f'{"red conserv":>13}{"red excluir":>13}{"borrado cf":>12}{"n":>7}')
    for df in DFS:
        for nom in CONFIGS:
            sub = [r for r in filas if r['df'] == df and r['cfg'] == nom]
            if not sub:
                continue
            nan = lambda k: np.array([np.nan if r[k] is None else r[k] for r in sub])
            ft, fp, fcf = nan('f_true'), nan('f_pert'), nan('f_cf')
            fp = np.where(nan('w_pert') >= d['umbral'], fp, np.nan)
            fcf = np.where(nan('w_cf') >= d['umbral'], fcf, np.nan)
            base = np.isfinite(fp)
            vis = base & np.isfinite(fcf)
            e = {'pert': rmse((fp - ft)[base]),
                 'conserv': rmse((np.where(np.isfinite(fcf), fcf, fp) - ft)[base]),
                 'excluir': rmse((fcf - ft)[vis]),
                 'pert_vis': rmse((fp - ft)[vis]),
                 'borrado': float(1 - vis.sum() / max(base.sum(), 1)),
                 'n': int(base.sum())}
            e['red_conserv'] = 100 * (1 - e['conserv'] / e['pert'])
            e['red_excluir'] = 100 * (1 - e['excluir'] / e['pert_vis'])
            tabla[f'{df}|{nom}'] = e
            print(f'  {df:5d} {nom:<22}{e["pert"]:9.1f}{e["conserv"]:10.1f}'
                  f'{e["red_conserv"]:12.1f}%{e["red_excluir"]:12.1f}%'
                  f'{100 * e["borrado"]:11.1f}%{e["n"]:7d}', flush=True)

    ref = tabla.get('300|referencia')
    if ref:
        print(f'\n{"=" * 108}\nGANANCIA SOBRE LA REFERENCIA (df=300, '
              f'{ref["red_conserv"]:.1f}% conservador)')
        print(f'  {"configuracion":<22}{"red conserv":>13}{"d puntos":>11}'
              f'{"borrado":>10}{"d borrado":>12}')
        for nom in CONFIGS:
            e = tabla.get(f'300|{nom}')
            if not e:
                continue
            print(f'  {nom:<22}{e["red_conserv"]:12.1f}%'
                  f'{e["red_conserv"] - ref["red_conserv"]:+11.1f}'
                  f'{100 * e["borrado"]:9.1f}%'
                  f'{100 * (e["borrado"] - ref["borrado"]):+11.1f}')

    print(f'\n{"=" * 108}\nCONFUSION DE BANDA en el carril de r')
    print(f'  {"df":>5} {"configuracion":<22}{"% banda mala":>15}{"% sin tinta":>14}')
    conf = {}
    for df in (100, 300):
        for nom in CONFIGS:
            sub = [r for r in d['color'] if r['df'] == df and r['cfg'] == nom]
            if not sub:
                continue
            cf = [r['cf'] for r in sub if r['cf'] is not None]
            e = {'mala': float(np.mean([b != J_R for b in cf])) if cf else np.nan,
                 'sin_tinta': float(np.mean([r['cf'] is None for r in sub]))}
            conf[f'{df}|{nom}'] = e
            print(f'  {df:5d} {nom:<22}{100 * e["mala"]:14.1f}%'
                  f'{100 * e["sin_tinta"]:13.1f}%')

    with open(OUT / 'contrafactual_flujo_ablacion_resumen.json', 'w') as f:
        json.dump({'umbral': d['umbral'], 'tabla': tabla, 'confusion': conf,
                   'configs': {k: list(v) for k, v in CONFIGS.items()}}, f, indent=2)
    print(f'\nResumen en {OUT}/contrafactual_flujo_ablacion_resumen.json', flush=True)


if __name__ == '__main__':
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_flujo_ablacion.json') as f:
            analizar(json.load(f))
    else:
        main()
