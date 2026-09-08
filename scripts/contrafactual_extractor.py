"""Extractor calibrado y recalculo de las cifras de flujo con un unico criterio.

PROBLEMA QUE CIERRA. El umbral de tinta del inversor, 0.2, nunca se calibro
contra nada. Resulto costar la mitad del efecto medido: la mitad de las epocas
que reportabamos como borradas por el modelo eran en realidad trazas debiles que
el extractor no queria ver. Peor, en el diagnostico anterior el extractor laxo
se aplico solo al contrafactual y el estricto a la perturbada, una asimetria que
favorece a un lado de la comparacion.

CALIBRACION. El umbral se fija como el mas alto que no borra ninguna epoca sobre
reconstrucciones VQ-VAE de curvas LIMPIAS. El criterio se ancla en un caso donde
la respuesta correcta se conoce: el punto esta ahi, asi que todo borrado es por
definicion fallo del extractor y no del modelo.

  Sobre reconstrucciones, no sobre renders crudos, porque el borroneo que el
  extractor debe tolerar es el que mete el decodificador.

  Sobre un conjunto de calibracion DISJUNTO de los 300 objetos de evaluacion.
  candidatos() es determinista y corta al llegar al cupo, asi que pedir 400 y
  quedarse con los ultimos 100 da objetos nuevos con identicos criterios.

  Bajar el umbral no es gratis: en algun punto empieza a tomar tinta que se
  filtra de los carriles vecinos y el centroide se ensucia. Por eso se barre el
  umbral reportando las dos cosas a la vez, tasa de borrado y error de ida y
  vuelta sobre la curva limpia, que es el piso del analisis. Si el umbral de
  cero borrado degrada el piso, el criterio no sirve tal cual y hay que apuntar
  a 1% en vez de cero.

Luego se recalculan las cifras de reduccion del residuo aplicando ese unico
umbral de forma identica a la limpia, la perturbada y el contrafactual.

EXTRA. Se cuantifica con que frecuencia el color dominante en el carril de la
banda r no corresponde a r, en las epocas reescritas. Confundir bandas es peor
que difuminar, y puede explicar parte del borrado. Se mide tambien sobre la
reconstruccion limpia, como linea base.

Uso: python scripts/contrafactual_extractor.py [N_EVAL] [N_CAL]
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.eval_enc_cmp import (
    AMP, STEP, all_bands, col_band, col_flux, col_mjd, fig_params, mejor_ckpt,
    render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, TARGET_BAND, VQ_CKPT, candidatos
from scripts.contrafactuales import PX, reconstruir
from scripts.contrafactual_comparacion import metodo_base, metodo_azar, metodo_rect
from scripts.contrafactual_flujo import U_MAX, Y_HI, Y_LO, a_np, col_de, fila_a_y
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
N_EVAL, N_CAL, SEED = 300, 100, 0
UMBRALES = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
U_VIEJO = 0.2
J_R = all_bands[TARGET_BAND]
RGB = {j: np.array(mcolors.to_rgb(fig_params['colors'][j]))
       for j in all_bands.values()}


def stats(v):
    v = np.asarray(v, dtype='float64')
    m = np.median(v)
    s = np.median(np.abs(v - m))
    return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)


def carril(j):
    a = max(int(np.floor((Y_HI - (j * STEP + AMP)) / (Y_HI - Y_LO) * PX)), 0)
    b = min(int(np.ceil((Y_HI - j * STEP) / (Y_HI - Y_LO) * PX)), PX)
    return a, b


def columnas(obj, m, t0, rango):
    c = col_de((obj.loc[m, col_mjd].values - t0) / rango)
    return np.clip(np.round(c).astype(int), 0, PX - 1)


def tinta_de(x):
    img = a_np(x)
    t = 1 - img.mean(axis=2)
    t[:5, :] = t[-5:, :] = t[:, :5] = t[:, -5:] = 0.0
    return img, t


def pesos(obj, x, t0, rango):
    """Suma de tinta en el carril, por epoca observada. El insumo del umbral."""
    _, tinta = tinta_de(x)
    w = np.full(len(obj), np.nan)
    for band_key, j in all_bands.items():
        m = (obj[col_band] == band_key).values
        if not m.any():
            continue
        a, b = carril(j)
        w[m] = tinta[a:b, :].sum(axis=0)[columnas(obj, m, t0, rango)]
    return w


def invertir_u(x, med, escala, obj, t0, rango, umbral):
    """Inversor con umbral parametrico. Identico para las tres imagenes."""
    _, tinta = tinta_de(x)
    rec = np.full(len(obj), np.nan)
    for band_key, j in all_bands.items():
        m = (obj[col_band] == band_key).values
        if not m.any():
            continue
        a, b = carril(j)
        sub = tinta[a:b, :]
        w = sub.sum(axis=0)
        cen = (sub * np.arange(a, b, dtype=np.float64)[:, None]).sum(axis=0) / \
            np.maximum(w, 1e-9)
        cen[w < umbral] = np.nan
        u = np.clip((fila_a_y(cen[columnas(obj, m, t0, rango)]) - j * STEP) / AMP,
                    0.0, 1.0)
        rec[m] = med + escala * np.sinh((2 * u - 1) * np.arcsinh(U_MAX))
    return rec


def banda_dominante(x, j, c0):
    """Banda cuyo color esta mas cerca del color medio de la tinta del carril."""
    img, tinta = tinta_de(x)
    a, b = carril(j)
    w = tinta[a:b, c0]
    if w.sum() < 1e-3:
        return None
    rgb = (img[a:b, c0, :] * w[:, None]).sum(axis=0) / w.sum()
    return min(RGB, key=lambda k: np.sum((RGB[k] - rgb) ** 2))


def rmse(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def calibrar(vq, dev, fr, ar, n_eval, n_cal):
    """Barrido de umbral sobre reconstrucciones limpias de objetos no evaluados."""
    todos = candidatos(n_eval + n_cal)
    snids = sorted(todos)
    # candidatos() es determinista, asi que los primeros n_eval son exactamente
    # el conjunto de evaluacion. La calibracion usa solo los siguientes.
    eval_ids = sorted(candidatos(n_eval))
    cal_ids = [s for s in snids if s not in set(eval_ids)][:n_cal]
    print(f'evaluacion {len(eval_ids)} objetos, calibracion {len(cal_ids)}, '
          f'interseccion {len(set(eval_ids) & set(cal_ids))}', flush=True)

    ws, recs = [], []
    for s in cal_ids:
        o = todos[s][0]
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        med, esc = stats(o[col_flux].values)
        with torch.no_grad():
            x = reconstruir(vq, vq.encode_to_indices(
                tf(render(o, ENC, fr, ar)).unsqueeze(0).to(dev)).view(1, -1).long().to(dev))
        ws.append(pesos(o, x, t0, rango))
        recs.append((x, med, esc, o, t0, rango))
    w = np.concatenate(ws)
    w = w[np.isfinite(w)]

    print(f'\nCALIBRACION sobre {len(w)} epocas de curvas limpias reconstruidas')
    print(f'  tinta en el carril: min {w.min():.5f}   p0.1 {np.percentile(w, 0.1):.5f}'
          f'   p1 {np.percentile(w, 1):.5f}   mediana {np.median(w):.3f}')
    print(f'  {"umbral":>8}{"% borrado":>12}{"piso RMSE":>12}   (piso = error de ida y '
          f'vuelta sobre la curva limpia)')
    barrido = {}
    for u in UMBRALES:
        err = np.concatenate([invertir_u(*r, u) - r[3][col_flux].values.astype('float64')
                              for r in recs])
        b = float((w < u).mean())
        barrido[str(u)] = {'borrado': b, 'piso': rmse(err)}
        print(f'  {u:8.3f}{100 * b:11.2f}%{rmse(err):12.2f}', flush=True)

    # El umbral exacto de cero borrado es el minimo observado: cualquier valor
    # por debajo tambien da cero, y el criterio pide el mas alto.
    elegido = float(np.floor(w.min() * 100) / 100)
    motivo = 'cero borrado alcanzable'
    if elegido <= 0:
        # Si alguna epoca limpia ya sale sin tinta, cero no es alcanzable y hay
        # que relajar el criterio a 1%.
        elegido = float(np.floor(np.percentile(w, 1) * 100) / 100)
        motivo = 'cero NO alcanzable, se apunta a 1%'
    err = np.concatenate([invertir_u(*r, elegido) - r[3][col_flux].values.astype('float64')
                          for r in recs])
    piso_el = rmse(err)
    print(f'\n  umbral elegido: {elegido}   ({motivo})')
    print(f'  piso con {U_VIEJO}: {barrido[str(U_VIEJO)]["piso"]:.2f}   '
          f'piso con {elegido}: {piso_el:.2f} FLUXCAL')
    print(f'  el umbral viejo {U_VIEJO} era {elegido / U_VIEJO:.1f} veces MAS PERMISIVO '
          f'que el minimo valido, no mas estricto')
    return elegido, barrido, {'min': float(w.min()), 'p01': float(np.percentile(w, 0.1)),
                              'p1': float(np.percentile(w, 1)),
                              'mediana': float(np.median(w)), 'n': int(len(w)),
                              'piso_elegido': piso_el, 'motivo': motivo,
                              'n_cal': len(cal_ids)}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else N_EVAL
    n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else N_CAL
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    U, barrido, cal = calibrar(vq, dev, fr, ar, n_eval, n_cal)
    curvas = candidatos(n_eval)
    snids = sorted(curvas)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def tok(img):
        return vq.encode_to_indices(tf(img).unsqueeze(0).to(dev)).view(1, -1).long().to(dev)

    filas, color = [], []
    for d in DFS:
        for s in snids:
            o, idx, sc = curvas[s]
            t0 = float(o[col_mjd].min())
            rango = max(float(o[col_mjd].max()) - t0, 1.0)
            med_c, esc_c = stats(o[col_flux].values)
            p = o.copy()
            p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
            med_p, esc_p = stats(p[col_flux].values)
            with torch.no_grad():
                xc = reconstruir(vq, tok(render(o, ENC, fr, ar)))
                tp = tok(render(p, ENC, fr, ar))
                xp = reconstruir(vq, tp)
                cfs = {m: reconstruir(vq, cf) for m, (cf, _) in {
                    'base p95': metodo_base(mg, tp, dev, g, 0.95),
                    'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                    'azar': metodo_azar(mg, tp, dev, g, 148)}.items()}
                # Se invierte SIN umbral y se guarda aparte la tinta de cada
                # epoca. Asi el umbral se aplica despues, identico para las tres
                # imagenes por construccion, y ademas queda registrada la
                # distribucion de tinta, que es la medicion libre de umbral y la
                # que de verdad contesta si el borrado es real.
                img = {'ctrl': (xc, med_c, esc_c), 'pert': (xp, med_p, esc_p)}
                img.update({m: (v, med_p, esc_p) for m, v in cfs.items()})
                inv = {k: (invertir_u(v[0], v[1], v[2], o, t0, rango, 0.0),
                           pesos(o, v[0], t0, rango)) for k, v in img.items()}

            pos = o.index.get_indexer(idx)
            ft = o[col_flux].values.astype('float64')
            # Columnas en el orden de pos, no en el del dataframe.
            c0s = np.clip(np.round(col_de((o[col_mjd].values[pos] - t0) / rango)
                                   ).astype(int), 0, PX - 1)
            for m in METODOS:
                for k, i in enumerate(pos):
                    fin = lambda v: None if not np.isfinite(v) else float(v)
                    filas.append({
                        'df': d, 'metodo': m, 'f_true': float(ft[i]),
                        'f_pert': fin(inv['pert'][0][i]),
                        'w_pert': float(inv['pert'][1][i]),
                        'f_cf': fin(inv[m][0][i]), 'w_cf': float(inv[m][1][i]),
                        'f_ctrl': fin(inv['ctrl'][0][i]),
                        'w_ctrl': float(inv['ctrl'][1][i])})
                    if d in (100, 300):
                        bc = banda_dominante(cfs[m], J_R, int(c0s[k]))
                        bl = banda_dominante(xc, J_R, int(c0s[k]))
                        color.append({'df': d, 'metodo': m,
                                      'cf': None if bc is None else int(bc),
                                      'limpia': None if bl is None else int(bl)})
        print(f'  df={d} listo ({len(filas)} epocas)', flush=True)
    plt.close(fr)

    salida = {'umbral': U, 'barrido': barrido, 'calibracion': cal,
              'filas': filas, 'color': color}
    with open(OUT / 'contrafactual_extractor.json', 'w') as f:
        json.dump(salida, f)
    analizar(salida)


def analizar(d):
    U, filas = d['umbral'], d['filas']
    cal = d['calibracion']
    tabla, rank = {}, {}

    print(f'\n{"=" * 100}\nTINTA POR EPOCA, medicion libre de umbral')
    print(f'  referencia en curvas LIMPIAS reconstruidas: min {cal["min"]:.3f}   '
          f'p0.1 {cal["p01"]:.3f}   p1 {cal["p1"]:.3f}   mediana {cal["mediana"]:.3f}')
    print(f'  {"df":>5} {"metodo":<13}{"mediana cf":>12}{"p25":>8}'
          f'{"% bajo el min limpio":>22}{"% bajo 0.2":>12}{"pert mediana":>14}')
    tinta = {}
    for df in DFS:
        for m in METODOS:
            sub = [r for r in filas if r['df'] == df and r['metodo'] == m]
            if not sub:
                continue
            w = np.array([r['w_cf'] for r in sub])
            wp = np.array([r['w_pert'] for r in sub])
            e = {'mediana': float(np.median(w)), 'p25': float(np.percentile(w, 25)),
                 'bajo_min': float((w < cal['min']).mean()),
                 'bajo_02': float((w < U_VIEJO).mean()),
                 'pert_mediana': float(np.median(wp))}
            tinta[f'{df}|{m}'] = e
            print(f'  {df:5d} {m:<13}{e["mediana"]:12.3f}{e["p25"]:8.3f}'
                  f'{100 * e["bajo_min"]:21.1f}%{100 * e["bajo_02"]:11.1f}%'
                  f'{e["pert_mediana"]:14.3f}')
    for u in (U, U_VIEJO):
        etiq = 'calibrado' if u == U else 'viejo, robustez'
        print(f'\n{"=" * 100}\nTABLA con extractor unico, umbral {u} ({etiq}), '
              f'criterio conservador')
        print(f'  {"df":>5} {"metodo":<13}{"pert":>9}{"conserv":>10}{"excluir":>10}'
              f'{"red conserv":>13}{"red excluir":>13}{"borrado cf":>12}'
              f'{"borrado pert":>14}')
        for df in DFS:
            for m in METODOS:
                sub = [r for r in filas if r['df'] == df and r['metodo'] == m]
                if not sub:
                    continue
                nan = lambda k: np.array([np.nan if r[k] is None else r[k]
                                          for r in sub])
                ft, fp, fcf = nan('f_true'), nan('f_pert'), nan('f_cf')
                # El umbral se aplica aqui, igual para las tres imagenes.
                fp = np.where(nan('w_pert') >= u, fp, np.nan)
                fcf = np.where(nan('w_cf') >= u, fcf, np.nan)
                # La epoca entra si la perturbada es legible: sin eso no hay
                # linea base contra la cual medir la reparacion.
                base = np.isfinite(fp)
                vis = base & np.isfinite(fcf)
                res_p = (fp - ft)[base]
                cons = (np.where(np.isfinite(fcf), fcf, fp) - ft)[base]
                e = {'pert': rmse(res_p), 'conserv': rmse(cons),
                     'excluir': rmse((fcf - ft)[vis]),
                     'pert_vis': rmse((fp - ft)[vis]),
                     'borrado': float(1 - vis.sum() / max(base.sum(), 1)),
                     'borrado_pert': float(1 - base.mean()), 'n': int(base.sum())}
                e['red_conserv'] = 100 * (1 - e['conserv'] / e['pert'])
                e['red_excluir'] = 100 * (1 - e['excluir'] / e['pert_vis'])
                tabla[f'{u}|{df}|{m}'] = e
                print(f'  {df:5d} {m:<13}{e["pert"]:9.1f}{e["conserv"]:10.1f}'
                      f'{e["excluir"]:10.1f}{e["red_conserv"]:12.1f}%'
                      f'{e["red_excluir"]:12.1f}%{100 * e["borrado"]:11.1f}%'
                      f'{100 * e["borrado_pert"]:13.1f}%', flush=True)

        print(f'\n  Ranking con umbral {u}: reduccion conservadora')
        print(f'  {"df":>5}{"base p95":>12}{"region":>12}{"diferencia":>13}{"gana":>10}')
        for df in DFS:
            a = tabla[f'{u}|{df}|base p95']['red_conserv']
            b = tabla[f'{u}|{df}|rect 1 pasa']['red_conserv']
            rank[f'{u}|{df}'] = {'base': a, 'rect': b,
                                 'gana': 'rect' if b > a else 'base'}
            print(f'  {df:5d}{a:11.1f}%{b:11.1f}%{b - a:+12.1f}'
                  f'{rank[f"{u}|{df}"]["gana"]:>10}')

    print(f'\n{"=" * 100}\nCONFUSION DE BANDA en el carril de r, epocas reescritas')
    print(f'  {"df":>5} {"metodo":<13}{"contrafactual":>16}{"limpia (base)":>16}'
          f'{"sin tinta cf":>14}')
    conf = {}
    for df in (100, 300):
        for m in METODOS:
            sub = [r for r in d['color'] if r['df'] == df and r['metodo'] == m]
            if not sub:
                continue
            cf = [r['cf'] for r in sub if r['cf'] is not None]
            li = [r['limpia'] for r in sub if r['limpia'] is not None]
            e = {'cf': float(np.mean([b != J_R for b in cf])) if cf else np.nan,
                 'limpia': float(np.mean([b != J_R for b in li])) if li else np.nan,
                 'sin_tinta': float(np.mean([r['cf'] is None for r in sub])),
                 'n': len(sub)}
            conf[f'{df}|{m}'] = e
            print(f'  {df:5d} {m:<13}{100 * e["cf"]:15.1f}%{100 * e["limpia"]:15.1f}%'
                  f'{100 * e["sin_tinta"]:13.1f}%', flush=True)

    with open(OUT / 'contrafactual_extractor_resumen.json', 'w') as f:
        json.dump({'umbral': U, 'umbral_viejo': U_VIEJO, 'barrido': d['barrido'],
                   'calibracion': d['calibracion'], 'tinta': tinta, 'tabla': tabla,
                   'ranking': rank, 'confusion': conf}, f, indent=2)
    print(f'\nResumen en {OUT}/contrafactual_extractor_resumen.json', flush=True)


if __name__ == '__main__':
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_extractor.json') as f:
            analizar(json.load(f))
    else:
        main()
