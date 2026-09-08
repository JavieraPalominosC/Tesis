"""Sesgo de seleccion por borrado, y diagnostico de si el borrado es real.

CONTEXTO. Al medir la reparacion en flujo, las epocas donde el inversor no
encuentra traza quedaban fuera del promedio. Con tasas de borrado de 30% en
df=300 eso es excluir un tercio de los casos, y no son casos cualquiera: son
plausiblemente los mas dificiles. Todas las cifras de reparacion podrian estar
infladas por seleccion.

PARTE 1: cuanto se mueven las cifras. Se recalcula la reduccion del residuo
dentro de la ventana bajo tres criterios que forman una horquilla.

  excluir       lo que veniamos haciendo. Solo epocas donde el contrafactual
                dejo algo legible. Cota optimista por seleccion.
  conservador   la epoca borrada se imputa con el flujo de la PERTURBADA
                reconstruida, o sea ganancia cero. Es la lectura correcta si uno
                exige que el contrafactual explique: un contrafactual que borra
                la observacion no dice cual seria el valor normal, no la
                repara. Este deberia ser el numero principal.
  optimista     la epoca borrada se imputa con el flujo del CONTROL limpio
                reconstruido, o sea ganancia total. Es la lectura si uno cree
                que borrar equivale a "aqui no habia anomalia". Cota superior.

La verdad esta entre conservador y optimista, y mas cerca del conservador. Se
reportan los tres para que el lector vea el ancho de la horquilla.

PARTE 2: el sesgo es adverso? Se compara la amplitud de la anomalia en las
epocas borradas contra las sobrevivientes. Si las borradas son las de mayor
flujo, la exclusion no es aleatoria y empeora el sesgo.

PARTE 3, la importante: el borrado es artefacto del extractor o comportamiento
real del decodificador? El extractor marca NaN cuando la suma de tinta de la
columna dentro del carril de la banda baja de 0.2. Hay varias formas de que eso
ocurra sin que el decodificador haya borrado nada:

  fuera del carril  el carril de la banda j va de y=j*STEP a y=j*STEP+AMP, y un
                    punto saturado cae JUSTO en el borde superior. Si la
                    reconstruccion difumina un poco, la tinta se va al hueco de
                    0.2 entre carriles y la suma dentro del carril se desploma.
                    Seria un bug nuestro y se arregla ensanchando la ventana.
  corrida en x      la traza esta, pero un par de columnas al lado.
  tinta debil       la traza esta donde corresponde pero difuminada, y la suma
                    queda apenas bajo el umbral. Tambien es del extractor.
  vacio real        no hay tinta en toda la columna. El decodificador no dibujo
                    nada. Esa si es una limitacion del metodo.

Como el renderizador usa 'o-', o sea marcador Y linea continua, en la columna de
una epoca observada siempre deberia haber tinta de la linea que la conecta con
sus vecinas. Que no la haya seria un borrado de verdad.

Uso: python scripts/contrafactual_borrado.py [N_OBJETOS]
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

from scripts.eval_enc_cmp import (
    AMP, N_BANDS, STEP, all_bands, col_band, col_flux, col_mjd, fig_params,
    mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, TARGET_BAND, VQ_CKPT, candidatos
from scripts.contrafactuales import PX, reconstruir
from scripts.contrafactual_comparacion import metodo_base, metodo_azar, metodo_rect
from scripts.contrafactual_flujo import (
    U_MAX, Y_HI, Y_LO, a_np, col_de, fila_a_y, invertir,
)
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
N_OBJ, SEED = 300, 0
UMBRAL_TINTA = 0.2            # el que usa trazas_filas
MARGEN = 8                    # filas extra al ensanchar el carril
J_R = all_bands[TARGET_BAND]


def stats(v):
    v = np.asarray(v, dtype='float64')
    m = np.median(v)
    s = np.median(np.abs(v - m))
    return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)


def carril(j):
    a = max(int(np.floor((Y_HI - (j * STEP + AMP)) / (Y_HI - Y_LO) * PX)), 0)
    b = min(int(np.ceil((Y_HI - j * STEP) / (Y_HI - Y_LO) * PX)), PX)
    return a, b


def diagnostico(x, j, c0):
    """Por que el extractor no vio traza en la columna c0 del carril j."""
    tinta = 1 - a_np(x).mean(axis=2)
    tinta[:5, :] = tinta[-5:, :] = tinta[:, :5] = tinta[:, -5:] = 0.0
    a, b = carril(j)
    ae, be = max(a - MARGEN, 0), min(b + MARGEN, PX)
    col = tinta[:, c0]
    w = float(tinta[a:b, c0].sum())
    w_ext = float(tinta[ae:be, c0].sum())
    lo, hi = max(c0 - 3, 0), min(c0 + 4, PX)
    w_vec = float(tinta[a:b, lo:hi].sum(axis=0).max())
    fila_pico = int(ae + np.argmax(tinta[ae:be, c0]))
    return {
        'w': w, 'w_ext': w_ext, 'w_vecinas': w_vec, 'w_col': float(col.sum()),
        'fila_pico': fila_pico, 'sobre_carril': bool(fila_pico < a),
        'bajo_carril': bool(fila_pico >= b),
    }


def clasificar(d):
    if d['w'] >= UMBRAL_TINTA:
        return 'sin borrado'
    if d['w_col'] < UMBRAL_TINTA:
        return 'vacio real'
    if d['w_ext'] >= 0.5 and (d['sobre_carril'] or d['bajo_carril']):
        return 'fuera del carril'
    if d['w_vecinas'] >= 0.5:
        return 'corrida en x'
    if d['w'] >= 0.02:
        return 'tinta debil'
    return 'vacio en el carril'


def rmse(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def invertir_laxo(x, med, escala, obj, t0, rango, umbral=0.02, buscar=3):
    """Inversor tolerante, para separar el borrado real del artefacto.

    Baja el umbral de tinta de 0.2 a 0.02 y, si aun asi no hay centroide en la
    columna exacta, busca en las columnas vecinas. Si el borrado fuera un
    problema del extractor, con esto tendria que desaparecer casi entero. Lo que
    quede es comportamiento del decodificador.
    """
    img = a_np(x)
    tinta = 1 - img.mean(axis=2)
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
        cols = col_de((obj.loc[m, col_mjd].values - t0) / rango)
        c0 = np.clip(np.round(cols).astype(int), 0, PX - 1)
        filas = cen[c0]
        for k in np.where(~np.isfinite(filas))[0]:
            for paso in range(1, buscar + 1):
                for cc in (c0[k] - paso, c0[k] + paso):
                    if 0 <= cc < PX and np.isfinite(cen[cc]):
                        filas[k] = cen[cc]
                        break
                if np.isfinite(filas[k]):
                    break
        u = np.clip((fila_a_y(filas) - j * STEP) / AMP, 0.0, 1.0)
        rec[m] = med + escala * np.sinh((2 * u - 1) * np.arcsinh(U_MAX))
    return rec


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    curvas = candidatos(n_obj)
    snids = sorted(curvas)
    print(f'objetos: {len(snids)}   carril r: filas {carril(J_R)}', flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    @torch.no_grad()
    def tok(img):
        return vq.encode_to_indices(tf(img).unsqueeze(0).to(dev)).view(1, -1).long().to(dev)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    filas = []          # una por (objeto, df, metodo, epoca inyectada)
    casos = []          # imagenes para inspeccion visual
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
                fc = invertir(xc, med_c, esc_c, o, t0, rango)
                fp = invertir(xp, med_p, esc_p, o, t0, rango)
                cfs = {m: reconstruir(vq, cf) for m, (cf, _) in {
                    'base p95': metodo_base(mg, tp, dev, g, 0.95),
                    'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                    'azar': metodo_azar(mg, tp, dev, g, 148)}.items()}
                fq = {m: invertir(v, med_p, esc_p, o, t0, rango) for m, v in cfs.items()}
                fl = {m: invertir_laxo(v, med_p, esc_p, o, t0, rango)
                      for m, v in cfs.items()}

            pos = o.index.get_indexer(idx)
            ft = o[col_flux].values.astype('float64')
            cols = col_de((o[col_mjd].values[pos] - t0) / rango)
            c0s = np.clip(np.round(cols).astype(int), 0, PX - 1)
            # Amplitud de la anomalia en unidades del MAD global, que es lo que
            # decide la altura del punto dentro del carril.
            amp = np.abs(p[col_flux].values.astype('float64')[pos] - med_p) / esc_p

            for m in METODOS:
                for k, i in enumerate(pos):
                    if not np.isfinite(fp[i]):
                        continue
                    borr = not np.isfinite(fq[m][i])
                    lax = np.isfinite(fl[m][i])
                    r = {'snid': s, 'df': d, 'metodo': m, 'amp': float(amp[k]),
                         'borrado': bool(borr), 'f_true': float(ft[i]),
                         'f_pert': float(fp[i]),
                         'f_cf': None if borr else float(fq[m][i]),
                         'f_laxo': float(fl[m][i]) if lax else None,
                         'f_ctrl': None if not np.isfinite(fc[i]) else float(fc[i])}
                    if borr:
                        dg = diagnostico(cfs[m], J_R, int(c0s[k]))
                        r['diag'] = dg
                        r['clase'] = clasificar(dg)
                        # Se guardan hasta dos ejemplos de cada clase, para que
                        # la inspeccion visual cubra todos los mecanismos y no
                        # solo el mas frecuente.
                        vistos = sum(1 for c in casos if c['clase'] == r['clase'])
                        if d == 300 and m == 'base p95' and vistos < 2:
                            casos.append({
                                'snid': s, 'col': int(c0s[k]), 'amp': float(amp[k]),
                                'clase': r['clase'], 'diag': dg,
                                'limpia': a_np(xc), 'pert': a_np(xp),
                                'cf': a_np(cfs[m]),
                            })
                    filas.append(r)
        print(f'  df={d} listo, {len(filas)} epocas acumuladas', flush=True)
    plt.close(fr)

    with open(OUT / 'contrafactual_borrado.json', 'w') as f:
        json.dump({'filas': filas, 'n_objetos': len(snids)}, f)
    analizar(filas)
    figura_casos(casos)


def analizar(filas):
    print(f'\n{"=" * 96}\nPARTE 3  DIAGNOSTICO DEL BORRADO: artefacto o real', flush=True)
    borradas = [r for r in filas if r['borrado']]
    print(f'  epocas borradas en total: {len(borradas)} de {len(filas)}')
    clases = {}
    for r in borradas:
        clases[r['clase']] = clases.get(r['clase'], 0) + 1
    for k, v in sorted(clases.items(), key=lambda x: -x[1]):
        print(f'    {k:<20} {v:6d}   {100 * v / max(len(borradas), 1):5.1f}%')
    if borradas:
        for k in ('w', 'w_ext', 'w_vecinas', 'w_col'):
            v = np.array([r['diag'][k] for r in borradas])
            print(f'    {k:<10} mediana {np.median(v):7.3f}   p90 {np.percentile(v, 90):7.3f}'
                  f'   (umbral del extractor {UMBRAL_TINTA})')
        sob = np.mean([r['diag']['sobre_carril'] for r in borradas])
        print(f'    pico de tinta POR ENCIMA del borde superior del carril: '
              f'{100 * sob:.1f}%')
        rec = np.mean([r['f_laxo'] is not None for r in borradas])
        print(f'\n  Extractor laxo (umbral 0.02 en vez de 0.2, y busqueda en +-3 '
              f'columnas):\n    recupera el {100 * rec:.1f}% de las epocas borradas. '
              f'El resto es del decodificador.')
        print(f'  {"df":>5} {"metodo":<13}{"borrado estricto":>18}{"borrado laxo":>14}'
              f'{"recuperado":>12}')
        for d in DFS:
            for m in METODOS:
                sub = [r for r in filas if r['df'] == d and r['metodo'] == m]
                if not sub:
                    continue
                be = np.mean([r['borrado'] for r in sub])
                bl = np.mean([r['f_laxo'] is None for r in sub])
                print(f'  {d:5d} {m:<13}{100 * be:17.1f}%{100 * bl:13.1f}%'
                      f'{100 * (1 - bl / be) if be > 0 else 0:11.1f}%')

    print(f'\n{"=" * 96}\nPARTE 2  EL SESGO ES ADVERSO? amplitud en unidades de MAD global')
    print(f'  {"df":>5} {"metodo":<13}{"borradas":>10}{"sobrevive":>11}'
          f'{"razon":>8}{"n borr":>8}')
    sesgo = {}
    for d in DFS:
        for m in METODOS:
            sub = [r for r in filas if r['df'] == d and r['metodo'] == m]
            b = np.array([r['amp'] for r in sub if r['borrado']])
            n = np.array([r['amp'] for r in sub if not r['borrado']])
            if len(b) < 5 or len(n) < 5:
                continue
            sesgo[f'{d}|{m}'] = {'borradas': float(np.median(b)),
                                 'sobreviven': float(np.median(n)),
                                 'razon': float(np.median(b) / np.median(n)),
                                 'n': int(len(b))}
            z = sesgo[f'{d}|{m}']
            print(f'  {d:5d} {m:<13}{z["borradas"]:10.1f}{z["sobreviven"]:11.1f}'
                  f'{z["razon"]:8.2f}{z["n"]:8d}')

    print(f'\n{"=" * 96}\nPARTE 1  REDUCCION DEL RESIDUO EN LA VENTANA, tres criterios')
    print(f'  {"df":>5} {"metodo":<13}{"pert":>9}' +
          ''.join(f'{c:>11}' for c in ('excluir', 'conserv', 'optimista')) +
          f'{"reduccion: excl / conserv / optim / laxo":>43}{"borrado":>9}')
    tabla = {}
    for d in DFS:
        for m in METODOS:
            sub = [r for r in filas if r['df'] == d and r['metodo'] == m]
            if not sub:
                continue
            nan = lambda k: np.array([np.nan if r.get(k) is None else r[k]
                                      for r in sub])
            ft, fp, fcf, fct = (nan('f_true'), nan('f_pert'), nan('f_cf'),
                                nan('f_ctrl'))
            flx = nan('f_laxo')
            vis = np.isfinite(fcf)
            res_p = fp - ft
            # Conservador: la borrada se imputa con la perturbada, ganancia cero.
            # Optimista: se imputa con el control limpio reconstruido; si el
            # control tampoco es legible ahi, se cae al conservador.
            cons = np.where(vis, fcf, fp) - ft
            opt = np.where(vis, fcf, np.where(np.isfinite(fct), fct, fp)) - ft
            # Con extractor laxo, tambien conservador para lo que ni asi aparece.
            lax = np.where(np.isfinite(flx), flx, fp) - ft
            e = {'pert': rmse(res_p[vis]), 'excluir': rmse(cons[vis]),
                 'pert_todas': rmse(res_p), 'conserv': rmse(cons),
                 'optimista': rmse(opt), 'laxo': rmse(lax),
                 'borrado': float(1 - vis.mean()),
                 'borrado_laxo': float(np.mean(~np.isfinite(flx))), 'n': len(sub)}
            e['red_excluir'] = 100 * (1 - e['excluir'] / e['pert'])
            e['red_conserv'] = 100 * (1 - e['conserv'] / e['pert_todas'])
            e['red_optimista'] = 100 * (1 - e['optimista'] / e['pert_todas'])
            e['red_laxo'] = 100 * (1 - e['laxo'] / e['pert_todas'])
            tabla[f'{d}|{m}'] = e
            print(f'  {d:5d} {m:<13}{e["pert_todas"]:9.1f}{e["excluir"]:11.1f}'
                  f'{e["conserv"]:11.1f}{e["optimista"]:11.1f}'
                  f'{e["red_excluir"]:14.1f}%{e["red_conserv"]:9.1f}%'
                  f'{e["red_optimista"]:9.1f}%{e["red_laxo"]:9.1f}%'
                  f'{100 * e["borrado"]:8.1f}%', flush=True)

    with open(OUT / 'contrafactual_borrado_resumen.json', 'w') as f:
        json.dump({'clases': clases, 'sesgo': sesgo, 'reduccion': tabla,
                   'n_borradas': len(borradas), 'n_epocas': len(filas)}, f, indent=2)


def figura_casos(casos):
    if not casos:
        print('sin casos para inspeccion visual')
        return
    a, b = carril(J_R)
    ae, be = max(a - MARGEN, 0), min(b + MARGEN, PX)
    casos = sorted(casos, key=lambda c: c['clase'])
    n = min(len(casos), 6)
    fig, ax = plt.subplots(4, n, figsize=(3.6 * n, 11))
    for k in range(n):
        c = casos[k]
        c0 = c['col']
        lo, hi = max(c0 - 26, 0), min(c0 + 26, PX)
        for i, (nom, img) in enumerate((('limpia', c['limpia']),
                                        ('perturbada', c['pert']),
                                        ('contrafactual', c['cf']))):
            z = ax[i][k]
            z.imshow(img[ae:be, lo:hi], interpolation='nearest',
                     extent=[lo, hi, be, ae], aspect='auto')
            z.axhline(a, color='#0000ff', lw=1.1)
            z.axhline(b, color='#0000ff', lw=1.1)
            z.axvline(c0, color='#ff0000', lw=1.1, ls='--')
            z.set_ylabel(nom, fontsize=9)
            if i == 0:
                z.set_title(f'SNID {c["snid"]}\n{c["clase"]},  '
                            f'amplitud {c["amp"]:.0f} MAD', fontsize=8.5)
        z = ax[3][k]
        tin = 1 - c['cf'].mean(axis=2)
        z.plot(tin[ae:be, c0], np.arange(ae, be), '-', color='#333333',
               label='columna exacta')
        z.plot(tin[ae:be, max(c0 - 3, 0):c0 + 4].max(axis=1), np.arange(ae, be), ':',
               color='#e08214', label='max en +-3 columnas')
        z.axhline(a, color='#0000ff', lw=1.1)
        z.axhline(b, color='#0000ff', lw=1.1)
        z.invert_yaxis()
        z.set_xlabel('tinta por fila', fontsize=8)
        z.set_title(f'suma en carril {c["diag"]["w"]:.3f}  '
                    f'ensanchado {c["diag"]["w_ext"]:.3f}\n'
                    f'columna entera {c["diag"]["w_col"]:.3f}   '
                    f'(umbral {UMBRAL_TINTA})', fontsize=8)
        if k == 0:
            z.legend(frameon=False, fontsize=7)
    fig.suptitle('Casos con borrado reportado, df=300, base p95. Lineas azules: '
                 'borde del carril de la banda r que usa el extractor. '
                 'Linea roja: columna de la epoca.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / 'contrafactual_borrado.png', dpi=140)
    print(f'\nFiguras y json en {OUT}/', flush=True)


if __name__ == '__main__':
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_borrado.json') as f:
            analizar(json.load(f)['filas'])
    else:
        main()
