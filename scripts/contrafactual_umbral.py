"""Umbral de tamano minimo, afinado con varios cientos de objetos.

Con 40 SNIa la fraccion de exito tenia un error estandar de ~8 puntos y el cruce
del 50% quedaba borroso entre 30 y 45 dias. Esa cifra se va a citar, asi que
tiene que venir con incertidumbre declarada.

QUE CAMBIA RESPECTO DE contrafactual_flujo_cierre.py

  300 objetos    candidatos() baraja con semilla fija y corta al llegar al cupo,
                 asi que pedir 300 devuelve un superconjunto exacto de los 40 de
                 antes, con los mismos criterios (fold de validacion, MAD_r
                 finito y positivo, minimo de puntos en la banda r). El error
                 estandar de una fraccion cerca de 0.5 baja de 7.9 a 2.9 puntos.
                 Subir a 400 solo lo llevaria a 2.5 porque cae como 1/sqrt(n), y
                 no compensa el 33% de computo extra.

  grilla densa   13 duraciones con paso de 5 dias entre 20 y 50, que es donde
                 esta el cruce. No tiene sentido gastar puntos en 90 dias, donde
                 ya sabemos que todos los metodos funcionan.

  en puntos      La causa real del umbral es cuantos puntos fotometricos caen en
                 la ventana, no cuantos dias dura. Se agrupan todas las
                 combinaciones (objeto, dt) por numero de puntos perturbados,
                 deduplicando las que tocan exactamente las mismas epocas, y se
                 mide la misma curva contra el conteo. Se comparan las dos
                 versiones por devianza para decidir cual predice mejor.

  cruce medido   El 50% se obtiene de un ajuste logistico, no de mirar entre que
                 dos puntos de la grilla cae. El ajuste da ademas el ancho de la
                 transicion (de 25% a 75%), que es lo que distingue un umbral
                 nitido de una transicion gradual. La incertidumbre sale de
                 remuestrear OBJETOS, no filas, porque cada objeto aporta todas
                 las duraciones y esas estan correlacionadas.

Uso: python scripts/contrafactual_umbral.py [N_OBJETOS]
"""
import sys
sys.path.insert(0, '.')
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.eval_enc_cmp import (
    col_band, col_flux, col_mjd, fig_params, mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import reconstruir
from scripts.contrafactual_comparacion import metodo_base, metodo_azar, metodo_rect
from scripts.contrafactual_flujo import U_MAX, invertir
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
DTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90]
DFS_DT = [100, 300]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
N_OBJ, SEED, N_BOOT = 300, 0, 2000


def stats(v):
    v = np.asarray(v, dtype='float64')
    m = np.median(v)
    s = np.median(np.abs(v - m))
    return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)


def rmse(a, b, m):
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    with open(OUT / 'contrafactual_flujo.json') as f:
        piso = json.load(f)['piso_inversor_rmse']
    print(f'piso del inversor: {piso:.2f} FLUXCAL', flush=True)

    curvas = candidatos(n_obj)
    snids = sorted(curvas)
    print(f'objetos: {len(snids)}   error estandar de una fraccion en 0.5: '
          f'{50 / np.sqrt(len(snids)):.1f} puntos', flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    @torch.no_grad()
    def tokens_de(img):
        x = tf(img).unsqueeze(0).to(dev)
        return vq.encode_to_indices(x).view(1, -1).long().to(dev)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    def evaluar(o, idx_pert, idx_cola, d, sc):
        """Residuos en flujo, con mascara PROPIA de cada metodo.

        El inversor devuelve NaN en una epoca cuando no encuentra tinta en el
        carril de esa banda, o sea cuando la reconstruccion borro el punto. Si se
        usara la interseccion de los tres metodos, el borrado de uno anularia la
        epoca para todos, y con ventanas de 1 a 2 puntos eso descarta la fila
        entera. Peor todavia, esas filas terminaban contadas como fracaso.

        Aca cada metodo se compara contra la perturbada sobre exactamente las
        epocas donde ambos son medibles, y el borrado se cuenta aparte como el
        modo de falla que es.
        """
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        p = o.copy()
        p.loc[idx_pert, col_flux] = p.loc[idx_pert, col_flux] + d * sc
        med_p, esc_p = stats(p[col_flux].values)
        tp = tokens_de(render(p, ENC, fr, ar))
        ftrue = o[col_flux].values.astype('float64')
        vent = np.zeros(len(o), bool)
        vent[o.index.get_indexer(idx_pert)] = True
        cola = np.zeros(len(o), bool)
        cola[o.index.get_indexer(idx_cola)] = True

        with torch.no_grad():
            fp = invertir(reconstruir(vq, tp), med_p, esc_p, o, t0, rango)
            out = {}
            for m, (cf, _) in {
                    'base p95': metodo_base(mg, tp, dev, g, 0.95),
                    'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                    'azar': metodo_azar(mg, tp, dev, g, 148)}.items():
                out[m] = invertir(reconstruir(vq, cf), med_p, esc_p, o, t0, rango)

        ok_p = np.isfinite(fp)
        res = {}
        for m, v in out.items():
            ok = ok_p & np.isfinite(v)
            sv, sr, sc_ = ok & vent, ok & ~vent, ok & cola
            res[m] = {
                'vent': rmse(v, ftrue, sv), 'pert_vent': rmse(fp, ftrue, sv),
                'resto': rmse(v, ftrue, sr), 'pert_resto': rmse(fp, ftrue, sr),
                'cola': rmse(v, ftrue, sc_), 'pert_cola': rmse(fp, ftrue, sc_),
                'n_vent': int(sv.sum()), 'n_resto': int(sr.sum()),
                'borrados': int((vent & ok_p & ~np.isfinite(v)).sum()),
                'n_vent_pos': int((vent & ok_p).sum()),
            }
        return res

    salida = {'piso_inversor': piso, 'n_objetos': len(snids), 'dts': DTS, 'dfs': DFS}

    # ------------------------------------------------------------ saturacion
    print(f'\nSATURACION (recorte en {U_MAX:.0f} MAD del objeto)', flush=True)
    sat = {}
    for d in DFS:
        rel, rec = [], []
        for s in snids:
            o, idx_cola, sc = curvas[s]
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            med, esc = stats(p[col_flux].values)
            r = np.abs(p.loc[idx_cola, col_flux].values.astype('float64') - med) / esc
            rel.append(np.median(r))
            rec.append(float(np.mean(r > U_MAX)))
        sat[str(d)] = {'rel_mediano': float(np.median(rel)),
                       'rel_p90': float(np.percentile(rel, 90)),
                       'frac_recortada': float(np.mean(rec))}
        print(f'  df={d:<4} rel={sat[str(d)]["rel_mediano"]:6.1f} MAD   '
              f'recorte={100 * sat[str(d)]["frac_recortada"]:5.1f}%', flush=True)
    salida['saturacion'] = sat

    # --------------------------------------------------------- amplificacion
    print(f'\nAMPLIFICACION con {len(snids)} objetos', flush=True)
    amp = {}
    for d in DFS:
        peor = {m: [] for m in METODOS}
        borr = {m: [] for m in METODOS}
        for s in snids:
            o, idx_cola, sc = curvas[s]
            r = evaluar(o, idx_cola, idx_cola, d, sc)
            for m in METODOS:
                q = r[m]
                peor[m].append(None if not np.isfinite(q['cola'])
                               else bool(q['cola'] > q['pert_cola']))
                borr[m].append(q['borrados'] / max(q['n_vent_pos'], 1))
        amp[str(d)] = {m: {'peor': peor[m], 'borrado': borr[m]} for m in METODOS}
        print(f'  df={d:<4}' + ''.join(
            f'{m} {100 * np.mean([x for x in peor[m] if x is not None]):5.1f}% '
            f'(borra {100 * np.mean(borr[m]):4.1f}%)   ' for m in METODOS), flush=True)
    salida['amplificacion_cruda'] = amp

    # ------------------------------------------------------- barrido dt
    ventanas = {}
    for s in snids:
        o, idx_cola, _ = curvas[s]
        sub = o.loc[idx_cola]
        t_fin = float(sub[col_mjd].max())
        for dt in DTS:
            ventanas[(s, dt)] = list(sub.index[sub[col_mjd] >= t_fin - dt])

    crudo = {}
    for d in DFS_DT:
        print(f'\nBARRIDO dt   df={d}', flush=True)
        fila = {'snid': [], 'dt': [], 'n': [], 'clave': []}
        for m in METODOS:
            fila[m] = {'vent': [], 'pert_vent': [], 'resto': [], 'pert_resto': [],
                       'neto': [], 'borrados': [], 'n_vent_pos': []}
        for dt in DTS:
            for s in snids:
                o, idx_cola, sc = curvas[s]
                iv = ventanas[(s, dt)]
                r = evaluar(o, iv, idx_cola, d, sc)
                fila['snid'].append(s)
                fila['dt'].append(dt)
                fila['n'].append(len(iv))
                fila['clave'].append(f'{s}|{min(iv)}|{len(iv)}')
                for m in METODOS:
                    q = r[m]
                    gan = (q['pert_vent'] ** 2 - q['vent'] ** 2) * q['n_vent']
                    per = (q['resto'] ** 2 - q['pert_resto'] ** 2) * q['n_resto']
                    for k in ('vent', 'pert_vent', 'resto', 'pert_resto',
                              'borrados', 'n_vent_pos'):
                        fila[m][k].append(q[k])
                    fila[m]['neto'].append(gan - per)
            j = DTS.index(dt) * len(snids)
            linea = f'  dt={dt:<3} n_pts={np.mean(fila["n"][j:]):.2f}  '
            for m in METODOS:
                v = np.array(fila[m]['neto'][j:], float)
                v = v[np.isfinite(v)]
                linea += f'{m} {100 * (v > 0).mean():5.1f}% (n={len(v)})  '
            print(linea, flush=True)
        crudo[str(d)] = fila
    salida['barrido'] = crudo
    plt.close(fr)

    with open(OUT / 'contrafactual_umbral.json', 'w') as f:
        json.dump(salida, f)
    print(f'\nCrudo guardado. Analizando...', flush=True)
    analizar(salida)


# --------------------------------------------------------------------- ajuste
def logistica(x, y, ridge=1e-4):
    """Regresion logistica de dos parametros por IRLS, con ridge por estabilidad.

    El ridge evita que reviente cuando una remuestra del bootstrap queda
    separada, o sea cuando todos los exitos estan a un lado del corte.
    """
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-8)
        A = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        paso = np.linalg.solve(A, X.T @ (y - p) - ridge * b)
        b = b + paso
        if np.max(np.abs(paso)) < 1e-9:
            break
    return b


def devianza(x, y, b):
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(b[0] + b[1] * x, -30, 30))), 1e-9, 1 - 1e-9)
    return float(-2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cruce(b, log_x):
    """Punto donde el ajuste pasa por 0.5 y ancho de la transicion 25%-75%."""
    if abs(b[1]) < 1e-8:
        return np.nan, np.nan
    x50 = -b[0] / b[1]
    lo, hi = (-np.log(3) - b[0]) / b[1], (np.log(3) - b[0]) / b[1]
    if log_x:
        x50, lo, hi = np.exp(x50), np.exp(lo), np.exp(hi)
    return float(x50), float(abs(hi - lo))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def analizar(d):
    snids = sorted(set(d['barrido'][str(DFS_DT[0])]['snid']))
    idx_obj = {s: i for i, s in enumerate(snids)}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(snids), len(snids)) for _ in range(N_BOOT)]
    res = {'n_objetos': len(snids), 'piso_inversor': d['piso_inversor'],
           'saturacion': d['saturacion'], 'amplificacion': {}, 'curvas': {},
           'umbrales': {}, 'devianzas': {}}

    print(f'\n{"=" * 92}\nAMPLIFICACION  [IC 95% de Wilson]  y borrado de puntos')
    print(f'  {"df":>5}' + ''.join(f'{m:>30}' for m in METODOS))
    for df in DFS:
        res['amplificacion'][str(df)] = {}
        linea = f'  {df:5d}'
        for m in METODOS:
            a = d['amplificacion_cruda'][str(df)][m]
            v = np.array([x for x in a['peor'] if x is not None], bool)
            lo, hi = wilson(int(v.sum()), len(v))
            res['amplificacion'][str(df)][m] = {
                'p': float(v.mean()), 'lo': lo, 'hi': hi, 'n': int(len(v)),
                'borrado': float(np.mean(a['borrado'])),
            }
            linea += (f'{100 * v.mean():13.1f}% [{100 * lo:4.1f},{100 * hi:5.1f}]'
                      f' b{100 * np.mean(a["borrado"]):4.1f}%')
        print(linea, flush=True)

    for df in DFS_DT:
        f = d['barrido'][str(df)]
        obj = np.array([idx_obj[s] for s in f['snid']])
        dt = np.array(f['dt'], float)
        npts = np.array(f['n'], float)
        res['curvas'][str(df)] = {}
        res['umbrales'][str(df)] = {}
        res['devianzas'][str(df)] = {}

        # Deduplicacion para la version en puntos: dos duraciones distintas que
        # tocan exactamente las mismas epocas son la misma configuracion. No
        # depende del metodo, se calcula una vez.
        vistos, unico = set(), np.zeros(len(dt), bool)
        for i, k in enumerate(f['clave']):
            if k not in vistos:
                vistos.add(k)
                unico[i] = True

        print(f'\n{"=" * 92}\ndf={df}   {len(dt)} filas, {int(unico.sum())} '
              f'configuraciones distintas de ventana')
        for m in METODOS:
            neto = np.array(f[m]['neto'], float)
            finito = np.isfinite(neto)
            borra = np.array(f[m]['borrados'], float) > 0
            frac_borr = (np.array(f[m]['borrados'], float) /
                         np.maximum(np.array(f[m]['n_vent_pos'], float), 1))

            # Dos convenciones para el exito, porque el borrado de un punto no es
            # dato faltante sino un desenlace del metodo:
            #   estricto  el contrafactual tiene que reparar SIN borrar ninguna
            #             epoca. Borrar es fallar, asi que toda fila es evaluable
            #             y no hay datos faltantes que sesguen nada.
            #   medible   solo entre las filas donde se pudo medir, o sea la
            #             calidad de la reparacion cuando el punto sobrevive.
            conv = {'estricto': ((neto > 0) & ~borra).astype(float),
                    'medible': (neto > 0).astype(float)}
            sel_base = {'estricto': np.ones(len(dt), bool), 'medible': finito}

            # Comparacion justa entre los dos predictores: mismas filas, misma
            # familia, mismo numero de parametros. La version por puntos se
            # ajusta sobre configuraciones deduplicadas, asi que la de dias se
            # reajusta sobre ese mismo subconjunto solo para comparar devianza.
            just = unico & finito
            yj = (neto > 0).astype(float)[just]
            cmpd = {}
            for nom, x in (('dias', np.log(dt)[just]), ('puntos', np.log(npts)[just])):
                cmpd[nom] = devianza(x, yj, logistica(x, yj)) / int(just.sum())
            cmpd['n_filas'] = int(just.sum())
            cmpd['mejor'] = 'puntos' if cmpd['puntos'] < cmpd['dias'] else 'dias'
            res['devianzas'].setdefault(str(df), {})[m] = cmpd

            res['curvas'][str(df)][m] = {}
            res['umbrales'][str(df)][m] = {}
            print(f'\n  [{m}]')
            print('    dias        ' + '  '.join(f'{t:>4}' for t in DTS))
            print('    % borrado   ' + '  '.join(
                f'{100 * frac_borr[dt == t].mean():4.0f}' for t in DTS))
            print('    puntos      ' + '  '.join(f'{i:>4}' for i in range(1, 13)))
            print('    casos       ' + '  '.join(
                f'{int((unico & (npts == i)).sum()):4d}' for i in range(1, 13)))
            print(f'    predictor sobre las mismas {cmpd["n_filas"]} filas: '
                  f'dias {cmpd["dias"]:.4f}  vs  puntos {cmpd["puntos"]:.4f}   '
                  f'gana {cmpd["mejor"]}')

            for cv, y in conv.items():
                base = sel_base[cv]
                keep = unico & base
                filas = {'dias': [np.where(base & (obj == o))[0]
                                  for o in range(len(snids))],
                         'puntos': [np.where(keep & (obj == o))[0]
                                    for o in range(len(snids))]}

                emp = {}
                for t in DTS:
                    s = base & (dt == t)
                    lo, hi = wilson(int(y[s].sum()), int(s.sum()))
                    emp[str(t)] = {'p': float(y[s].mean()), 'lo': lo, 'hi': hi,
                                   'n': int(s.sum()), 'n_pts': float(npts[s].mean())}
                empn = {}
                for k in range(1, 13):
                    s = keep & (npts == k)
                    if s.sum() >= 15:
                        lo, hi = wilson(int(y[s].sum()), int(s.sum()))
                        empn[str(k)] = {'p': float(y[s].mean()), 'lo': lo, 'hi': hi,
                                        'n': int(s.sum())}
                res['curvas'][str(df)][m][cv] = {'por_dt': emp, 'por_puntos': empn}

                # Ambos predictores en log, para que la comparacion de devianza
                # entre dias y puntos sea justa: mismo numero de parametros y
                # misma familia funcional.
                u = {}
                for nom, x, sel, x_lo, x_hi in (
                        ('dias', np.log(dt), base, DTS[0], DTS[-1]),
                        ('puntos', np.log(npts), keep, 1, 12)):
                    b = logistica(x[sel], y[sel])
                    x50, anc = cruce(b, True)
                    bs = []
                    for bi in boots:
                        tomar = np.concatenate([filas[nom][o] for o in bi])
                        c, _ = cruce(logistica(x[tomar], y[tomar]), True)
                        if np.isfinite(c):
                            bs.append(c)
                    bs = np.clip(np.array(bs), 0, 1e4)
                    u[nom] = {
                        'x50': x50, 'ancho_25_75': anc, 'beta': float(b[1]),
                        'lo': float(np.percentile(bs, 2.5)),
                        'hi': float(np.percentile(bs, 97.5)),
                        'frac_dentro_grilla': float(np.mean((bs >= x_lo) & (bs <= x_hi))),
                        'devianza': devianza(x[sel], y[sel], b) / int(sel.sum()),
                        'n_filas': int(sel.sum()),
                    }
                res['umbrales'][str(df)][m][cv] = u

                print(f'    -- {cv}')
                print('       % dt     ' + '  '.join(
                    f'{100 * emp[str(t)]["p"]:4.0f}' for t in DTS))
                print('       +-IC     ' + '  '.join(
                    f'{100 * (emp[str(t)]["hi"] - emp[str(t)]["lo"]) / 2:4.1f}'
                    for t in DTS))
                print('       % pts    ' + '  '.join(
                    f'{100 * empn[str(i)]["p"]:4.0f}' if str(i) in empn else '   .'
                    for i in range(1, 13)))
                for nom in ('dias', 'puntos'):
                    z = u[nom]
                    print(f'       cruce 50% en {nom:>6}: {z["x50"]:7.2f}  '
                          f'IC95 [{z["lo"]:6.2f}, {z["hi"]:8.2f}]   '
                          f'ancho 25-75%: {z["ancho_25_75"]:7.2f}   '
                          f'dev/fila {z["devianza"]:.4f}', flush=True)

    with open(OUT / 'contrafactual_umbral_analisis.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\nAnalisis en {OUT}/contrafactual_umbral_analisis.json', flush=True)


if __name__ == '__main__':
    # El crudo queda guardado antes de analizar, asi que un error en el ajuste no
    # obliga a repetir las horas de GPU.
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_umbral.json') as f:
            analizar(json.load(f))
    else:
        main()
