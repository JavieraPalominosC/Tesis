"""Cierre en espacio de flujo: umbral de tamano, amplificacion y saturacion.

Rehace en FLUXCAL la pregunta del tamano minimo de anomalia. La conclusion
anterior ("neto negativo en todos los dt, sin umbral") era cierta en espacio de
pixeles, que ahora sabemos que mide temblor subpixel y no fisica.

TRES PARTES.

  saturacion    mad300 recorta en 300 MAD del objeto. Se mide que fraccion de
                las epocas inyectadas cae sobre ese techo en cada df, para saber
                si las conclusiones principales pueden apoyarse en df=300 y 500
                o hay que quedarse en los niveles no saturados.

  amplificacion En que fraccion de casos el contrafactual deja el residuo de
                flujo PEOR que la perturbada, o sea que amplifica la anomalia en
                vez de reducirla. La figura de curvas mostro dos ejemplos; esto
                le pone numero, por metodo y por df.

  umbral        Barrido de duracion en flujo. Aprovecha algo que el espacio de
                pixeles no permitia: identificar exactamente cuales epocas
                fueron perturbadas. Entonces la zona deja de ser un rectangulo
                fijo que se diluye cuando la anomalia es corta, y pasa a ser el
                conjunto exacto de puntos inyectados. Se reportan igual los dos
                cortes, la ventana real y la cola completa, para poder comparar
                con el analisis anterior.

Se guardan valores absolutos ademas de relativos: el problema de la vez pasada
fue que las razones explotaban cuando el denominador colapsaba. Y se reporta
siempre el piso del inversor, porque con ventanas cortas el residuo puede
acercarse a ese piso y ahi un porcentaje ya no significa nada.

Uso: python scripts/contrafactual_flujo_cierre.py
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
DTS = [5, 10, 15, 20, 30, 45, 60, 90]
DFS_DT = [100, 300]
METODOS = ['base p95', 'rect 1 pasa', 'azar']
N_OBJ, SEED = 40, 0


def stats(v):
    v = np.asarray(v, dtype='float64')
    m = np.median(v)
    s = np.median(np.abs(v - m))
    return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)


def rmse(a, b, m):
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(OUT / 'contrafactual_flujo.json') as f:
        piso = json.load(f)['piso_inversor_rmse']
    print(f'piso del inversor: {piso:.2f} FLUXCAL (RMSE de ida y vuelta sobre la '
          f'curva limpia reconstruida)', flush=True)

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
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

    salida = {'piso_inversor': piso}

    # ------------------------------------------------------------ saturacion
    print(f'\n{"=" * 84}\nSATURACION: mad300 recorta en {U_MAX:.0f} MAD del objeto',
          flush=True)
    print(f'  {"df":>6}{"|f-med|/MAD mediano":>22}{"p90":>9}'
          f'{"% epocas inyectadas recortadas":>32}')
    sat = {}
    for d in DFS:
        rel, rec = [], []
        for s in snids:
            o, idx_cola, sc = curvas[s]
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            med, esc = stats(p[col_flux].values)
            f = p.loc[idx_cola, col_flux].values.astype('float64')
            r = np.abs(f - med) / esc
            rel.append(np.median(r))
            rec.append(float(np.mean(r > U_MAX)))
        sat[str(d)] = {'rel_mediano': float(np.median(rel)),
                       'rel_p90': float(np.percentile(rel, 90)),
                       'frac_recortada': float(np.mean(rec))}
        x = sat[str(d)]
        marca = '   <- saturado' if x['frac_recortada'] > 0.10 else ''
        print(f'  {d:6d}{x["rel_mediano"]:22.1f}{x["rel_p90"]:9.1f}'
              f'{100 * x["frac_recortada"]:31.1f}%{marca}', flush=True)
    salida['saturacion'] = sat

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    def evaluar(o, idx_pert, idx_cola, d, sc):
        """Devuelve residuos en flujo de la perturbada y de cada metodo."""
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
            xp = reconstruir(vq, tp)
            fp = invertir(xp, med_p, esc_p, o, t0, rango)
            out = {'perturbada': fp}
            for m, (cf, _) in {
                    'base p95': metodo_base(mg, tp, dev, g, 0.95),
                    'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                    'azar': metodo_azar(mg, tp, dev, g, 148)}.items():
                out[m] = invertir(reconstruir(vq, cf), med_p, esc_p, o, t0, rango)

        ok = np.isfinite(out['perturbada'])
        for v in out.values():
            ok &= np.isfinite(v)
        return {k: {
            'vent': rmse(v, ftrue, ok & vent),
            'cola': rmse(v, ftrue, ok & cola),
            'resto': rmse(v, ftrue, ok & ~vent),
            'n_vent': int((ok & vent).sum()), 'n_resto': int((ok & ~vent).sum()),
        } for k, v in out.items()}

    # --------------------------------------------------------- amplificacion
    print(f'\n{"=" * 84}\nAMPLIFICACION: casos donde el contrafactual deja el residuo '
          f'PEOR que la perturbada', flush=True)
    print(f'  {"df":>6}' + ''.join(f'{m:>16}' for m in METODOS))
    amp = {m: {} for m in METODOS}
    for d in DFS:
        peor = {m: [] for m in METODOS}
        for s in snids:
            o, idx_cola, sc = curvas[s]
            r = evaluar(o, idx_cola, idx_cola, d, sc)
            for m in METODOS:
                peor[m].append(r[m]['cola'] > r['perturbada']['cola'])
        for m in METODOS:
            amp[m][str(d)] = float(np.mean(peor[m]))
        print(f'  {d:6d}' + ''.join(f'{100 * amp[m][str(d)]:15.0f}%' for m in METODOS),
              flush=True)
    salida['amplificacion'] = amp

    # ---------------------------------------------------------------- umbral
    print(f'\n{"=" * 84}\nUMBRAL DE TAMANO EN FLUJO', flush=True)
    ventanas = {}
    for s in snids:
        o, idx_cola, _ = curvas[s]
        sub = o.loc[idx_cola]
        t_fin = float(sub[col_mjd].max())
        for dt in DTS:
            ventanas[(s, dt)] = list(sub.index[sub[col_mjd] >= t_fin - dt])
    print('\n  puntos fotometricos reales por ventana (banda r):')
    print(f'  {"dt[dias]":>9}{"mediana":>9}{"media":>8}{"min":>6}{"max":>6}')
    pts = {}
    for dt in DTS:
        c = np.array([len(ventanas[(s, dt)]) for s in snids])
        pts[str(dt)] = c.tolist()
        print(f'  {dt:9d}{np.median(c):9.1f}{c.mean():8.2f}{c.min():6d}{c.max():6d}')
    salida['puntos_por_dt'] = pts

    umbral = {}
    for d in DFS_DT:
        umbral[str(d)] = {m: {} for m in METODOS}
        print(f'\n{"=" * 84}\ndf={d}   (piso del inversor {piso:.2f} FLUXCAL)')
        for dt in DTS:
            acc = {m: [] for m in METODOS}
            base = []
            for s in snids:
                o, idx_cola, sc = curvas[s]
                r = evaluar(o, ventanas[(s, dt)], idx_cola, d, sc)
                base.append(r['perturbada'])
                for m in METODOS:
                    q = r[m]
                    q['g'] = (r['perturbada']['vent'] ** 2 - q['vent'] ** 2) * q['n_vent']
                    q['p'] = (q['resto'] ** 2 - r['perturbada']['resto'] ** 2) * q['n_resto']
                    q['neto'] = q['g'] - q['p']
                    acc[m].append(q)
            pr = lambda a, k: float(np.nanmean([x[k] for x in a]))
            bv, br = pr(base, 'vent'), pr(base, 'resto')
            npts = np.mean(pts[str(dt)])
            aviso = '  [residuo perturbado al orden del piso]' if bv < 3 * piso else ''
            print(f'\n  dt={dt}d  {npts:.1f} puntos   perturbada: ventana {bv:8.1f}   '
                  f'resto {br:5.2f}{aviso}')
            print(f'  {"metodo":<13}{"ventana":>10}{"resto":>8}{"reduccion":>11}'
                  f'{"ganancia":>11}{"perdida":>10}{"neto":>12}{"% neto>0":>10}')
            for m in METODOS:
                a = acc[m]
                v, rr, ne = pr(a, 'vent'), pr(a, 'resto'), pr(a, 'neto')
                red = 100 * (bv - v) / bv if bv > 0 else np.nan
                fpos = float(np.mean([x['neto'] > 0 for x in a]))
                umbral[str(d)][m][str(dt)] = {
                    'vent': v, 'resto': rr, 'cola': pr(a, 'cola'),
                    'vent_pert': bv, 'resto_pert': br, 'reduccion_pct': red,
                    'ganancia': pr(a, 'g'), 'perdida': pr(a, 'p'), 'neto': ne,
                    'frac_neto_pos': fpos, 'n_puntos': float(npts),
                }
                print(f'  {m:<13}{v:10.1f}{rr:8.2f}{red:10.1f}%{pr(a, "g"):11.0f}'
                      f'{pr(a, "p"):10.0f}{ne:12.0f}{100 * fpos:9.0f}%', flush=True)
    salida['umbral_flujo'] = umbral
    plt.close(fr)

    with open(OUT / 'contrafactual_flujo_cierre.json', 'w') as f:
        json.dump(salida, f, indent=2)
    figura(umbral, pts, piso, amp)
    print(f'\nGuardado en {OUT}/', flush=True)


def figura(umbral, pts, piso, amp):
    COL = {'base p95': '#f2a640', 'rect 1 pasa': '#31a354', 'azar': '#8856a7'}
    MRK = {'base p95': 'v', 'rect 1 pasa': '^', 'azar': 's'}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for a, d in zip(axes[:2], DFS_DT):
        for m in METODOS:
            a.plot(DTS, [umbral[str(d)][m][str(dt)]['neto'] for dt in DTS],
                   MRK[m] + '-', color=COL[m], label=m, ms=6)
        a.axhline(0, color='#333333', lw=1.2)
        a.set_xlabel('dt: duracion de la anomalia [dias]')
        a.set_ylabel('neto en flujo = ganancia - perdida')
        a.set_title(f'df={d}\nsobre cero, repara mas de lo que rompe', fontsize=10)
        a.legend(frameon=False, fontsize=8)
        b = a.twiny()
        b.set_xlim(a.get_xlim())
        b.set_xticks(DTS)
        b.set_xticklabels([f'{np.mean(pts[str(dt)]):.1f}' for dt in DTS], fontsize=7)
        b.set_xlabel('puntos fotometricos perturbados', fontsize=8)

    a = axes[2]
    w = 0.26
    xs = np.arange(len(DFS))
    for i, m in enumerate(METODOS):
        a.bar(xs + (i - 1) * w, [100 * amp[m][str(d)] for d in DFS], w,
              color=COL[m], label=m)
    a.set_xticks(xs)
    a.set_xticklabels([str(d) for d in DFS])
    a.set_xlabel(r'$df$  [veces MAD$_r$]')
    a.set_ylabel('% de objetos')
    a.set_title('Tasa de amplificacion\nel contrafactual deja el residuo peor',
                fontsize=10)
    a.legend(frameon=False, fontsize=8)

    fig.suptitle(f'Cierre en flujo: umbral de tamano y tasa de fallo  '
                 f'(piso del inversor {piso:.1f} FLUXCAL, 40 SNIa fuera del '
                 f'entrenamiento)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'contrafactual_flujo_cierre.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
