"""Segunda etapa de la ablacion: la forma de la mascara, que es lo que domina.

La primera etapa dejo tres cosas medidas sobre 300 objetos:

  - el prior no explica nada. El MaskGIT de ventana NO es mejor rellenando
    rectangulos; sobre mascara rect es peor que el aleatorio (39.6% contra
    35.8% bajo el piso de tinta). Solo ayuda sobre mascara dispersa. Lo
    contrario de lo que predice el argumento de la forma del agujero.
  - la decodificacion tampoco. Entre muestreo recocido, greedy y top-k el rango
    completo es de un punto sobre p95 y de cinco sobre rect, y greedy es la PEOR
    sobre rect. Mas pasos de revelado empeoran de forma monotona.
  - lo unico que mueve la aguja es la forma de la mascara: rect ~35%, p95 ~57%,
    consistente en los cuatro cruces de prior por decodificacion.

Y dejo una pregunta abierta. La mascara p95 es genuinamente dispersa: el 99.3%
de sus tokens conserva un vecino visible, contra 43% en rect. O sea que p95 es
lo mas parecido al entrenamiento aleatorio y aun asi es la que peor funciona, lo
que mata la explicacion por discrepancia de forma. La hipotesis que queda es la
coherencia del contexto: p95 marca 50 tokens sueltos y deja en pie el resto de
la anomalia, asi que el modelo reconstruye cada token enmascarado condicionado a
los tokens anomalos vecinos y reproduce la anomalia. El rectangulo borra el
barrio completo y obliga a completar desde contexto limpio.

Este script pone a prueba esa explicacion midiendo, para cada forma, cuanto de
la zona real de inyeccion queda cubierta, y probando formas con mas margen. El
oraculo (la zona de inyeccion exacta) va incluido como techo: dice si el limite
esta en localizar o en generar.

Uso: python scripts/contrafactual_ablacion2.py [N_OBJETOS] [DF]
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
from scripts.contrafactuales import PX, reconstruir
from scripts.contrafactual_comparacion import (
    GRID, R_LOC, nll_por_token, rect_max_exceso, seleccionar,
)
from scripts.contrafactual_flujo import col_de
from scripts.contrafactual_extractor import J_R, banda_dominante, pesos, stats
from scripts.analizar_maskgits import zona_objetivo
from scripts.contrafactual_ablacion import PISO_TINTA, U_CAL, ee_agrupado, geometria
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
N_OBJ, DF, SEED = 300, 300, 0
FORMAS = ['p95', 'p95 dil1', 'p95 dil2', 'rect', 'rect m1', 'rect m2',
          'rect ancho', 'oraculo']
DECOD = {'actual': dict(modo='muestreo', temperatura=True),
         'topk10': dict(modo='topk', k=10)}
# El prior de ventana solo se vuelve a mirar sobre la forma ganadora, para
# confirmar en la etapa 2 lo que la etapa 1 ya dijo, y no para explorarlo otra
# vez: sobre rect quedo medido que no aporta.
PRIOR_EXTRA = ('ventana', 'rect m1', 'topk10')


def dilatar(m, r):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0.5


def con_margen(r0, r1, c0, c1, m):
    z = np.zeros((GRID, GRID), bool)
    z[max(0, r0 - m):min(GRID, r1 + 1 + m), max(0, c0 - m):min(GRID, c1 + 1 + m)] = True
    return z


def formas_de(sel, zona):
    """Todas las mascaras candidatas para un mismo mapa de NLL."""
    s = sel.reshape(GRID, GRID)
    r0, r1, c0, c1, _, _ = rect_max_exceso(s)
    # 'rect ancho' usa penalizacion de area 1.0 en vez de 2.0, o sea acepta
    # rectangulos mas grandes por el mismo numero de tokens marcados.
    a0, a1, b0, b1, _, _ = rect_max_exceso(s, q=1.0 * s.astype(float).mean())
    return {'p95': s, 'p95 dil1': dilatar(s, 1), 'p95 dil2': dilatar(s, 2),
            'rect': con_margen(r0, r1, c0, c1, 0),
            'rect m1': con_margen(r0, r1, c0, c1, 1),
            'rect m2': con_margen(r0, r1, c0, c1, 2),
            'rect ancho': con_margen(a0, a1, b0, b1, 0),
            'oraculo': zona}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    df = int(sys.argv[2]) if len(sys.argv) > 2 else DF
    curvas = candidatos(n_obj)
    snids = sorted(curvas)

    zr0, zr1, zc0, zc1 = zona_objetivo()
    zona = np.zeros((GRID, GRID), bool)
    zona[zr0:zr1 + 1, zc0:zc1 + 1] = True
    print(f'objetos {len(snids)}   df={df}   zona de inyeccion {int(zona.sum())} tokens',
          flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mgs = {p: MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS[p]), map_location=dev,
                                           strict=False).eval().to(dev)
           for p in ('aleatoria', 'ventana')}
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def tok(img):
        return vq.encode_to_indices(tf(img).unsqueeze(0).to(dev)).view(1, -1).long().to(dev)

    filas, geo = [], []
    for n, s in enumerate(snids):
        o, idx, sc = curvas[s]
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        p = o.copy()
        p.loc[idx, col_flux] = p.loc[idx, col_flux] + df * sc
        pos = o.index.get_indexer(idx)
        c0s = np.clip(np.round(col_de((o[col_mjd].values[pos] - t0) / rango)
                               ).astype(int), 0, PX - 1)
        with torch.no_grad():
            tp = tok(render(p, ENC, fr, ar))
            nll = nll_por_token(mgs['aleatoria'], tp, dev, g, rondas=R_LOC)
            sel, _ = seleccionar(nll, 0.95)
            masc = formas_de(sel, zona)
            for f, mm in masc.items():
                nz = max(int(zona.sum()), 1)
                geo.append({'forma': f, **geometria(mm),
                            'cobertura': float((mm & zona).sum() / nz),
                            'precision': float((mm & zona).sum() / max(int(mm.sum()), 1))})
                mt = torch.from_numpy(mm.reshape(1, -1)).to(dev)
                combos = [('aleatoria', d) for d in DECOD]
                if (PRIOR_EXTRA[1], PRIOR_EXTRA[2]) == (f, 'topk10'):
                    combos.append((PRIOR_EXTRA[0], PRIOR_EXTRA[2]))
                for pr, dec in combos:
                    cf = mgs[pr].generate_counterfactual(tp, mt, n_steps=10,
                                                         **DECOD[dec])
                    x = reconstruir(vq, cf)
                    w = pesos(o, x, t0, rango)
                    for k, i in enumerate(pos):
                        b = banda_dominante(x, J_R, int(c0s[k]))
                        filas.append({'cfg': f'{pr}|{f}|{dec}', 'snid': s,
                                      'w': float(w[i]),
                                      'banda_ok': None if b is None else int(b) == J_R})
        if (n + 1) % 50 == 0:
            print(f'  {n + 1}/{len(snids)} objetos', flush=True)
    plt.close(fr)

    salida = {'df': df, 'n_obj': len(snids), 'filas': filas, 'geo': geo}
    with open(OUT / 'contrafactual_ablacion2.json', 'w') as f:
        json.dump(salida, f)
    analizar(salida)


def analizar(d):
    print(f'\n{"=" * 112}\nQUE CUBRE CADA FORMA DE LA ZONA REAL DE INYECCION')
    print(f'  {"forma":<13}{"n tokens":>10}{"% con vecino visible":>22}'
          f'{"cobertura de la zona":>22}{"% del area dentro":>19}')
    for f in FORMAS:
        sub = [x for x in d['geo'] if x['forma'] == f]
        if not sub:
            continue
        print(f'  {f:<13}{np.mean([x["n"] for x in sub]):10.1f}'
              f'{100 * np.nanmean([x["borde"] for x in sub]):21.1f}%'
              f'{100 * np.mean([x["cobertura"] for x in sub]):21.1f}%'
              f'{100 * np.mean([x["precision"] for x in sub]):18.1f}%')

    print(f'\n{"=" * 112}\nDIAGNOSTICOS   df={d["df"]}, {d["n_obj"]} objetos')
    print(f'  {"configuracion":<28}{"tinta mediana":>15}{"% bajo piso":>17}'
          f'{"% borrado 0.73":>16}{"% banda mala":>14}{"n":>7}')
    res = {}
    orden = sorted({r['cfg'] for r in d['filas']},
                   key=lambda c: (c.split('|')[0], FORMAS.index(c.split('|')[1]),
                                  c.split('|')[2]))
    for cfg in orden:
        sub = [r for r in d['filas'] if r['cfg'] == cfg]
        w = np.array([r['w'] for r in sub])
        ok = [r['banda_ok'] for r in sub if r['banda_ok'] is not None]
        e = {'tinta': float(np.median(w)),
             'bajo_piso': float((w < PISO_TINTA).mean()),
             'ee': ee_agrupado(sub, PISO_TINTA),
             'borrado': float((w < U_CAL).mean()),
             'banda_mala': float(np.mean([not b for b in ok])) if ok else np.nan,
             'n': len(sub)}
        res[cfg] = e
        print(f'  {cfg:<28}{e["tinta"]:15.3f}'
              f'{100 * e["bajo_piso"]:11.1f} +-{100 * e["ee"]:3.1f}'
              f'{100 * e["borrado"]:15.1f}%{100 * e["banda_mala"]:13.1f}%{e["n"]:7d}',
              flush=True)

    ref = res.get('aleatoria|rect|actual')
    if ref:
        print(f'\n{"=" * 112}\nMEJORA SOBRE LA CONFIGURACION EN USO '
              f'(aleatoria|rect|actual, {100 * ref["bajo_piso"]:.1f}% bajo piso)')
        print(f'  {"configuracion":<28}{"d % bajo piso":>15}{"d % borrado":>14}'
              f'{"d % banda mala":>16}')
        for cfg, e in sorted(res.items(), key=lambda x: x[1]['bajo_piso']):
            print(f'  {cfg:<28}{100 * (e["bajo_piso"] - ref["bajo_piso"]):+14.1f}'
                  f'{100 * (e["borrado"] - ref["borrado"]):+13.1f}'
                  f'{100 * (e["banda_mala"] - ref["banda_mala"]):+15.1f}')

    with open(OUT / 'contrafactual_ablacion2_resumen.json', 'w') as f:
        json.dump({'res': res, 'df': d['df'], 'n_obj': d['n_obj']}, f, indent=2)
    print(f'\nResumen en {OUT}/contrafactual_ablacion2_resumen.json', flush=True)


if __name__ == '__main__':
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_ablacion2.json') as f:
            analizar(json.load(f))
    else:
        main()
