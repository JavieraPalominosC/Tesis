"""Ablacion para MEJORAR el contrafactual, no para medirlo mejor.

Hasta aca medimos cada vez mejor un metodo que nunca intentamos arreglar. El
estado del arte propio es 29.2% de reduccion con 35% de borrado y 21% de
confusion de banda, que no es bueno. Se prueban tres palancas.

  prior         Todos los scripts cargan MODELOS['aleatoria'], el MaskGIT
                entrenado con tokens dispersos. Existe MODELOS['ventana'],
                entrenado sobre el mismo mad300 completo con mascara rectangular
                contigua, y nunca se uso para generar. La variante de region
                contigua enmascara justamente un rectangulo.

                Matiz importante: el entrenamiento aleatorio NO usa 15% fijo,
                samplea la razon de una distribucion coseno sobre todo [0,1], asi
                que ese modelo si vio agujeros grandes. Lo que no vio nunca es un
                agujero COMPACTO. La discrepancia es de forma, no de tamano. Por
                eso se mide la geometria de cada mascara: cuantos tokens
                enmascarados conservan un vecino visible. Si la mascara p95
                resulta tan agrupada como el rectangulo, la hipotesis de la
                discrepancia de forma no explica que a p95 le vaya peor, y habra
                que atribuirlo a otra cosa.

  decodificacion  El muestreo actual es multinomial sobre todo el vocabulario
                con recocido desde T=1. Para un contrafactual se quiere la moda,
                no una muestra: el revelado es progresivo y lo revelado queda
                fijo, asi que un token de la cola en el paso 1 contamina el
                contexto de todos los demas. Traza tenue, fragmentada y de color
                equivocado es la firma de muestrear caliente.

  n_steps       10 pasos para ~150 tokens revela quince por paso.

METRICA. Se usan los diagnosticos baratos ya calibrados, que no necesitan el
pipeline de flujo: tinta por epoca contra el piso limpio 0.740, borrado con el
extractor calibrado 0.73, y confusion de banda. Solo las configuraciones que
sobrevivan pasan a FLUXCAL.

Uso: python scripts/contrafactual_ablacion.py [N_OBJETOS] [DF]
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
    col_flux, col_mjd, fig_params, mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import PX, reconstruir
from scripts.contrafactual_comparacion import (
    GRID, R_LOC, nll_por_token, rect_max_exceso, seleccionar,
)
from scripts.contrafactual_flujo import col_de
from scripts.contrafactual_extractor import J_R, banda_dominante, pesos, stats
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
PISO_TINTA = 0.740        # minimo de tinta en reconstrucciones limpias
U_CAL = 0.73              # extractor calibrado
N_OBJ, DF, SEED = 80, 300, 0
PRIORS = ['aleatoria', 'ventana']
FORMAS = ['p95', 'rect']
DECOD = {'actual': dict(modo='muestreo', temperatura=True),
         'greedy': dict(modo='greedy'),
         'topk3': dict(modo='topk', k=3),
         'topk10': dict(modo='topk', k=10)}
PASOS_EXTRA = [25, 50]
# El barrido de n_steps solo tiene sentido donde el revelado progresivo puede
# ayudar, o sea donde el agujero es grande y el prior no esta perdido. Se limita
# a las combinaciones que sobrevivieron la prueba corta para no gastar 3x en
# ramas ya descartadas.
CON_PASOS = {('ventana', 'rect', 'greedy'), ('ventana', 'rect', 'topk10'),
             ('ventana', 'rect', 'actual'), ('ventana', 'p95', 'topk10'),
             ('aleatoria', 'rect', 'topk10')}


def pasos_de(pr, f, dec):
    return [10] + (PASOS_EXTRA if (pr, f, dec) in CON_PASOS else [])


def geometria(m):
    """Cuantos tokens enmascarados conservan un vecino visible, y cuantos son.

    Es el descriptor que separa un agujero disperso de uno compacto, que es
    justo la diferencia entre lo que vio cada prior en entrenamiento.
    """
    g = m.reshape(GRID, GRID)
    n = int(g.sum())
    if n == 0:
        return {'n': 0, 'borde': np.nan, 'borde_azar': np.nan}
    vis = ~g
    vec = np.zeros_like(g)
    vec[:-1, :] |= vis[1:, :]
    vec[1:, :] |= vis[:-1, :]
    vec[:, :-1] |= vis[:, 1:]
    vec[:, 1:] |= vis[:, :-1]
    rho = n / g.size
    return {'n': n, 'borde': float((g & vec).sum() / n),
            'borde_azar': float(1 - rho ** 4)}


def diagnosticos(x, o, t0, rango, pos, c0s):
    """Tinta, borrado y confusion de banda en las epocas inyectadas."""
    w = pesos(o, x, t0, rango)
    out = []
    for k, i in enumerate(pos):
        b = banda_dominante(x, J_R, int(c0s[k]))
        out.append({'w': float(w[i]), 'banda_ok': None if b is None else int(b) == J_R})
    return out


def ee_agrupado(sub, umbral, n_boot=400, seed=0):
    """Error estandar de la tasa remuestreando OBJETOS, no epocas."""
    por_obj = {}
    for r in sub:
        por_obj.setdefault(r['snid'], []).append(r['w'] < umbral)
    v = [np.array(x, float) for x in por_obj.values()]
    if len(v) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    b = [np.concatenate([v[i] for i in rng.integers(0, len(v), len(v))]).mean()
         for _ in range(n_boot)]
    return float(np.std(b))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_OBJ
    df = int(sys.argv[2]) if len(sys.argv) > 2 else DF
    curvas = candidatos(n_obj)
    snids = sorted(curvas)
    print(f'objetos {len(snids)}   df={df}   piso de tinta {PISO_TINTA}', flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mgs = {p: MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS[p]), map_location=dev,
                                           strict=False).eval().to(dev)
           for p in PRIORS}
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
    for s in snids:
        o, idx, sc = curvas[s]
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        p = o.copy()
        p.loc[idx, col_flux] = p.loc[idx, col_flux] + df * sc
        med_p, esc_p = stats(p[col_flux].values)
        pos = o.index.get_indexer(idx)
        c0s = np.clip(np.round(col_de((o[col_mjd].values[pos] - t0) / rango)
                               ).astype(int), 0, PX - 1)
        with torch.no_grad():
            tp = tok(render(p, ENC, fr, ar))
            # Referencia: la perturbada sin tocar, para separar lo que aporta el
            # contrafactual de lo que ya trae la reconstruccion.
            for d in diagnosticos(reconstruir(vq, tp), o, t0, rango, pos, c0s):
                filas.append({**d, 'cfg': 'perturbada|-|-|0', 'snid': s})

            for pr in PRIORS:
                mg = mgs[pr]
                nll = nll_por_token(mg, tp, dev, g, rondas=R_LOC)
                sel, _ = seleccionar(nll, 0.95)
                r0, r1, cc0, cc1, _, _ = rect_max_exceso(sel.reshape(GRID, GRID))
                mr = np.zeros((GRID, GRID), bool)
                mr[r0:r1 + 1, cc0:cc1 + 1] = True
                masc = {'p95': sel.reshape(GRID, GRID), 'rect': mr}
                for f, mm in masc.items():
                    geo.append({'prior': pr, 'forma': f, **geometria(mm)})
                    mt = torch.from_numpy(mm.reshape(1, -1)).to(dev)
                    for dec, kw in DECOD.items():
                        for ns in pasos_de(pr, f, dec):
                            cf = mg.generate_counterfactual(tp, mt, n_steps=ns, **kw)
                            x = reconstruir(vq, cf)
                            for d in diagnosticos(x, o, t0, rango, pos, c0s):
                                filas.append({**d, 'cfg': f'{pr}|{f}|{dec}|{ns}', 'snid': s})
        if (snids.index(s) + 1) % 20 == 0:
            print(f'  {snids.index(s) + 1}/{len(snids)} objetos', flush=True)
    plt.close(fr)

    salida = {'df': df, 'n_obj': len(snids), 'filas': filas, 'geo': geo}
    with open(OUT / 'contrafactual_ablacion.json', 'w') as f:
        json.dump(salida, f)
    analizar(salida)


def analizar(d):
    print(f'\n{"=" * 104}\nGEOMETRIA DE LA MASCARA  (es p95 realmente dispersa?)')
    print(f'  {"prior":<11}{"forma":<7}{"n tokens":>10}{"% con vecino visible":>22}'
          f'{"lo mismo si fuera al azar":>28}')
    for pr in PRIORS:
        for f in FORMAS:
            sub = [x for x in d['geo'] if x['prior'] == pr and x['forma'] == f]
            if not sub:
                continue
            n = np.mean([x['n'] for x in sub])
            b = np.nanmean([x['borde'] for x in sub])
            ba = np.nanmean([x['borde_azar'] for x in sub])
            print(f'  {pr:<11}{f:<7}{n:10.1f}{100 * b:21.1f}%{100 * ba:27.1f}%')

    print(f'\n{"=" * 104}\nDIAGNOSTICOS BARATOS   df={d["df"]}, {d["n_obj"]} objetos')
    print('  El +- es bootstrap agrupado por objeto: las epocas de una misma curva\n'
          '  no son independientes, asi que el error binomial simple mentiria.')
    print(f'  {"configuracion":<32}{"tinta mediana":>15}{"% bajo piso":>17}'
          f'{"% borrado 0.73":>16}{"% banda mala":>14}{"n":>7}')
    res = {}
    orden = ['perturbada|-|-|0'] + [
        f'{pr}|{f}|{dec}|{ns}' for pr in PRIORS for f in FORMAS
        for dec in DECOD for ns in pasos_de(pr, f, dec)]
    for cfg in orden:
        sub = [r for r in d['filas'] if r['cfg'] == cfg]
        if not sub:
            continue
        w = np.array([r['w'] for r in sub])
        ok = [r['banda_ok'] for r in sub if r['banda_ok'] is not None]
        e = {'tinta': float(np.median(w)),
             'bajo_piso': float((w < PISO_TINTA).mean()),
             'ee': ee_agrupado(sub, PISO_TINTA),
             'borrado': float((w < U_CAL).mean()),
             'banda_mala': float(np.mean([not b for b in ok])) if ok else np.nan,
             'n': len(sub)}
        res[cfg] = e
        print(f'  {cfg:<32}{e["tinta"]:15.3f}'
              f'{100 * e["bajo_piso"]:11.1f} +-{100 * e["ee"]:3.1f}'
              f'{100 * e["borrado"]:15.1f}%{100 * e["banda_mala"]:13.1f}%{e["n"]:7d}',
              flush=True)

    base = res.get('aleatoria|p95|actual|10')
    if base:
        print(f'\n{"=" * 104}\nMEJORA RESPECTO DE LA CONFIGURACION ACTUAL '
              f'(aleatoria|p95|actual|10)')
        print(f'  {"configuracion":<32}{"d % bajo piso":>15}{"d % borrado":>14}'
              f'{"d % banda mala":>16}')
        for cfg, e in sorted(res.items(), key=lambda x: x[1]['bajo_piso']):
            if cfg.startswith('perturbada'):
                continue
            print(f'  {cfg:<32}{100 * (e["bajo_piso"] - base["bajo_piso"]):+14.1f}'
                  f'{100 * (e["borrado"] - base["borrado"]):+13.1f}'
                  f'{100 * (e["banda_mala"] - base["banda_mala"]):+15.1f}')

    with open(OUT / 'contrafactual_ablacion_resumen.json', 'w') as f:
        json.dump({'res': res, 'geo': d['geo'][:200], 'df': d['df']}, f, indent=2)
    print(f'\nResumen en {OUT}/contrafactual_ablacion_resumen.json', flush=True)


if __name__ == '__main__':
    if '--solo-analisis' in sys.argv:
        with open(OUT / 'contrafactual_ablacion.json') as f:
            analizar(json.load(f))
    else:
        main()
