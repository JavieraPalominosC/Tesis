"""Linea base contra variante mejorada de generacion de contrafactuales.

LINEA BASE (protocolo de la metodologia): NLL por token con R=5, enmascarar los
tokens sobre el percentil de esa imagen, una sola pasada de remuestreo. Se corre
con p85 (el valor escrito en la metodologia) y con p95, porque el barrido previo
mostro que p95 domina en df alto.

VARIANTE MEJORADA, dos cambios:

  region contigua   El NLL localiza, pero se enmascara el RECTANGULO contiguo
                    que concentra los tokens marcados, no los tokens sueltos. La
                    razon: los tokens del percentil superior estan dispersos por
                    construccion, asi que al enmascararlos sus vecinos siguen
                    visibles y siguen llevando la anomalia, y el prior puede
                    redibujar el pico desde ese contexto contaminado.

                    El rectangulo se elige con un estadistico de barrido: entre
                    todos los rectangulos posibles se toma el que maximiza
                    (tokens marcados adentro) - q * (area), con q la fraccion
                    global de marcados. Ese objetivo premia la concentracion y
                    penaliza el tamano solo, asi que no hace falta fijar un area
                    a mano; un rectangulo grande solo gana si de verdad esta
                    mas denso que el promedio. Se recorta a MAX_AREA por si
                    degenera.

  bucle de verificacion  En vez de decretar cuanto reescribir, se itera:
                    reescribir, volver a puntuar, parar cuando el puntaje cae
                    bajo el umbral de deteccion tau o al llegar al tope. Asi la
                    parsimonia deja de ser un parametro y pasa a ser un
                    resultado medido.

SOBRE EL UMBRAL. tau = 10.3927 sale de la celda aleatoria|aleatoria de
final_mad300.json, y ahi se calculo con 24 rondas, no con 5. Un maximo de NLL
sobre 1024 tokens estimado con 5 rondas es mas ruidoso y por lo tanto mas alto,
asi que no es comparable con ese tau. Se resuelve separando los dos usos: R=5
para LOCALIZAR (que es lo que pide la metodologia, basta para ordenar
posiciones) y R=24 para PUNTUAR contra tau, que es el regimen en que tau fue
calibrado.

SOBRE LA CIRCULARIDAD DEL BUCLE. El puntaje baja por remuestrear, no
necesariamente por deshacer la anomalia: ya vimos que una curva limpia tambien
baja su NLL al pasar por el contrafactual. Un bucle que para cuando el puntaje
baja puede estar explotando eso. Se blinda de tres maneras y ninguna depende del
NLL: se corre el mismo bucle sobre curvas limpias (no deberia dispararse), el
exito se declara por fidelidad separada por region contra la curva limpia
conocida, y se compara siempre contra el piso de reescribir posiciones al azar.
El criterio de tau queda como regla de PARADA, no como metrica de exito.

Uso: python scripts/contrafactual_comparacion.py
"""
import sys
sys.path.insert(0, '.')
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_enc_cmp import (
    AMP, N_BANDS, STEP, col_flux, fig_params, mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import (
    GRID, N_STEPS, PX, nll_por_token, reconstruir, seleccionar, zona_px,
)
from scripts.contrafactual_diagnostico import parte
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
TAU = 10.392719364166258   # p95 de controles, celda aleatoria|aleatoria, R=24
R_LOC, R_VER = 5, 24       # rondas para localizar y para puntuar contra tau
MAX_ITER = 4
MAX_AREA = int(0.40 * GRID * GRID)
# Penalizacion de area del estadistico de barrido, en multiplos de la densidad
# global de tokens marcados. Con factor 1 el rectangulo crece gratis sobre el
# fondo (en la prueba sintetica daba area 207 para un cumulo real de 50). Con 2
# recupera area 51 e IoU 0.78 contra el cumulo plantado. La meseta va de 1.5 a
# 2.5, asi que la eleccion no es de filo de cuchillo, pero si esta calibrada
# sobre un cumulo del tamano que nosotros inyectamos.
PENAL = 2.0
N_OBJ, SEED = 40, 0


# ---------------------------------------------------------------- rectangulo

def rect_max_exceso(flag, q=None, max_area=MAX_AREA):
    """Rectangulo que maximiza (marcados adentro) - q * area.

    Estadistico de barrido clasico. Se resuelve exacto: para cada par de filas
    (r0, r1) el problema se reduce a subarreglo contiguo de suma maxima sobre
    las columnas, que se calcula con prefijos y minimo acumulado. Todo
    vectorizado, asi que recorre los 32*33/2 pares de filas de una vez.
    """
    g = flag.shape[0]
    f = flag.astype(np.float64)
    if q is None:
        q = PENAL * f.mean()

    cs = np.concatenate([np.zeros((1, g)), np.cumsum(f, axis=0)], axis=0)
    r0s, r1s = np.triu_indices(g)
    banda = cs[r1s + 1] - cs[r0s]                       # (P, g) conteo por columna
    alto = (r1s - r0s + 1).astype(np.float64)[:, None]
    v = banda - q * alto                                # exceso por columna

    P = np.concatenate([np.zeros((len(r0s), 1)), np.cumsum(v, axis=1)], axis=1)
    minant = np.minimum.accumulate(P[:, :-1], axis=1)
    ganancia = P[:, 1:] - minant                        # (P, g), c1 = 0..g-1

    # El area tambien depende de c0, asi que el recorte por MAX_AREA se aplica
    # despues, descartando el optimo si se pasa y quedandose con el mejor valido.
    orden = np.argsort(ganancia, axis=None)[::-1]
    for plano in orden[:4096]:
        p, c1 = np.unravel_index(plano, ganancia.shape)
        c0 = int(np.argmin(P[p, :c1 + 1]))
        r0, r1 = int(r0s[p]), int(r1s[p])
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area <= max_area:
            return r0, r1, c0, int(c1), float(ganancia[p, c1]), area
    return 0, g - 1, 0, g - 1, 0.0, g * g


def _test_rect():
    """Contraste contra fuerza bruta en grillas chicas."""
    rng = np.random.default_rng(0)
    for _ in range(6):
        g = 7
        fl = rng.random((g, g)) < 0.3
        q = fl.mean()
        mejor, best = None, -np.inf
        for r0 in range(g):
            for r1 in range(r0, g):
                for c0 in range(g):
                    for c1 in range(c0, g):
                        s = fl[r0:r1 + 1, c0:c1 + 1].sum() - q * (r1 - r0 + 1) * (c1 - c0 + 1)
                        if s > best:
                            best, mejor = s, (r0, r1, c0, c1)
        got = rect_max_exceso(fl, q, max_area=g * g)
        assert abs(got[4] - best) < 1e-9, (got, best, mejor)
    print('  rect_max_exceso: coincide con fuerza bruta', flush=True)


# -------------------------------------------------------------- plausibilidad

def trazas(x):
    """Traza de cada banda: fila promedio ponderada por tinta, por columna."""
    img = np.clip((x[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)
    tinta = 1 - img.mean(axis=2)
    y_lo, y_hi = -0.2, (N_BANDS - 1) * STEP + AMP + 0.2

    def fila(y):
        return (y_hi - y) / (y_hi - y_lo) * PX

    out = []
    for j in range(N_BANDS):
        a = max(int(np.floor(fila(j * STEP + AMP))), 0)
        b = min(int(np.ceil(fila(j * STEP))), PX)
        sub = tinta[a:b, :]
        w = sub.sum(axis=0)
        filas = np.arange(a, b, dtype=np.float64)[:, None]
        cen = (sub * filas).sum(axis=0) / np.maximum(w, 1e-9)
        cen[w < 0.2] = np.nan
        out.append(cen)
    return np.array(out)


def plausibilidad(x):
    """Rugosidad de las trazas y coherencia entre bandas."""
    tr = trazas(x)
    rug, val = [], []
    for c in tr:
        ok = np.isfinite(c)
        d2 = c[2:] - 2 * c[1:-1] + c[:-2]
        m = ok[2:] & ok[1:-1] & ok[:-2]
        val.append(ok.mean())
        if m.sum() > 5:
            rug.append(float(np.abs(d2[m]).mean()))
    cs = []
    for a, b in combinations(range(len(tr)), 2):
        m = np.isfinite(tr[a]) & np.isfinite(tr[b])
        if m.sum() > 20:
            u, v = tr[a][m], tr[b][m]
            if u.std() > 1e-6 and v.std() > 1e-6:
                cs.append(float(np.corrcoef(u, v)[0, 1]))
    return {
        'rugosidad': float(np.mean(rug)) if rug else np.nan,
        'corr_bandas': float(np.mean(cs)) if cs else np.nan,
        'cols_validas': float(np.mean(val)),
    }


# ------------------------------------------------------------------- metodos

@torch.no_grad()
def metodo_base(mg, tokens, dev, g, percentil):
    nll = nll_por_token(mg, tokens, dev, g, rondas=R_LOC)
    sel, _ = seleccionar(nll, percentil)
    m = torch.from_numpy(sel).view(1, -1).to(dev)
    cf = mg.generate_counterfactual(tokens, m, n_steps=N_STEPS)
    return cf, {'enmascarados': int(sel.sum()), 'iteraciones': 1, 'area_rect': None}


@torch.no_grad()
def metodo_azar(mg, tokens, dev, g, n_sel):
    T = tokens.shape[1]
    m = torch.zeros(1, T, dtype=torch.bool, device=dev)
    m[0, torch.randperm(T, device=dev, generator=g)[:n_sel]] = True
    cf = mg.generate_counterfactual(tokens, m, n_steps=N_STEPS)
    return cf, {'enmascarados': n_sel, 'iteraciones': 1, 'area_rect': None}


@torch.no_grad()
def metodo_rect(mg, tokens, dev, g, percentil=0.85):
    """Solo el cambio de region contigua, sin bucle: aisla cual de los dos
    cambios de la variante mejorada aporta y cual estorba."""
    nll = nll_por_token(mg, tokens, dev, g, rondas=R_LOC)
    sel, _ = seleccionar(nll, percentil)
    r0, r1, c0, c1, _, area = rect_max_exceso(sel.reshape(GRID, GRID))
    m = np.zeros((GRID, GRID), bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    cf = mg.generate_counterfactual(
        tokens, torch.from_numpy(m.reshape(1, -1)).to(dev), n_steps=N_STEPS)
    return cf, {'enmascarados': area, 'iteraciones': 1, 'area_rect': float(area)}


@torch.no_grad()
def metodo_mejorado(mg, tokens, dev, g, percentil=0.85):
    """Rectangulo contiguo mas bucle de verificacion contra tau."""
    t = tokens.clone()
    iters, areas, hist = 0, [], []
    for _ in range(MAX_ITER):
        s = float(np.nanmax(nll_por_token(mg, t, dev, g, rondas=R_VER)))
        hist.append(s)
        if s < TAU:
            break
        nll = nll_por_token(mg, t, dev, g, rondas=R_LOC)
        sel, _ = seleccionar(nll, percentil)
        r0, r1, c0, c1, _, area = rect_max_exceso(sel.reshape(GRID, GRID))
        m = np.zeros((GRID, GRID), bool)
        m[r0:r1 + 1, c0:c1 + 1] = True
        t = mg.generate_counterfactual(
            t, torch.from_numpy(m.reshape(1, -1)).to(dev), n_steps=N_STEPS)
        areas.append(area)
        iters += 1
    else:
        hist.append(float(np.nanmax(nll_por_token(mg, t, dev, g, rondas=R_VER))))
    return t, {'enmascarados': int(np.mean(areas)) if areas else 0,
               'iteraciones': iters, 'area_rect': float(np.mean(areas)) if areas else 0,
               'paso_tau': bool(hist[-1] < TAU), 'nll_ini': hist[0],
               'nll_fin': hist[-1], 'historia': hist}


# ---------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('autotest del rectangulo:', flush=True)
    _test_rect()

    zona = zona_objetivo()
    pz = zona_px(*zona)
    area_tot = PX * PX
    area_z = (pz[1] - pz[0]) * (pz[3] - pz[2])
    area_f = area_tot - area_z

    def fuera(a, b):
        tot = float(F.mse_loss(a, b).item())
        return (tot * area_tot - parte(a, b, pz) * area_z) / area_f

    zt = np.zeros((GRID, GRID), bool)
    zt[zona[0]:zona[1] + 1, zona[2]:zona[3] + 1] = True
    zt = zt.reshape(-1)

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MODELOS['aleatoria']),
                                      map_location=dev, strict=False).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imgs = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        imgs[(s, 'limpia')] = render(o, ENC, fig, ax)
        for d in DFS:
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            imgs[(s, d)] = render(p, ENC, fig, ax)
    plt.close(fig)

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    @torch.no_grad()
    def codificar(img):
        x = tf(img).unsqueeze(0).to(dev)
        t = vq.encode_to_indices(x).view(1, -1).long().to(dev)
        return t, reconstruir(vq, t)

    print('\ncurvas limpias: referencia de plausibilidad y control del bucle',
          flush=True)
    limpias, ref = {}, []
    for s in snids:
        t, xr = codificar(imgs[(s, 'limpia')])
        limpias[s] = (t, xr)
        ref.append(plausibilidad(xr))
    ref_rug = np.array([r['rugosidad'] for r in ref])
    ref_cor = np.array([r['corr_bandas'] for r in ref])
    print(f'  columnas validas por traza: {np.mean([r["cols_validas"] for r in ref]):.2f}')
    print(f'  rugosidad de curvas reales limpias: {ref_rug.mean():.4f} +- {ref_rug.std():.4f}')
    print(f'  correlacion entre bandas:            {ref_cor.mean():.4f} +- {ref_cor.std():.4f}',
          flush=True)

    # El bucle no deberia dispararse sobre curvas limpias: si lo hace tanto como
    # sobre las perturbadas, esta midiendo el remuestreo y no la anomalia.
    disparos = 0
    for s in snids:
        _, info = metodo_mejorado(mg, limpias[s][0], dev, g)
        disparos += info['iteraciones'] > 0
    print(f'  el bucle se dispara en {disparos}/{len(snids)} curvas limpias '
          f'({100 * disparos / len(snids):.0f}%; por construccion tau es el p95, '
          f'asi que lo esperado es 5%)', flush=True)

    METODOS = ['base p85', 'base p95', 'azar', 'rect 1 pasa', 'mejorada']
    res = {m: {} for m in METODOS}

    for d in DFS:
        acc = {m: [] for m in METODOS}
        for s in snids:
            tp, xp = codificar(imgs[(s, d)])
            tl, xl = limpias[s]
            base_z, base_f = parte(xp, xl, pz), fuera(xp, xl)

            nll = nll_por_token(mg, tp, dev, g, rondas=R_LOC)
            n85 = int(seleccionar(nll, 0.85)[0].sum())

            salidas = {
                'base p85': metodo_base(mg, tp, dev, g, 0.85),
                'base p95': metodo_base(mg, tp, dev, g, 0.95),
                'azar': metodo_azar(mg, tp, dev, g, n85),
                'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                'mejorada': metodo_mejorado(mg, tp, dev, g),
            }
            for nombre, (cf, info) in salidas.items():
                xc = reconstruir(vq, cf)
                dmap = (xp - xc).abs().sum(dim=1)[0].cpu().numpy()
                masa = float(dmap.sum())
                pl = plausibilidad(xc)
                mz, mf = parte(xc, xl, pz), fuera(xc, xl)
                acc[nombre].append({
                    'repara': 100 * (base_z - mz) / base_z,
                    'dana': 100 * (mf - base_f) / base_f,
                    'mse_zona': mz, 'mse_fuera': mf,
                    'tok_cambiados': float((cf != tp).float().mean().item()),
                    'n_tok_cambiados': int((cf != tp).sum().item()),
                    'loc': (float(dmap[pz[0]:pz[1], pz[2]:pz[3]].sum()) / masa)
                           if masa else 0.0,
                    'rugosidad': pl['rugosidad'], 'corr_bandas': pl['corr_bandas'],
                    'iteraciones': info['iteraciones'],
                    'paso_tau': float(info.get('paso_tau', np.nan)),
                    'area_rect': info['area_rect'] if info['area_rect'] else np.nan,
                })

        print(f'\n{"=" * 88}\ndf={d}')
        print(f'  {"metodo":<11}{"repara":>8}{"dana":>8}{"tokens":>8}'
              f'{"rugosid.":>10}{"corr":>7}{"loc":>7}{"iter":>6}{"pasa tau":>10}')
        for m in METODOS:
            a = acc[m]
            f_ = lambda k: float(np.nanmean([x[k] for x in a]))
            res[m][str(d)] = {k: f_(k) for k in a[0]}
            res[m][str(d)]['rug_z'] = (f_('rugosidad') - ref_rug.mean()) / ref_rug.std()
            pt = f_('paso_tau')
            print(f'  {m:<11}{f_("repara"):7.1f}%{f_("dana"):7.1f}%'
                  f'{f_("n_tok_cambiados"):8.0f}{f_("rugosidad"):10.4f}'
                  f'{f_("corr_bandas"):7.3f}{100 * f_("loc"):6.1f}%'
                  f'{f_("iteraciones"):6.1f}'
                  f'{("-" if np.isnan(pt) else f"{100 * pt:.0f}%"):>10}', flush=True)
        print(f'  {"sin tocar":<11}{0.0:7.1f}%{0.0:7.1f}%{0:8d}'
              f'{ref_rug.mean():10.4f}{ref_cor.mean():7.3f}'
              f'{"":>7}{"":>6}{"":>10}   <- curvas reales limpias')

    salida = {'dfs': DFS, 'tau': TAU, 'R_localizar': R_LOC, 'R_verificar': R_VER,
              'max_iter': MAX_ITER, 'objetos': snids,
              'referencia_limpias': {'rugosidad_media': float(ref_rug.mean()),
                                     'rugosidad_sd': float(ref_rug.std()),
                                     'corr_media': float(ref_cor.mean()),
                                     'corr_sd': float(ref_cor.std())},
              'bucle_en_limpias': disparos / len(snids),
              'resultados': res}
    with open(OUT / 'contrafactual_comparacion.json', 'w') as f:
        json.dump(salida, f, indent=2)

    figuras(res, ref_rug, ref_cor, salida)
    grilla(vq, mg, imgs, limpias, snids, dev, g, pz, codificar)
    print(f'\nGuardado en {OUT}/', flush=True)


def figuras(res, ref_rug, ref_cor, salida):
    METODOS = ['base p85', 'base p95', 'azar', 'rect 1 pasa', 'mejorada']
    COL = {'base p85': '#e45756', 'base p95': '#f2a640', 'azar': '#8856a7',
           'rect 1 pasa': '#31a354', 'mejorada': '#4c78a8'}
    MRK = {'base p85': 'o', 'base p95': 'v', 'azar': 's', 'rect 1 pasa': '^',
           'mejorada': 'D'}

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.7))
    paneles = [
        ('repara', 'repara dentro de la zona [%]', 'Reparacion\nmas alto es mejor'),
        ('dana', 'dana fuera de la zona [%]', 'Dano colateral\nmas bajo es mejor'),
        ('n_tok_cambiados', 'tokens cambiados', 'Parsimonia\nmas bajo es mejor'),
        ('rugosidad', 'rugosidad de las trazas', 'Plausibilidad\ncerca de las reales'),
    ]
    for a, (k, ylab, tit) in zip(axes, paneles):
        for m in METODOS:
            a.plot(DFS, [res[m][str(d)][k] for d in DFS], MRK[m] + '-',
                   color=COL[m], label=m, ms=5)
        if k == 'rugosidad':
            a.axhline(ref_rug.mean(), color='#333333', ls='--', lw=1.3,
                      label='curvas reales limpias')
            a.axhspan(ref_rug.mean() - ref_rug.std(), ref_rug.mean() + ref_rug.std(),
                      color='#333333', alpha=0.10)
        if k in ('repara', 'dana'):
            a.axhline(0, color='#999999', lw=0.8)
        a.set_xscale('log')
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel(ylab)
        a.set_title(tit, fontsize=10)
        a.legend(frameon=False, fontsize=7.5)
    fig.suptitle('Linea base contra variante mejorada, 40 SNIa fuera del entrenamiento '
                 '(codificacion mad300)', fontsize=11)
    fig.text(0.5, 0.005, 'la caida del NLL no se reporta como eficacia: baja tambien '
             'para curvas limpias, asi que mide el remuestreo y no la reparacion',
             ha='center', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.savefig(OUT / 'contrafactual_comparacion.png', dpi=140)
    plt.close(fig)


def grilla(vq, mg, imgs, limpias, snids, dev, g, pz, codificar):
    """Los mismos objetos, linea base contra variante mejorada, lado a lado."""
    d = 100
    muestra = snids[:4]
    cols = ['limpia', f'con pico df={d}', 'base p85', 'mejorada',
            'diff base', 'diff mejorada']
    fig, axes = plt.subplots(len(muestra), 6, figsize=(19, 3.25 * len(muestra)))

    def a_img(t):
        return np.clip((t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)

    for i, s in enumerate(muestra):
        tp, xp = codificar(imgs[(s, d)])
        _, xl = limpias[s]
        cb, _ = metodo_base(mg, tp, dev, g, 0.85)
        cm, info = metodo_mejorado(mg, tp, dev, g)
        xb, xm = reconstruir(vq, cb), reconstruir(vq, cm)
        db = (xp - xb).abs().sum(dim=1)[0].cpu().numpy()
        dm = (xp - xm).abs().sum(dim=1)[0].cpu().numpy()
        paneles = [a_img(xl), a_img(xp), a_img(xb), a_img(xm), db, dm]
        subt = ['', '', f'{int((cb != tp).sum())} tokens',
                f'{int((cm != tp).sum())} tokens, {info["iteraciones"]} iter',
                '', '']
        for j, (im, st) in enumerate(zip(paneles, subt)):
            a = axes[i, j]
            a.imshow(im, cmap=None if im.ndim == 3 else 'magma')
            a.add_patch(mpatches.Rectangle((pz[2], pz[0]), pz[3] - pz[2],
                                           pz[1] - pz[0], fill=False,
                                           edgecolor='#39ff14', lw=1.4))
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=9.5)
            if st:
                a.set_xlabel(st, fontsize=8)
        axes[i, 0].set_ylabel(f'SNID {s}', fontsize=8)
    fig.suptitle('Misma perturbacion, dos protocolos: tokens dispersos (base p85) '
                 'contra region contigua con verificacion (mejorada)\n'
                 'recuadro verde: zona donde inyectamos el pico', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / 'contrafactual_lado_a_lado.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
