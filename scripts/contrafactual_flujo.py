"""Evaluar los contrafactuales en espacio de flujo, no de pixeles.

MOTIVO. Toda la evaluacion anterior mide MSE sobre los 65536 pixeles de un
grafico de matplotlib. La zona anomala es el 4.9% de esa area, pero buena parte
del 95.1% restante es fondo blanco, o sea que no es dato. Contarlo como
informacion penaliza al contrafactual por marcas que caen en el vacio y puede
ser el origen del neto negativo que medimos en todos los dt. TimeVQVAE-AD
(arXiv 2311.12550) usa la misma arquitectura conceptual pero sobre un
espectrograma STFT, donde cada celda del latente es una cantidad fisica y ese
problema no existe.

Se evaluan los MISMOS contrafactuales, sin tocar el metodo de generacion, en
tres espacios:

  1. pixeles global    lo que ya teniamos, como referencia
  2. pixeles de traza  restringido a donde hay dato, con la mascara de no-fondo
                       de la imagen limpia dilatada
  3. flujo (FLUXCAL)   invirtiendo la imagen a curva de luz y comparando en las
                       epocas de observacion reales

VALIDACION DEL INVERSOR. La extraccion desde pixeles tiene error de
cuantizacion, asi que antes de concluir nada se invierte la imagen de la curva
LIMPIA y se compara contra la curva verdadera del parquet. Ese error de ida y
vuelta es el piso de ruido del analisis. Se reportan dos pisos: el del render
crudo (solo cuantizacion) y el de la reconstruccion del VQ-VAE, que es el que
de verdad aplica porque todas nuestras comparaciones son reconstruccion contra
reconstruccion. Si el piso resulta del orden del efecto buscado, este analisis
no concluye y hay que decirlo.

ESTADISTICOS PARA INVERTIR. Cada imagen se invierte con los estadisticos que se
usaron para dibujarla, que es el inverso matematicamente correcto: la limpia con
la mediana y el MAD de la curva limpia, y la perturbada y su contrafactual con
los de la curva perturbada, porque el contrafactual vive en el marco de la
imagen de la que salio. Como control de sensibilidad se reporta tambien la
variante que invierte todo con los estadisticos limpios.

Uso: python scripts/contrafactual_flujo.py
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

from scripts.eval_enc_cmp import (
    AMP, N_BANDS, STEP, all_bands, col_band, col_flux, col_mjd, fig_params,
    mejor_ckpt, render, tf,
)
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.contrafactuales import PX, reconstruir, zona_px
from scripts.contrafactual_comparacion import metodo_base, metodo_azar, metodo_rect
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
U_MAX = 300.0
N_OBJ, SEED = 40, 0
DILATAR = 2
METODOS = ['base p95', 'rect 1 pasa', 'azar']

Y_LO = -0.2
Y_HI = (N_BANDS - 1) * STEP + AMP + 0.2


def col_de(t_norm):
    """Coordenada x normalizada a columna de pixel, igual que el renderizador."""
    return (np.asarray(t_norm) + 0.05) / 1.10 * PX


def fila_a_y(fila):
    return Y_HI - np.asarray(fila) / PX * (Y_HI - Y_LO)


def a_np(x):
    """Tensor (1,3,256,256) en [-1,1] a arreglo HxWx3 en [0,1]."""
    return np.clip((x[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)


def trazas_filas(x):
    """Fila del centroide de tinta por columna, para cada banda."""
    img = a_np(x)
    tinta = 1 - img.mean(axis=2)
    out = np.full((N_BANDS, PX), np.nan)
    for j in range(N_BANDS):
        a = max(int(np.floor((Y_HI - (j * STEP + AMP)) / (Y_HI - Y_LO) * PX)), 0)
        b = min(int(np.ceil((Y_HI - j * STEP) / (Y_HI - Y_LO) * PX)), PX)
        sub = tinta[a:b, :]
        w = sub.sum(axis=0)
        filas = np.arange(a, b, dtype=np.float64)[:, None]
        cen = (sub * filas).sum(axis=0) / np.maximum(w, 1e-9)
        cen[w < 0.2] = np.nan
        out[j] = cen
    return out


def invertir(x, med, escala, obj, t0, rango):
    """Imagen -> flujo FLUXCAL en las epocas observadas.

    Deshace: fila -> y -> u dentro del carril de la banda -> asinh inverso.
    """
    tr = trazas_filas(x)
    rec = np.full(len(obj), np.nan)
    for band_key, j in all_bands.items():
        m = (obj[col_band] == band_key).values
        if not m.any():
            continue
        cols = col_de((obj.loc[m, col_mjd].values - t0) / rango)
        c0 = np.clip(np.round(cols).astype(int), 0, PX - 1)
        filas = tr[j][c0]
        y = fila_a_y(filas)
        u = np.clip((y - j * STEP) / AMP, 0.0, 1.0)
        us = 2 * u - 1
        rec[m] = med + escala * np.sinh(us * np.arcsinh(U_MAX))
    return rec


def mascara_trazas(x_limpia, dilatar=DILATAR):
    """Pixeles con dato en la imagen limpia, dilatados un poco.

    Se construye desde el tensor ya redimensionado a 256x256, no desde el PNG
    original, que sale en el tamano de la figura de matplotlib.
    """
    tinta = 1 - a_np(x_limpia).mean(axis=2)
    m = tinta > 0.05
    m[:5, :] = m[-5:, :] = m[:, :5] = m[:, -5:] = False   # borde negro del render
    if dilatar:
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        k = 2 * dilatar + 1
        m = (F.max_pool2d(t, k, stride=1, padding=dilatar)[0, 0].numpy() > 0.5)
    return m


def mse_mascara(a, b, m):
    d = ((a - b) ** 2).mean(dim=1)[0].cpu().numpy()
    return float(d[m].mean()) if m.any() else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    pz = zona_px(*zona)

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
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

    def stats(vals):
        v = np.asarray(vals, dtype='float64')
        m = np.median(v)
        s = np.median(np.abs(v - m))
        return float(m), float(s if np.isfinite(s) and s > 0 else 1.0)

    # ------------------------------------------- cuanto de la imagen es fondo
    fr = []
    for s in snids:
        o, _, _ = curvas[s]
        fr.append(mascara_trazas(tf(render(o, ENC, fig, ax)).unsqueeze(0)).mean())
    print(f'\nfraccion de la imagen que es dato (mascara dilatada {DILATAR}px): '
          f'{100 * np.mean(fr):.1f}%   -> el {100 * (1 - np.mean(fr)):.1f}% es fondo',
          flush=True)

    # ------------------------------------------------- validacion del inversor
    print(f'\n{"=" * 80}\nVALIDACION DEL INVERSOR sobre las curvas limpias', flush=True)
    err_raw, err_rec, rel_raw, rel_rec, faltan = [], [], [], [], []
    escalas = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        t0 = float(o[col_mjd].min())
        rango = max(float(o[col_mjd].max()) - t0, 1.0)
        med, esc = stats(o[col_flux].values)
        escalas[s] = (med, esc, t0, rango)
        img = render(o, ENC, fig, ax)
        xraw = tf(img).unsqueeze(0).to(dev)
        _, xrec = codificar(img)
        ftrue = o[col_flux].values.astype('float64')
        disp = float(np.median(np.abs(ftrue - np.median(ftrue))))
        for xx, ea, ra in ((xraw, err_raw, rel_raw), (xrec, err_rec, rel_rec)):
            f = invertir(xx, med, esc, o, t0, rango)
            ok = np.isfinite(f)
            ea.append(float(np.sqrt(np.mean((f[ok] - ftrue[ok]) ** 2))))
            ra.append(float(np.median(np.abs(f[ok] - ftrue[ok])) / max(disp, 1e-9)))
        faltan.append(1 - np.isfinite(invertir(xrec, med, esc, o, t0, rango)).mean())

    print(f'  epocas sin traza recuperable: {100 * np.mean(faltan):.1f}%')
    print(f'  render crudo         RMSE {np.mean(err_raw):8.3f} FLUXCAL   '
          f'error mediano = {np.median(rel_raw):.2f} MAD del objeto')
    print(f'  reconstruccion VQVAE RMSE {np.mean(err_rec):8.3f} FLUXCAL   '
          f'error mediano = {np.median(rel_rec):.2f} MAD del objeto', flush=True)
    piso = float(np.mean(err_rec))

    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)

    res = {m: {} for m in METODOS}
    res['perturbada'] = {}
    print(f'\n{"=" * 80}\nTRES ESPACIOS DE EVALUACION', flush=True)

    for d in DFS:
        acc = {m: [] for m in METODOS}
        acc['perturbada'] = []
        for s in snids:
            o, idx_cola, sc = curvas[s]
            med_l, esc_l, t0, rango = escalas[s]
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            med_p, esc_p = stats(p[col_flux].values)

            img_l = render(o, ENC, fig, ax)
            img_p = render(p, ENC, fig, ax)
            mtr = mascara_trazas(tf(img_l).unsqueeze(0))
            mzona = np.zeros((PX, PX), bool)
            mzona[pz[0]:pz[1], pz[2]:pz[3]] = True

            _, xl = codificar(img_l)
            tp, xp = codificar(img_p)

            ftrue = o[col_flux].values.astype('float64')
            en_zona = np.zeros(len(o), bool)
            en_zona[o.index.get_indexer(idx_cola)] = True

            f_pert = invertir(xp, med_p, esc_p, o, t0, rango)
            salidas = {'base p95': metodo_base(mg, tp, dev, g, 0.95),
                       'rect 1 pasa': metodo_rect(mg, tp, dev, g),
                       'azar': metodo_azar(mg, tp, dev, g, 148)}

            def paquete(x, f):
                ok = np.isfinite(f)
                dz, df_ = ok & en_zona, ok & ~en_zona
                return {
                    'px_glob': float(F.mse_loss(x, xl).item()),
                    'px_traza_z': mse_mascara(x, xl, mtr & mzona),
                    'px_traza_f': mse_mascara(x, xl, mtr & ~mzona),
                    'px_traza': mse_mascara(x, xl, mtr),
                    'fuera_mascara': float(
                        (((x - xp).abs().sum(dim=1)[0].cpu().numpy())[~mtr].sum())
                        / max((x - xp).abs().sum().item(), 1e-9)),
                    'fx_z': float(np.sqrt(np.mean((f[dz] - ftrue[dz]) ** 2)))
                            if dz.any() else np.nan,
                    'fx_f': float(np.sqrt(np.mean((f[df_] - ftrue[df_]) ** 2)))
                            if df_.any() else np.nan,
                    'n_z': int(dz.sum()), 'n_f': int(df_.sum()),
                }

            base = paquete(xp, f_pert)
            acc['perturbada'].append(base)
            for m, (cf, _) in salidas.items():
                xc = reconstruir(vq, cf)
                fc = invertir(xc, med_p, esc_p, o, t0, rango)
                q = paquete(xc, fc)
                q['g_px_traza'] = (base['px_traza_z'] - q['px_traza_z']) * \
                    float((mtr & mzona).sum())
                q['p_px_traza'] = (q['px_traza_f'] - base['px_traza_f']) * \
                    float((mtr & ~mzona).sum())
                q['g_fx'] = (base['fx_z'] ** 2 - q['fx_z'] ** 2) * base['n_z']
                q['p_fx'] = (q['fx_f'] ** 2 - base['fx_f'] ** 2) * base['n_f']
                acc[m].append(q)

        prom = lambda a, k: float(np.nanmean([x[k] for x in a]))
        b = acc['perturbada']
        res['perturbada'][str(d)] = {k: prom(b, k) for k in b[0]}
        print(f'\n{"=" * 80}\ndf={d}    perturbada: residuo en flujo '
              f'{prom(b, "fx_z"):.2f} en la zona ({prom(b, "n_z"):.1f} epocas), '
              f'{prom(b, "fx_f"):.2f} fuera ({prom(b, "n_f"):.1f} epocas)')
        print(f'  {"metodo":<13}{"px global":>11}{"px traza z":>12}{"px traza f":>12}'
              f'{"flujo z":>10}{"flujo f":>10}{"neto px":>10}{"neto flujo":>12}')
        print(f'  {"perturbada":<13}{prom(b, "px_glob"):11.5f}'
              f'{prom(b, "px_traza_z"):12.5f}{prom(b, "px_traza_f"):12.5f}'
              f'{prom(b, "fx_z"):10.2f}{prom(b, "fx_f"):10.2f}')
        for m in METODOS:
            a = acc[m]
            res[m][str(d)] = {k: prom(a, k) for k in a[0]}
            res[m][str(d)]['neto_px_traza'] = prom(a, 'g_px_traza') - prom(a, 'p_px_traza')
            res[m][str(d)]['neto_fx'] = prom(a, 'g_fx') - prom(a, 'p_fx')
            res[m][str(d)]['frac_neto_fx_pos'] = float(
                np.mean([(x['g_fx'] - x['p_fx']) > 0 for x in a]))
            print(f'  {m:<13}{prom(a, "px_glob"):11.5f}{prom(a, "px_traza_z"):12.5f}'
                  f'{prom(a, "px_traza_f"):12.5f}{prom(a, "fx_z"):10.2f}'
                  f'{prom(a, "fx_f"):10.2f}'
                  f'{res[m][str(d)]["neto_px_traza"]:10.1f}'
                  f'{res[m][str(d)]["neto_fx"]:12.1f}', flush=True)
        print(f'  (piso del inversor: RMSE {piso:.2f} FLUXCAL; masa del cambio que '
              f'cae en el fondo: {100 * prom(acc["base p95"], "fuera_mascara"):.0f}% '
              f'para p95)')

    plt.close(fig)
    with open(OUT / 'contrafactual_flujo.json', 'w') as f:
        json.dump({'dfs': DFS, 'piso_inversor_rmse': piso,
                   'piso_render_crudo_rmse': float(np.mean(err_raw)),
                   'frac_dato': float(np.mean(fr)), 'resultados': res}, f, indent=2)

    figura_curvas(vq, mg, curvas, snids, dev, g, codificar, escalas, stats)
    figura_neto(res)
    print(f'\nGuardado en {OUT}/', flush=True)


def figura_curvas(vq, mg, curvas, snids, dev, g, codificar, escalas, stats):
    """Curvas de luz, no imagenes: limpia, perturbada y contrafactual."""
    d = 300
    muestra = snids[:3]
    fig, axes = plt.subplots(len(muestra), 1, figsize=(11, 3.6 * len(muestra)))
    if len(muestra) == 1:
        axes = [axes]
    fr = plt.figure(figsize=(fig_params['figsize']))
    ar = fr.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for a, s in zip(axes, muestra):
        o, idx_cola, sc = curvas[s]
        med_l, esc_l, t0, rango = escalas[s]
        p = o.copy()
        p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
        med_p, esc_p = stats(p[col_flux].values)
        tp, _ = codificar(render(p, ENC, fig=fr, ax=ar))
        cf, _ = metodo_base(mg, tp, dev, g, 0.95)
        xc = reconstruir(vq, cf)
        fc = invertir(xc, med_p, esc_p, o, t0, rango)
        t = (o[col_mjd].values - t0)
        for band_key, j in all_bands.items():
            m = (o[col_band] == band_key).values
            if not m.any():
                continue
            c = fig_params['colors'][j]
            a.plot(t[m], o[col_flux].values[m], '-', color=c, lw=1.0, alpha=0.85)
            a.plot(t[m], o[col_flux].values[m], 'o', color=c, ms=3,
                   label=band_key if j == 0 else None)
            pm = p[col_flux].values[m]
            a.plot(t[m], pm, ':', color=c, lw=1.1, alpha=0.7)
            a.plot(t[m], fc[m], 'x--', color=c, ms=5, lw=1.0, alpha=0.9)
        a.set_ylabel('FLUXCAL')
        a.set_title(f'SNID {s}   (df={d})', fontsize=10)
        a.set_yscale('symlog')
    axes[-1].set_xlabel('dias desde la primera observacion')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], color='#444', marker='o', ls='-', label='limpia (verdad)'),
        Line2D([], [], color='#444', ls=':', label='perturbada'),
        Line2D([], [], color='#444', marker='x', ls='--',
               label='contrafactual (invertido a flujo)')],
        frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Contrafactuales como curvas de luz, no como imagenes  '
                 '(base p95, codificacion mad300)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / 'contrafactual_curvas.png', dpi=140)
    plt.close(fig)
    plt.close(fr)


def figura_neto(res):
    COL = {'base p95': '#f2a640', 'rect 1 pasa': '#31a354', 'azar': '#8856a7'}
    MRK = {'base p95': 'v', 'rect 1 pasa': '^', 'azar': 's'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for a, (k, tit) in zip(axes, [
            ('neto_px_traza', 'Espacio 2: pixeles de traza'),
            ('neto_fx', 'Espacio 3: flujo (FLUXCAL)')]):
        for m in METODOS:
            a.plot(DFS, [res[m][str(d)][k] for d in DFS], MRK[m] + '-',
                   color=COL[m], label=m, ms=6)
        a.axhline(0, color='#333333', lw=1.2)
        a.set_xscale('log')
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel('neto = ganancia en la zona - perdida fuera')
        a.set_title(tit + '\nsobre cero, repara mas de lo que rompe', fontsize=10)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle('El signo del neto segun el espacio en que se mide', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'contrafactual_flujo.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
