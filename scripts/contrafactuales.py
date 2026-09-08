"""Generacion de contrafactuales y su validacion contra la curva limpia conocida.

El protocolo base es el de la metodologia (seccion "Generacion de
contrafactuales"): NLL por token con R=5 rondas, enmascarar los tokens sobre el
percentil 85 de esa misma imagen (~154 de 1024), remuestrear condicionalmente y
localizar con el mapa |x_hat - x_hat'|.

La metodologia evalua por criterios cualitativos. Aca se aprovecha algo que las
anomalias sinteticas permiten y que una anomalia real no: conocemos la curva
limpia. Entonces el contrafactual se genera desde la version PERTURBADA y se
mide si se acerca a la limpia. Todas las distancias son reconstruccion contra
reconstruccion, nunca reconstruccion contra imagen cruda, para no mezclar el
error propio del VQ-VAE con el efecto del contrafactual.

Por que el MaskGIT de mascara aleatoria y no el de ventana: en el cruce 2x2 la
ventana detecta mejor (AUC 0.97 contra 0.90) pero localiza peor (0.62 contra
0.80 de acierto del argmax en la zona objetivo). El contrafactual depende de
acertar QUE tokens reescribir, no solo de saber que algo anda mal, asi que aca
manda la localizacion.

Los 40 SNIa son los mismos del fold de validacion que uso eval_final_mad300.py:
ninguno fue visto en entrenamiento por el VQ-VAE ni por el MaskGIT.

Uso: python scripts/contrafactuales.py
"""
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_enc_cmp import col_flux, fig_params, mejor_ckpt, render, tf
from scripts.eval_final_mad300 import ENC, MODELOS, VQ_CKPT, candidatos
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

OUT = Path('results/enc_cmp_full/eval')
DFS = [10, 50, 100, 300, 500]
GRID, PX = 32, 256
R_NLL, RATIO = 5, 0.5          # el protocolo fija R=5
PERCENTIL, N_STEPS = 0.85, 10  # el protocolo fija p85
PERCENTILES_EXTRA = [0.70, 0.95]
DF_SWEEP = 100
N_OBJ, SEED = 40, 0
# Se decide con maskgit de mascara aleatoria y puntuando aleatorio: mejor
# localizacion, que es lo que importa para elegir que reescribir.
MG_CKPT = MODELOS['aleatoria']


@torch.no_grad()
def nll_por_token(mg, tokens, dev, g, rondas=R_NLL):
    """NLL por token promediado sobre las rondas en que la posicion toco mascara.

    Con R=5 y ratio 0.5 una posicion tiene 0.5^5 = 3% de probabilidad de no caer
    nunca en mascara. Esas quedan NaN para que no entren ni al percentil ni a la
    seleccion, en vez de quedar en cero y hundirse al fondo del ranking.
    """
    T = tokens.shape[1]
    n = int(T * RATIO)
    total = torch.zeros(1, T, device=dev)
    counts = torch.zeros(1, T, device=dev)
    for _ in range(rondas):
        mask = torch.zeros(1, T, dtype=torch.bool, device=dev)
        mask[0, torch.randperm(T, device=dev, generator=g)[:n]] = True
        masked = tokens.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    nll = (total / counts.clamp(min=1.0))[0].cpu().numpy()
    nll[(counts[0] == 0).cpu().numpy()] = np.nan
    return nll


def seleccionar(nll, percentil):
    """Tokens sobre el percentil dado de ESA imagen."""
    umbral = np.nanpercentile(nll, percentil * 100)
    sel = np.nan_to_num(nll, nan=-np.inf) > umbral
    return sel, float(umbral)


@torch.no_grad()
def reconstruir(vq, idx_plano):
    return vq.decode_from_indices(idx_plano.view(1, GRID, GRID))


def mapa_diff(a, b):
    """|x_hat - x_hat'| sumado sobre canales, en (256, 256)."""
    return (a - b).abs().sum(dim=1)[0].cpu().numpy()


def zona_px(r0, r1, c0, c1):
    k = PX // GRID
    return r0 * k, (r1 + 1) * k, c0 * k, (c1 + 1) * k


@torch.no_grad()
def contrafactual(vq, mg, img, dev, g, percentil=PERCENTIL, temperatura=True):
    """Devuelve tokens originales, tokens contrafactuales y todo lo medible."""
    x = tf(img).unsqueeze(0).to(dev)
    idx = vq.encode_to_indices(x)
    tokens = idx.view(1, -1).long().to(dev)

    nll = nll_por_token(mg, tokens, dev, g)
    sel, umbral = seleccionar(nll, percentil)
    mask = torch.from_numpy(sel).view(1, -1).to(dev)

    cf = mg.generate_counterfactual(tokens, mask, n_steps=N_STEPS,
                                    temperatura=temperatura)
    nll_cf = nll_por_token(mg, cf, dev, g)

    return {
        'tokens': tokens, 'cf': cf,
        'nll_max': float(np.nanmax(nll)), 'nll_cf_max': float(np.nanmax(nll_cf)),
        'n_enmascarados': int(sel.sum()), 'umbral': umbral,
        'n_sin_ver': int(np.isnan(nll).sum()),
        'x_rec': reconstruir(vq, tokens), 'x_cf': reconstruir(vq, cf),
        'nll_mapa': nll.reshape(GRID, GRID),
    }


def metricas(res_p, res_l, zona):
    """Compara el contrafactual de la perturbada contra la limpia reconstruida."""
    r0, r1, c0, c1 = zona
    p0, p1, q0, q1 = zona_px(r0, r1, c0, c1)

    t_p = res_p['tokens'][0].cpu().numpy()
    t_c = res_p['cf'][0].cpu().numpy()
    t_l = res_l['tokens'][0].cpu().numpy()

    d = mapa_diff(res_p['x_rec'], res_p['x_cf'])
    masa = float(d.sum())
    dentro = float(d[p0:p1, q0:q1].sum())

    m = {
        'nll_pert': res_p['nll_max'], 'nll_cf': res_p['nll_cf_max'],
        'nll_limpia': res_l['nll_max'],
        'tok_cambiados': float((t_c != t_p).mean()),
        'tok_dist_pert_limpia': float((t_p != t_l).mean()),
        'tok_dist_cf_limpia': float((t_c != t_l).mean()),
        'px_frac_cambiada': float((d > 0.05).mean()),
        'loc_masa_en_zona': dentro / masa if masa > 0 else 0.0,
        'n_enmascarados': res_p['n_enmascarados'],
    }
    mse = lambda a, b: float(F.mse_loss(a, b).item())
    m['mse_pert_limpia'] = mse(res_p['x_rec'], res_l['x_rec'])
    m['mse_cf_limpia'] = mse(res_p['x_cf'], res_l['x_rec'])
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zona = zona_objetivo()
    r0, r1, c0, c1 = zona
    frac_zona = ((r1 - r0 + 1) * (c1 - c0 + 1)) / (GRID * GRID)
    print(f'dispositivo: {dev}', flush=True)
    print(f'zona objetivo: filas {r0}-{r1}, cols {c0}-{c1} '
          f'({100 * frac_zona:.1f}% de la imagen: ese es el nivel de azar)', flush=True)

    curvas = candidatos(N_OBJ)
    snids = sorted(curvas)
    print(f'objetos: {len(snids)} (ninguno visto en entrenamiento)', flush=True)

    vq = VQVAE.load_from_checkpoint(mejor_ckpt(VQ_CKPT), map_location=dev).eval().to(dev)
    mg = MaskGIT.load_from_checkpoint(mejor_ckpt(MG_CKPT), map_location=dev,
                                      strict=False).eval().to(dev)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    print('renderizando...', flush=True)
    imgs = {}
    for s in snids:
        o, idx_cola, sc = curvas[s]
        imgs[(s, 'limpia')] = render(o, ENC, fig, ax)
        for d in DFS:
            p = o.copy()
            p.loc[idx_cola, col_flux] = p.loc[idx_cola, col_flux] + d * sc
            imgs[(s, d)] = render(p, ENC, fig, ax)
    plt.close(fig)
    print(f'imagenes: {len(imgs)}', flush=True)

    salida = {'dfs': DFS, 'objetos': snids, 'zona_token': list(zona),
              'frac_zona': frac_zona, 'protocolo': {
                  'R_nll': R_NLL, 'ratio_mascara': RATIO, 'percentil': PERCENTIL,
                  'n_steps': N_STEPS, 'maskgit': 'mascara aleatoria',
              }, 'variantes': {}}

    for temperatura in (True, False):
        etiqueta = 'con recocido' if temperatura else 'sin recocido'
        print(f'\n{"=" * 72}\nVARIANTE: {etiqueta} de temperatura', flush=True)
        torch.manual_seed(SEED)
        g = torch.Generator(device=dev)
        g.manual_seed(SEED)

        limpias = {}
        for s in snids:
            limpias[s] = contrafactual(vq, mg, imgs[(s, 'limpia')], dev, g,
                                       temperatura=temperatura)

        # Baseline: contrafactual de una curva limpia. Como el percentil es
        # relativo a la imagen, siempre reescribe ~154 tokens aunque no haya nada
        # anomalo; lo que se mide es si esos cambios se concentran o se dispersan.
        base = []
        for s in snids:
            rl = limpias[s]
            d = mapa_diff(rl['x_rec'], rl['x_cf'])
            p0, p1, q0, q1 = zona_px(*zona)
            masa = float(d.sum())
            t_l = rl['tokens'][0].cpu().numpy()
            t_c = rl['cf'][0].cpu().numpy()
            base.append({
                'nll_limpia': rl['nll_max'], 'nll_cf': rl['nll_cf_max'],
                'tok_cambiados': float((t_c != t_l).mean()),
                'loc_masa_en_zona': (float(d[p0:p1, q0:q1].sum()) / masa) if masa else 0.0,
                'px_frac_cambiada': float((d > 0.05).mean()),
            })

        def prom(lst, k):
            v = [x[k] for x in lst if np.isfinite(x[k])]
            return float(np.mean(v)) if v else float('nan')

        print(f'\n[baseline: contrafactual de la curva LIMPIA]  n={len(base)}')
        print(f'  NLL max {prom(base, "nll_limpia"):.2f} -> {prom(base, "nll_cf"):.2f}')
        print(f'  tokens cambiados {100 * prom(base, "tok_cambiados"):.1f}%   '
              f'masa en la zona {100 * prom(base, "loc_masa_en_zona"):.1f}% '
              f'(azar {100 * frac_zona:.1f}%)')

        por_df = {}
        for d in DFS:
            ms = []
            for s in snids:
                rp = contrafactual(vq, mg, imgs[(s, d)], dev, g,
                                   temperatura=temperatura)
                ms.append(metricas(rp, limpias[s], zona))
            por_df[d] = {k: prom(ms, k) for k in ms[0]}
            por_df[d]['detalle'] = ms
            x = por_df[d]
            print(f'\n[df={d}]  n={len(ms)}')
            print(f'  eficacia    NLL max: perturbada {x["nll_pert"]:.2f} -> '
                  f'contrafactual {x["nll_cf"]:.2f}   (limpia {x["nll_limpia"]:.2f})')
            print(f'  fidelidad   MSE a la limpia: perturbada {x["mse_pert_limpia"]:.5f} '
                  f'-> contrafactual {x["mse_cf_limpia"]:.5f}')
            print(f'              tokens != limpia: perturbada '
                  f'{100 * x["tok_dist_pert_limpia"]:.1f}% -> contrafactual '
                  f'{100 * x["tok_dist_cf_limpia"]:.1f}%')
            print(f'  parsimonia  tokens cambiados {100 * x["tok_cambiados"]:.1f}%   '
                  f'pixeles {100 * x["px_frac_cambiada"]:.1f}%')
            print(f'  localizac.  masa en la zona {100 * x["loc_masa_en_zona"]:.1f}% '
                  f'(azar {100 * frac_zona:.1f}%)', flush=True)

        salida['variantes'][etiqueta] = {
            'baseline_limpia': {k: prom(base, k) for k in base[0]},
            'por_df': {str(d): {k: v for k, v in por_df[d].items() if k != 'detalle'}
                       for d in DFS},
            'detalle_por_df': {str(d): por_df[d]['detalle'] for d in DFS},
        }
        if temperatura:
            guardar_figuras(vq, mg, imgs, limpias, snids, dev, zona, por_df,
                            frac_zona, salida)

    # Barrido de percentil, como analisis adicional sobre el caso base.
    print(f'\n{"=" * 72}\nBARRIDO DE PERCENTIL en df={DF_SWEEP}', flush=True)
    torch.manual_seed(SEED)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    limpias = {s: contrafactual(vq, mg, imgs[(s, 'limpia')], dev, g) for s in snids}
    barrido = {}
    for pc in sorted([PERCENTIL] + PERCENTILES_EXTRA):
        ms = []
        for s in snids:
            rp = contrafactual(vq, mg, imgs[(s, DF_SWEEP)], dev, g, percentil=pc)
            ms.append(metricas(rp, limpias[s], zona))
        barrido[str(pc)] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        x = barrido[str(pc)]
        marca = '  <- caso base' if pc == PERCENTIL else ''
        print(f'  p{int(pc * 100)}: {x["n_enmascarados"]:.0f} tokens   '
              f'NLL {x["nll_pert"]:.2f}->{x["nll_cf"]:.2f}   '
              f'MSE {x["mse_pert_limpia"]:.5f}->{x["mse_cf_limpia"]:.5f}   '
              f'cambia {100 * x["tok_cambiados"]:.1f}% tokens{marca}', flush=True)
    salida['barrido_percentil'] = {'df': DF_SWEEP, 'resultados': barrido}

    with open(OUT / 'contrafactuales.json', 'w') as f:
        json.dump(salida, f, indent=2)
    print(f'\nGuardado en {OUT}/contrafactuales.json', flush=True)


def guardar_figuras(vq, mg, imgs, limpias, snids, dev, zona, por_df, frac_zona,
                    salida):
    """Grilla cualitativa y curvas agregadas."""
    r0, r1, c0, c1 = zona
    p0, p1, q0, q1 = zona_px(r0, r1, c0, c1)
    df_fig = 100
    muestra = snids[:4]

    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    torch.manual_seed(SEED)

    def a_img(t):
        return np.clip((t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)

    fig, axes = plt.subplots(len(muestra), 4, figsize=(13.5, 3.3 * len(muestra)))
    for fila, s in enumerate(muestra):
        rp = contrafactual(vq, mg, imgs[(s, df_fig)], dev, g)
        rl = limpias[s]
        d = mapa_diff(rp['x_rec'], rp['x_cf'])
        paneles = [
            (a_img(rl['x_rec']), f'limpia (reconstruida)\nNLL max {rl["nll_max"]:.1f}'),
            (a_img(rp['x_rec']), f'con pico df={df_fig}\nNLL max {rp["nll_max"]:.1f}'),
            (a_img(rp['x_cf']), f'contrafactual\nNLL max {rp["nll_cf_max"]:.1f}'),
            (d, r'$|\hat{x} - \hat{x}\,^\prime|$'),
        ]
        for col, (im, tit) in enumerate(paneles):
            a = axes[fila, col]
            a.imshow(im, cmap=None if im.ndim == 3 else 'magma')
            a.add_patch(mpatches.Rectangle((q0, p0), q1 - q0, p1 - p0, fill=False,
                                           edgecolor='#39ff14', lw=1.4))
            a.set_xticks([])
            a.set_yticks([])
            if fila == 0:
                a.set_title(tit, fontsize=9)
            elif col == 3:
                a.set_title(tit, fontsize=8)
            else:
                a.set_title(tit.split('\n')[-1], fontsize=8)
        axes[fila, 0].set_ylabel(f'SNID {s}', fontsize=8)
    fig.suptitle('Contrafactuales: el prior reescribe los tokens sobre el percentil 85 '
                 f'del NLL (df={df_fig})\nrecuadro verde: zona donde inyectamos el pico',
                 fontsize=11)
    fig.text(0.5, 0.005, '40 SNIa del fold de validacion: ningun modelo los vio en '
             'entrenamiento', ha='center', fontsize=8, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / 'contrafactual_ejemplos.png', dpi=140)
    plt.close(fig)

    b = salida['variantes']['con recocido']['baseline_limpia'] \
        if 'con recocido' in salida['variantes'] else None
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    dfs = DFS
    a = axes[0]
    a.plot(dfs, [por_df[d]['nll_pert'] for d in dfs], 'o-', color='#e45756',
           label='perturbada')
    a.plot(dfs, [por_df[d]['nll_cf'] for d in dfs], 'o-', color='#4c78a8',
           label='contrafactual')
    a.plot(dfs, [por_df[d]['nll_limpia'] for d in dfs], '--', color='#444444',
           label='limpia (referencia)')
    a.set_xscale('log')
    a.set_xlabel(r'$df$  [veces MAD$_r$]')
    a.set_ylabel('NLL maximo')
    a.set_title('Eficacia: baja el puntaje de anomalia?', fontsize=10)
    a.legend(frameon=False, fontsize=8)

    a = axes[1]
    a.plot(dfs, [por_df[d]['mse_pert_limpia'] for d in dfs], 'o-', color='#e45756',
           label='perturbada contra limpia')
    a.plot(dfs, [por_df[d]['mse_cf_limpia'] for d in dfs], 'o-', color='#4c78a8',
           label='contrafactual contra limpia')
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'$df$  [veces MAD$_r$]')
    a.set_ylabel('MSE entre reconstrucciones')
    a.set_title('Fidelidad: se acerca a la curva limpia?', fontsize=10)
    a.legend(frameon=False, fontsize=8)

    a = axes[2]
    a.plot(dfs, [100 * por_df[d]['loc_masa_en_zona'] for d in dfs], 'o-',
           color='#31a354', label='masa del cambio en la zona')
    a.axhline(100 * frac_zona, color='crimson', ls=':', lw=1.3,
              label=f'azar ({100 * frac_zona:.1f}%)')
    if b:
        a.axhline(100 * b['loc_masa_en_zona'], color='#888888', ls='--', lw=1.3,
                  label='baseline: contrafactual de la limpia')
    a.plot(dfs, [100 * por_df[d]['tok_cambiados'] for d in dfs], 's--',
           color='#8856a7', label='tokens cambiados (parsimonia)')
    a.set_xscale('log')
    a.set_xlabel(r'$df$  [veces MAD$_r$]')
    a.set_ylabel('%')
    a.set_title('Localizacion y parsimonia', fontsize=10)
    a.legend(frameon=False, fontsize=7.5)

    fig.suptitle('Contrafactuales sobre 40 SNIa fuera del entrenamiento, '
                 'codificacion mad300', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'contrafactual_metricas.png', dpi=140)
    plt.close(fig)
    print('  figuras guardadas', flush=True)


if __name__ == '__main__':
    main()
