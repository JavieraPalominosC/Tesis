"""Evaluacion pareada de las tres codificaciones de flujo.

Cada rama se evalua de punta a punta con SUS propios modelos: se renderizan las
curvas perturbadas con su codificacion, se tokenizan con su VQ-VAE y se puntuan
con su MaskGIT. Asi ninguna queda fuera de distribucion.

Sobre la comparabilidad entre ramas: el NLL absoluto NO es comparable, porque
cada rama tiene su propio vocabulario de tokens y su propia dificultad de
prediccion. Una rama puede tener NLL mas alto en todo simplemente porque sus
tokens son mas dificiles. Por eso todas las metricas de aca son tasas o rangos,
que si se pueden comparar:

  recall    fraccion de perturbadas sobre tau, con tau = P95 de los controles
            de esa misma rama
  AUC       area bajo ROC entre controles y perturbadas de esa rama
  pareada   fraccion de objetos cuyo NLL perturbado supera al de SU control
  monotonia Spearman entre df y NLL por objeto, mediana sobre objetos

La monotonia es la que responde la pregunta del profe: si un pico mas alto es
mas anomalo, el NLL deberia subir con df, no estancarse.

Uso: python scripts/eval_enc_cmp.py [N_OBJETOS]
"""
import sys
sys.path.insert(0, '.')
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import spearmanr

from scripts.enc_flujo import codificar
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

RAMAS = ('actual', 'mad100', 'mad300')
DFS = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500]
# Este objeto quedo en el subconjunto de entrenamiento, asi que su control tiene
# NLL artificialmente bajo y inflaria la deteccion pareada.
EXCLUIR = {'50539358'}

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
BASE_FILE = 'lc_SNIa-SALT2.parquet'
MANIFEST = 'data/synthetic_mad_tail_manifest.json'
OUT = Path('results/enc_cmp/eval')

TARGET_BAND, MIN_TAIL, TAIL_FRAC = 'r', 6, 0.30
STEP, AMP = 2.0, 1.8
GRID, N_ROUNDS, MASK_RATIO, CTRL_Q = 32, 10, 0.5, 0.95

cfg_run = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
fig_params = cfg_run['ft_classification']['imgs_params']['fig_params']
ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dc = ds['dict_columns']
all_bands = ds['all_bands']
col_mjd, col_flux, col_band = dc['mjd'], dc['flux'], dc['band']
N_BANDS = len(all_bands)

tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def mad(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def mejor_ckpt(patron):
    import glob
    import re
    c = glob.glob(patron, recursive=True)
    c = [p for p in c if 'last' not in os.path.basename(p)]
    if not c:
        raise SystemExit(f'sin checkpoints: {patron}')

    def loss(p):
        m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
        return float(m.group(1)) if m else float('inf')

    return min(c, key=loss)


def render(obj_df, enc, fig, ax):
    """Dibuja la curva con la codificacion dada y devuelve la imagen."""
    obj_flux = obj_df[col_flux].astype('float64').values
    t0, t1 = float(obj_df[col_mjd].min()), float(obj_df[col_mjd].max())
    rango = t1 - t0 if t1 > t0 else 1.0

    ax.clear()
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        y = codificar(bd[col_flux].values, obj_flux, enc) * AMP + j * STEP
        ax.plot((bd[col_mjd] - t0) / rango, y, 'o-', color=fig_params['colors'][j],
                markersize=2, linewidth=1.0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS - 1) * STEP + AMP + 0.2])
    ax.axis('off')
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                             facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').copy()
    buf.close()
    rect.remove()
    return img


@torch.no_grad()
def nll_max(vq, mg, img, dev):
    x = tf(img).unsqueeze(0).to(dev)
    z = vq.encoder(x)
    _, _, idx, *_ = vq.codebook(z)
    tokens = idx.view(1, -1).long().to(dev)

    T_ = tokens.shape[1]
    total = torch.zeros(1, T_, device=dev)
    counts = torch.zeros(1, T_, device=dev)
    n_mask = int(T_ * MASK_RATIO)
    for _ in range(N_ROUNDS):
        mask = torch.zeros(1, T_, dtype=torch.bool, device=dev)
        mask[0, torch.randperm(T_, device=dev)[:n_mask]] = True
        masked = tokens.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()
    nll = (total / counts.clamp(min=1.0))[0].cpu().numpy()
    return float(nll.max()), nll.reshape(GRID, GRID)


def auc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    o = np.argsort(-s)
    y = y[o]
    tpr = np.r_[0.0, np.cumsum(y) / max(y.sum(), 1.0)]
    fpr = np.r_[0.0, np.cumsum(1 - y) / max((1 - y).sum(), 1.0)]
    trapz = getattr(np, 'trapezoid', np.trapz)
    return float(trapz(tpr, fpr))


def objetos():
    """Los mismos SNIa del experimento anterior, menos el que quedo en train."""
    with open(MANIFEST) as f:
        man = json.load(f)
    snids = [r['snid'] for r in man if r.get('tag') == 'control']
    snids = [s for s in dict.fromkeys(snids) if s not in EXCLUIR]
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    return snids[:limite] if limite else snids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    snids = objetos()
    print(f'objetos: {len(snids)} (excluidos {sorted(EXCLUIR)})', flush=True)

    df_all = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df_all['SNID'] = df_all['SNID'].astype(str)
    df_all = df_all[df_all['SNID'].isin(snids)]

    curvas = {}
    for snid in snids:
        o = df_all[df_all['SNID'] == snid].copy()
        o[col_flux] = o[col_flux].astype('float64')
        sc = mad(o.loc[o[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(sc) or sc <= 0:
            continue
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        t0, t1 = float(o[col_mjd].min()), float(o[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])
        curvas[snid] = (o, idx, sc)
    print(f'curvas utilizables: {len(curvas)}', flush=True)

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    resultados = {}
    for rama in RAMAS:
        vq_ck = mejor_ckpt(f'results/enc_cmp/{rama}/vqvae/checkpoints/**/*.ckpt')
        mg_ck = mejor_ckpt(f'results/enc_cmp/{rama}/maskgit/checkpoints/**/*.ckpt')
        print(f'\n[{rama}]\n  vqvae   {vq_ck}\n  maskgit {mg_ck}', flush=True)

        vq = VQVAE.load_from_checkpoint(vq_ck, map_location=dev).eval().to(dev)
        mg = MaskGIT.load_from_checkpoint(mg_ck, map_location=dev, strict=False).eval().to(dev)

        ctrl, pert = {}, {d: {} for d in DFS}
        for k, (snid, (o, idx, sc)) in enumerate(curvas.items()):
            ctrl[snid], mapa_c = nll_max(vq, mg, render(o, rama, fig, ax), dev)
            for d in DFS:
                p = o.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
                pert[d][snid], mapa_p = nll_max(vq, mg, render(p, rama, fig, ax), dev)
                if k == 0 and d in (50, 500):
                    np.save(OUT / f'mapa_{rama}_df{d}.npy', mapa_p)
            if k == 0:
                np.save(OUT / f'mapa_{rama}_control.npy', mapa_c)
            if (k + 1) % 10 == 0:
                print(f'  {k + 1}/{len(curvas)}', flush=True)

        cs = np.array([ctrl[s] for s in curvas])
        tau = float(np.quantile(cs, CTRL_Q))
        met = {'tau': tau, 'ctrl_mean': float(cs.mean()), 'ctrl_std': float(cs.std()),
               'por_df': {}}
        for d in DFS:
            ps = np.array([pert[d][s] for s in curvas])
            met['por_df'][d] = {
                'recall': float((ps > tau).mean()),
                'auc': auc(cs, ps),
                'pareada': float(np.mean([pert[d][s] > ctrl[s] for s in curvas])),
                'dmax_mediano': float(np.median([pert[d][s] - ctrl[s] for s in curvas])),
            }

        rhos = []
        for s in curvas:
            serie = [pert[d][s] for d in DFS]
            r = spearmanr(DFS, serie).statistic
            if np.isfinite(r):
                rhos.append(float(r))
        met['monotonia_mediana'] = float(np.median(rhos))
        met['monotonia_frac_positiva'] = float(np.mean([r > 0 for r in rhos]))
        met['nll_control'] = {s: ctrl[s] for s in curvas}
        met['nll_perturbado'] = {str(d): {s: pert[d][s] for s in curvas} for d in DFS}
        resultados[rama] = met

        print(f'  tau={tau:.3f}  monotonia={met["monotonia_mediana"]:+.3f}', flush=True)
        for d in DFS:
            m = met['por_df'][d]
            print(f'    df{d:<4} recall={m["recall"]:.2f}  auc={m["auc"]:.3f}  '
                  f'pareada={m["pareada"]:.2f}  dmax={m["dmax_mediano"]:+.3f}', flush=True)

        del vq, mg
        torch.cuda.empty_cache()

    plt.close(fig)
    with open(OUT / 'resultados.json', 'w') as f:
        json.dump({'dfs': DFS, 'n_objetos': len(curvas), 'excluidos': sorted(EXCLUIR),
                   'ramas': resultados}, f, indent=2)

    graficar(resultados)
    print(f'\nListo en {OUT}/', flush=True)


def graficar(res):
    colores = {'actual': '#d7191c', 'mad100': '#2c7bb6', 'mad300': '#4C9F70'}
    nombres = {'actual': 'actual (p01-p99 + clip)', 'mad100': 'mad100 (asinh/MAD, U=100)',
               'mad300': 'mad300 (asinh/MAD, U=300)'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for clave, titulo, ylab in (
            ('recall', f'Recall con tau = P{int(CTRL_Q * 100)} de los controles', 'recall'),
            ('auc', 'AUC control contra perturbada', 'AUC'),
            ('pareada', 'Deteccion pareada: NLL supera al de su propio control', 'fraccion')):
        a = axes[('recall', 'auc', 'pareada').index(clave)]
        for rama, m in res.items():
            a.plot(DFS, [m['por_df'][d][clave] for d in DFS], 'o-', lw=2,
                   color=colores[rama], label=nombres[rama])
        a.set_xscale('log')
        a.set_xticks(DFS, [str(d) for d in DFS], fontsize=7)
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel(ylab)
        a.set_title(titulo, fontsize=10)
        a.set_ylim(0, 1.02)
        a.axhline(0.5, color='gray', ls=':', lw=1)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'comparacion_codificaciones.png', dpi=140)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for rama, m in res.items():
        a1.plot(DFS, [m['por_df'][d]['dmax_mediano'] for d in DFS], 'o-', lw=2,
                color=colores[rama], label=nombres[rama])
    a1.axhline(0, color='gray', ls=':', lw=1)
    a1.set_xscale('log')
    a1.set_xticks(DFS, [str(d) for d in DFS], fontsize=7)
    a1.set_xlabel(r'$df$  [veces MAD$_r$]')
    a1.set_ylabel('mediana de  NLL(perturbada) - NLL(control)')
    a1.set_title('Cuanto sube el NLL respecto del propio control\n'
                 '(el valor absoluto no es comparable entre ramas, la forma si)',
                 fontsize=9)
    a1.legend(frameon=False, fontsize=8)

    ramas = list(res)
    a2.bar(range(len(ramas)), [res[r]['monotonia_mediana'] for r in ramas],
           color=[colores[r] for r in ramas])
    a2.set_xticks(range(len(ramas)), ramas)
    a2.axhline(0, color='gray', lw=1)
    a2.set_ylim(-1, 1)
    a2.set_ylabel(r'Spearman($df$, NLL), mediana por objeto')
    a2.set_title('Monotonia: un pico mas alto,\nun NLL mas alto', fontsize=10)
    for i, r in enumerate(ramas):
        v = res[r]['monotonia_mediana']
        a2.text(i, v + (0.04 if v >= 0 else -0.09), f'{v:+.2f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'monotonia_y_delta.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
