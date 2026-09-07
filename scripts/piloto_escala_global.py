"""
Piloto: escala global fija (asinh) contra la normalizacion actual por objeto.

Renderiza las mismas 40 SNIa con varias codificaciones de flujo, las pasa por
el VQ-VAE y el MaskGIT actuales y compara.

AVISO IMPORTANTE
El VQ-VAE y el MaskGIT se entrenaron con imagenes de la codificacion actual.
Con asinh las imagenes le quedan fuera de distribucion, asi que el NLL
absoluto sale inflado y un umbral tipo tau no significa nada. Lo que si es
comparable es todo lo PAREADO, cada objeto contra su propio control con la
misma codificacion, porque el sesgo por estar fuera de distribucion se
cancela en la resta.

Metricas por codificacion:
  dmax        mediana de  max NLL(perturbada) - max NLL(control)
  sube        fraccion de objetos con dmax > 0
  monotonia   Spearman entre df y max NLL, dentro de cada objeto
  tokens      tokens del mapa 32x32 que cambian respecto del control
  acierto     fraccion de veces que el argmax del NLL cae en la banda r
              y en el tercio final del tiempo, que es donde esta el pico

La ultima es la que mas importa: no basta con que el NLL suba, tiene que
subir en el lugar correcto.

Uso: python scripts/piloto_escala_global.py [N_OBJ]
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT = ('results/vqvae_waterfall_global_full/checkpoints/'
              'vqvae-epoch=26-train/loss=0.0597.ckpt')
MASKGIT_CKPT = ('results/maskgit_full/waterfall_global/checkpoints/'
                'maskgit-epoch=20-val/loss=3.6256.ckpt')
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'results/figs_profe'))
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
BASE_FILE = 'lc_SNIa-SALT2.parquet'

N_OBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 42
TARGET_BAND = 'r'
MIN_OBS_BAND, MIN_TAIL, TAIL_FRAC = 8, 6, 0.30
DFS = [10, 20, 50, 100, 200, 500, 1000]
STEP, AMP_LAYOUT = 2.0, 1.8
GRID, N_ROUNDS, MASK_RATIO = 32, 10, 0.5

cfg = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
fig_params = cfg['ft_classification']['imgs_params']['fig_params']
ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dc = ds['dict_columns']
all_bands = ds['all_bands']
col_mjd, col_flux, col_band = dc['mjd'], dc['flux'], dc['band']
N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())
Y_LO, Y_HI = -0.2, (N_BANDS - 1) * STEP + AMP_LAYOUT + 0.2
Y_SPAN = Y_HI - Y_LO
X_LO, X_HI = -0.05, 1.05

# Zona donde deberia caer el pico: banda r, ultimo 30% del tiempo.
_j = all_bands[TARGET_BAND]
ROW_HI = int(np.floor((Y_HI - (_j * STEP + AMP_LAYOUT)) / Y_SPAN * GRID))
ROW_LO = int(np.ceil((Y_HI - _j * STEP) / Y_SPAN * GRID))
COL_LO = int(np.floor(((1 - TAIL_FRAC) - X_LO) / (X_HI - X_LO) * GRID))

tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def enc_actual(f, obj_f):
    p01, p99 = np.quantile(obj_f, 0.01), np.quantile(obj_f, 0.99)
    rng = (p99 - p01) if p99 > p01 else 1.0
    return np.clip((f - p01) / rng, 0.0, 1.0)


def mk_asinh(beta, fmax):
    def enc(f, obj_f):
        u = np.arcsinh(f / beta) / np.arcsinh(fmax / beta)
        return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0
    return enc


ENCS = {'actual': enc_actual}
for b in (2.0, 4.0, 8.0):
    for fm in (5e2, 5e3, 5e4):
        ENCS[f'asinh b{b:g} F{fm:.0e}'] = mk_asinh(b, fm)


def render(obj_df, enc):
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    m0 = float(obj_df[col_mjd].min())
    mr = max(float(obj_df[col_mjd].max()) - m0, 1.0)
    obj_f = obj_df[col_flux].astype('float64').values
    for bk, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == bk]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        y = enc(bd[col_flux].astype('float64').values, obj_f) * AMP_LAYOUT + j * STEP
        ax.plot((bd[col_mjd] - m0) / mr, y, 'o-', color=fig_params['colors'][j],
                markersize=2, linewidth=1.0)
    ax.set_xlim([X_LO, X_HI])
    ax.set_ylim([Y_LO, Y_HI])
    ax.axis('off')
    fig.add_artist(patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                                     facecolor='none', transform=fig.transFigure))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    buf.close()
    plt.close(fig)
    return im


@torch.no_grad()
def nll_maps(mg, tokens):
    B, Tn = tokens.shape
    tot = torch.zeros(B, Tn, device=tokens.device)
    cnt = torch.zeros(B, Tn, device=tokens.device)
    n_mask = int(Tn * MASK_RATIO)
    for _ in range(N_ROUNDS):
        mask = torch.zeros(B, Tn, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            mask[i, torch.randperm(Tn, device=tokens.device)[:n_mask]] = True
        masked = tokens.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        tot += (-tlp) * mask.float()
        cnt += mask.float()
    return (tot / cnt.clamp(min=1.0)).cpu().numpy()


@torch.no_grad()
def encode(vq, imgs, dev, bs=20):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(im) for im in imgs[i:i + bs]]).to(dev)
        z = vq.encoder(x)
        _, _, idx, *_ = vq.codebook(z)
        out.append(idx.view(x.shape[0], -1).long())
    return torch.cat(out, 0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev).eval().to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev,
                                             strict=False).eval().to(dev)
    print(f'device={dev} | zona esperada: filas {ROW_HI}-{ROW_LO}, '
          f'columnas {COL_LO}-{GRID}', flush=True)

    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df['SNID'] = df['SNID'].astype(str)
    cand = []
    for s in df['SNID'].unique()[:40000]:
        sub = df[df['SNID'] == s]
        if all((sub[col_band] == b).sum() >= MIN_OBS_BAND for b in BAND_NAMES):
            cand.append(str(s))
        if len(cand) >= N_OBJ * 2:
            break
    chosen = list(rng.choice(cand, size=min(N_OBJ, len(cand)), replace=False))
    print(f'{len(chosen)} objetos | {len(ENCS)} codificaciones | df={DFS}', flush=True)

    objs = []
    for snid in chosen:
        o = df[df['SNID'] == snid].copy()
        o[col_flux] = o[col_flux].astype('float64')
        sc = mad(o.loc[o[col_band] == TARGET_BAND, col_flux].values)
        if not np.isfinite(sc) or sc <= 0:
            continue
        sub = o[o[col_band] == TARGET_BAND].sort_values(col_mjd)
        t0, t1 = float(o[col_mjd].min()), float(o[col_mjd].max())
        idx = list(sub.index[sub[col_mjd] >= t1 - TAIL_FRAC * max(t1 - t0, 1.0)])
        if len(idx) < MIN_TAIL:
            idx = list(sub.index[-MIN_TAIL:])
        objs.append((snid, o, idx, sc))

    resumen = {}
    for name, enc in ENCS.items():
        ctrl_imgs = [render(o, enc) for _, o, _, _ in objs]
        Tc = encode(vq, ctrl_imgs, dev)
        Mc = nll_maps(mg, Tc)
        max_c = Mc.max(1)

        dmax, sube, toks, acierto, curvas = [], [], [], [], []
        for d in DFS:
            imgs = []
            for _, o, idx, sc in objs:
                p = o.copy()
                p.loc[idx, col_flux] = p.loc[idx, col_flux] + d * sc
                imgs.append(render(p, enc))
            Tp = encode(vq, imgs, dev)
            Mp = nll_maps(mg, Tp)
            max_p = Mp.max(1)
            dmax.append(float(np.median(max_p - max_c)))
            sube.append(float((max_p > max_c).mean()))
            toks.append(float(np.median((Tc != Tp).sum(1).cpu().numpy())))
            am = Mp.argmax(1)
            fila, col = am // GRID, am % GRID
            acierto.append(float(((fila >= ROW_HI) & (fila <= ROW_LO) &
                                  (col >= COL_LO)).mean()))
            curvas.append(max_p)
        curvas = np.array(curvas)
        mono = float(np.nanmedian([spearman(DFS, curvas[:, i])
                                   for i in range(curvas.shape[1])]))
        resumen[name] = dict(dmax=dmax, sube=sube, tokens=toks,
                             acierto=acierto, monotonia=mono,
                             max_ctrl=float(np.median(max_c)))
        print(f'{name:22s} dmax={dmax[2]:+5.2f}(df50) sube={sube[2]:.2f} '
              f'tok={toks[2]:6.1f} acierto={acierto[2]:.2f} mono={mono:+.2f}',
              flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    campos = [('dmax', r'$\Delta$ max NLL (perturbada $-$ control)'),
              ('acierto', 'el argmax cae en la banda r y en la cola'),
              ('sube', 'fraccion de objetos que suben'),
              ('tokens', 'tokens cambiados (de 1024)')]
    for ax, (k, lab) in zip(axes.ravel(), campos):
        for name in ENCS:
            est = dict(lw=2.6, color='#d7191c', zorder=5) if name == 'actual' \
                else dict(lw=1.3, alpha=0.85)
            ax.plot(DFS, resumen[name][k], 'o-', label=name, ms=4, **est)
        ax.set_xscale('log')
        ax.set_xticks(DFS, [str(d) for d in DFS])
        ax.set_xlabel(r'$df$  [veces MAD$_r$]')
        ax.set_ylabel(lab, fontsize=9)
        if k in ('acierto', 'sube'):
            ax.set_ylim(0, 1.02)
            ax.axhline(1.0 / GRID if k == 'acierto' else 0.5,
                       color='gray', ls=':', lw=1)
    axes[0, 0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle('Piloto de escala global fija. El NLL absoluto no es comparable '
                 '(imagenes fuera de distribucion); lo pareado si.', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'piloto_escala.png', dpi=140)
    plt.close(fig)

    with open(OUT / 'piloto_escala.json', 'w') as f:
        json.dump({'dfs': DFS, 'zona': [ROW_HI, ROW_LO, COL_LO],
                   'n_obj': len(objs), 'resumen': resumen}, f, indent=2)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
