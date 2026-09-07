"""Cruce 2x2: forma de mascara en entrenamiento contra forma de mascara al puntuar.

La primera version de esta comparacion tenia un sesgo: el evaluador enmascara
tokens al azar, que es exactamente el regimen que el modelo de ventana nunca vio
en entrenamiento. Cualquier diferencia mezclaba dos cosas, si la ventana es mejor
o peor como prior, y si lo estabamos evaluando fuera de su distribucion.

Aca se separan. Cada uno de los dos MaskGIT se puntua con las dos formas de
mascara. La diagonal (cada modelo con su propia forma) es la comparacion honesta
entre estrategias; la fila de cada modelo mide cuanto le cuesta el desajuste.

Se usan mas rondas que en eval_enc_cmp porque ya sabemos que el ruido del puntaje
era del orden de la mitad de la senal, y porque la mascara de rectangulo cubre los
tokens de forma despareja: los del borde entran en menos rondas que los del centro.
Los tokens que nunca quedaron enmascarados se excluyen del maximo.

Uso: python scripts/eval_mascara_cruzada.py
"""
import sys
sys.path.insert(0, '.')
import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

from scripts.eval_enc_cmp import GRID, auc, tf
from scripts.analizar_maskgits import zona_objetivo
from src.models.maskgit import MaskGIT
from src.models.vqvae.model import VQVAE

VQVAE_CKPT = ('results/vqvae_waterfall_global_full/checkpoints/'
              'vqvae-epoch=26-train/loss=0.0597.ckpt')
MODELOS = {
    'aleatoria': ('results/maskgit_full/waterfall_global/checkpoints/'
                  'maskgit-epoch=20-val/loss=3.6256.ckpt'),
    'ventana': ('results/maskgit_window/waterfall_global/checkpoints/'
                'maskgit-window-epoch=20-val/loss=4.2217.ckpt'),
}
MANIFEST = 'data/synthetic_mad_tail_manifest.json'
OUT = Path('results/enc_cmp/eval')
DFS = list(range(10, 101, 10))
N_RONDAS, RATIO, CTRL_Q = 24, 0.5, 0.95
SEED = 0


def mascara_aleatoria(T, n, dev, g):
    m = torch.zeros(1, T, dtype=torch.bool, device=dev)
    m[0, torch.randperm(T, device=dev, generator=g)[:n]] = True
    return m


def mascara_ventana(T, n, dev, g):
    """Rectangulo contiguo de area n, igual que en el entrenamiento."""
    H = W = int(round(math.sqrt(T)))
    w = int(torch.randint(1, W + 1, (1,), device=dev, generator=g).item())
    h = min(H, max(1, (n + w - 1) // w))
    w = min(W, max(1, (n + h - 1) // h))
    r0 = 0 if h == H else int(torch.randint(0, H - h + 1, (1,), device=dev, generator=g).item())
    c0 = 0 if w == W else int(torch.randint(0, W - w + 1, (1,), device=dev, generator=g).item())
    grid = torch.zeros(H, W, dtype=torch.bool, device=dev)
    grid[r0:r0 + h, c0:c0 + w] = True
    flat = grid.reshape(-1)
    idx = flat.nonzero(as_tuple=False).view(-1)
    if idx.numel() > n:
        keep = idx[torch.randperm(idx.numel(), device=dev, generator=g)[:n]]
        flat[:] = False
        flat[keep] = True
    elif idx.numel() < n:
        hueco = (~flat).nonzero(as_tuple=False).view(-1)
        extra = hueco[torch.randperm(hueco.numel(), device=dev, generator=g)[:n - int(idx.numel())]]
        flat[extra] = True
    return flat.view(1, T)


FORMAS = {'aleatoria': mascara_aleatoria, 'ventana': mascara_ventana}


@torch.no_grad()
def nll_map(vq, mg, img, dev, forma, g):
    x = tf(img).unsqueeze(0).to(dev)
    z = vq.encoder(x)
    _, _, idx, *_ = vq.codebook(z)
    tokens = idx.view(1, -1).long().to(dev)

    T = tokens.shape[1]
    n = int(T * RATIO)
    total = torch.zeros(1, T, device=dev)
    counts = torch.zeros(1, T, device=dev)
    for _ in range(N_RONDAS):
        mask = FORMAS[forma](T, n, dev, g)
        masked = tokens.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp) * mask.float()
        counts += mask.float()

    nll = (total / counts.clamp(min=1.0))[0].cpu().numpy()
    visto = (counts[0] > 0).cpu().numpy()
    nll = np.where(visto, nll, -np.inf)
    return float(np.max(nll)), nll.reshape(GRID, GRID)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r0, r1, c0, c1 = zona_objetivo()
    print(f'zona objetivo: filas {r0}-{r1}, columnas {c0}-{c1}', flush=True)

    with open(MANIFEST) as f:
        man = [r for r in json.load(f) if not r.get('frozen')]
    por_snid = collections.defaultdict(dict)
    for r in man:
        p = Path(r['path'])
        if p.exists():
            por_snid[r['snid']][r['tag']] = p
    snids = sorted(s for s, d in por_snid.items()
                   if 'control' in d and all(f'df{k}_unfrozen' in d for k in DFS))
    print(f'objetos: {len(snids)}   rondas: {N_RONDAS}', flush=True)

    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev).eval().to(dev)
    res = {}

    for modelo, ckpt in MODELOS.items():
        mg = MaskGIT.load_from_checkpoint(ckpt, map_location=dev, strict=False).eval().to(dev)
        for forma in FORMAS:
            g = torch.Generator(device=dev)
            g.manual_seed(SEED)
            ctrl, pert = {}, {d: {} for d in DFS}
            loc = {d: [] for d in DFS}
            loc_ctrl = []

            for s in snids:
                v, mapa = nll_map(vq, mg, Image.open(por_snid[s]['control']).convert('RGB'),
                                  dev, forma, g)
                ctrl[s] = v
                rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
                loc_ctrl.append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))
                for d in DFS:
                    v, mapa = nll_map(vq, mg,
                                      Image.open(por_snid[s][f'df{d}_unfrozen']).convert('RGB'),
                                      dev, forma, g)
                    pert[d][s] = v
                    rr, cc = np.unravel_index(np.argmax(mapa), mapa.shape)
                    loc[d].append(bool(r0 <= rr <= r1 and c0 <= cc <= c1))

            cs = np.array([ctrl[s] for s in snids])
            tau = float(np.quantile(cs, CTRL_Q))
            m = {'tau': tau, 'loc_control': float(np.mean(loc_ctrl)), 'por_df': {}}
            for d in DFS:
                ps = np.array([pert[d][s] for s in snids])
                m['por_df'][d] = {
                    'recall': float((ps > tau).mean()),
                    'auc': auc(cs, ps),
                    'pareada': float(np.mean([pert[d][s] > ctrl[s] for s in snids])),
                    'localizacion': float(np.mean(loc[d])),
                }
            rhos = [spearmanr(DFS, [pert[d][s] for d in DFS]).statistic for s in snids]
            m['monotonia'] = float(np.median([r for r in rhos if np.isfinite(r)]))
            res[f'{modelo}|{forma}'] = m

            marca = '  <- coincide' if modelo == forma else ''
            print(f'\n[entrenado {modelo} / puntuado {forma}]{marca}', flush=True)
            print(f'  monotonia={m["monotonia"]:+.3f}  loc control={m["loc_control"]:.2f}',
                  flush=True)
            for d in DFS:
                x = m['por_df'][d]
                print(f'    df{d:<4} recall={x["recall"]:.2f}  auc={x["auc"]:.3f}  '
                      f'pareada={x["pareada"]:.2f}  loc={x["localizacion"]:.2f}', flush=True)
        del mg
        torch.cuda.empty_cache()

    with open(OUT / 'mascara_cruzada.json', 'w') as f:
        json.dump({'dfs': DFS, 'n_objetos': len(snids), 'rondas': N_RONDAS, 'celdas': res},
                  f, indent=2)

    estilos = {
        'aleatoria|aleatoria': ('#8856a7', '-', 'entrenado aleatorio, puntuado aleatorio'),
        'ventana|ventana': ('#e6550d', '-', 'entrenado ventana, puntuado ventana'),
        'aleatoria|ventana': ('#8856a7', '--', 'entrenado aleatorio, puntuado ventana'),
        'ventana|aleatoria': ('#e6550d', '--', 'entrenado ventana, puntuado aleatorio'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, clave, tit in zip(axes, ('recall', 'auc', 'localizacion'),
                             ('Recall con tau = P95 de los controles',
                              'AUC control contra perturbada',
                              'El maximo del NLL cae donde metimos el pico')):
        for k, (col, ls, lab) in estilos.items():
            a.plot(DFS, [res[k]['por_df'][d][clave] for d in DFS], 'o', ls=ls, lw=2,
                   ms=4, color=col, label=lab)
        a.set_xlabel(r'$df$  [veces MAD$_r$]')
        a.set_ylabel(clave)
        a.set_title(tit, fontsize=10)
        a.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.suptitle('Forma de mascara: entrenamiento contra evaluacion  '
                 '(mismo VQ-VAE, codificacion antigua, linea llena = coinciden)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'mascara_cruzada.png', dpi=140)
    plt.close(fig)
    print(f'\nListo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
