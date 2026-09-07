"""
Verifica en el mapa de tokens lo que ya se vio en pixeles: que a partir de
cierto df la entrada del MaskGIT deja de cambiar.

Toma las imagenes de la cola que ya existen, las pasa por el VQ-VAE y cuenta
cuantos de los 1024 tokens cambian respecto del control en cada nivel de df.
Si esa cuenta se aplana, el prior no puede subir el NLL por mucho flujo que
se agregue: esta recibiendo la misma secuencia.

Uso: python scripts/verificar_tokens_congelan.py
"""
import sys
sys.path.insert(0, '.')
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.vqvae.model import VQVAE

VQVAE_CKPT = ('results/vqvae_waterfall_global_full/checkpoints/'
              'vqvae-epoch=26-train/loss=0.0597.ckpt')
TAIL_DIR = Path('data/images/elasticc_1/synthetic_mad_tail')
DT_DIR = Path('data/images/elasticc_1/synthetic_mad_dt_wfg')
TAIL_SUM = Path('results/synthetic_eval/mad_tail/summary.json')
OUT = Path(os.environ.get('PROFE_FIG_OUT', 'imagenes'))
DFS = list(range(10, 101, 10))
DFS_FROZEN = [1, 2, 5, 10, 20, 50, 100, 200]
DT_REF = 30

tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


@torch.no_grad()
def tokens_de(model, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([tf(Image.open(p).convert('RGB'))
                             for p in paths[i:i + bs]]).to(device)
        res = model(batch)
        tok = res[2]
        out.append(tok.reshape(tok.shape[0], -1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=device)
    model.eval().to(device)
    print(f'device={device}', flush=True)

    snids = sorted({r['snid'] for r in json.load(open(TAIL_SUM))['per_image']
                    if r['tag'] == 'control'})

    # --- escala libre ---
    filas = []
    for s in snids:
        c = TAIL_DIR / f'{s}__control.png'
        ps = [TAIL_DIR / f'{s}__df{d}_unfrozen.png' for d in DFS]
        if not c.exists() or not all(p.exists() for p in ps):
            continue
        T = tokens_de(model, [c] + ps, device)
        filas.append([(T[0] != T[i + 1]).sum() for i in range(len(DFS))])
    A = np.array(filas, float)
    med = np.median(A, axis=0)
    print(f'\nESCALA LIBRE  ({len(A)} objetos)  tokens distintos del control, de 1024')
    for d, m in zip(DFS, med):
        print(f'  df={d:3d}   {m:6.1f}', flush=True)

    # --- escala congelada ---
    filas_f = []
    for s in snids:
        c = DT_DIR / f'{s}__control.png'
        ps = [DT_DIR / f'{s}__df{d}_dt{DT_REF}.png' for d in DFS_FROZEN]
        if not c.exists() or not all(p.exists() for p in ps):
            continue
        T = tokens_de(model, [c] + ps, device)
        filas_f.append([(T[0] != T[i + 1]).sum() for i in range(len(DFS_FROZEN))])
    B = np.array(filas_f, float) if filas_f else None
    if B is not None:
        medf = np.median(B, axis=0)
        print(f'\nESCALA CONGELADA  ({len(B)} objetos)  dt={DT_REF}d')
        for d, m in zip(DFS_FROZEN, medf):
            print(f'  df={d:3d}   {m:6.1f}', flush=True)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.9))
    ax.plot(DFS, med, 'o-', color='#2c7bb6', lw=2)
    ax.fill_between(DFS, *np.percentile(A, [25, 75], axis=0),
                    color='#2c7bb6', alpha=0.18)
    ax.set_xlabel(r'$df$  [veces MAD$_r$]')
    ax.set_ylabel('tokens distintos del control  (de 1024)')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Escala libre: lo que realmente recibe el MaskGIT\n'
                 f'mediana sobre {len(A)} SNIa', fontsize=10)
    if B is not None:
        ax2.plot(DFS_FROZEN, medf, 'o-', color='#d7191c', lw=2)
        ax2.fill_between(DFS_FROZEN, *np.percentile(B, [25, 75], axis=0),
                         color='#d7191c', alpha=0.18)
        ax2.set_xscale('log')
        ax2.set_xticks(DFS_FROZEN, [str(d) for d in DFS_FROZEN])
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel(r'$df$  [veces MAD$_r$],  $\Delta t$ = 30 d')
        ax2.set_ylabel('tokens distintos del control  (de 1024)')
        ax2.set_title('Escala congelada', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'tokens_congelan.png', dpi=140)
    plt.close(fig)

    with open(OUT / 'tokens_congelan.json', 'w') as f:
        json.dump({'dfs_libre': DFS, 'tokens_libre': med.tolist(),
                   'dfs_congelado': DFS_FROZEN,
                   'tokens_congelado': medf.tolist() if B is not None else None,
                   'n_objetos': len(A)}, f, indent=2)
    print(f'\nFigura en {OUT}/tokens_congelan.png', flush=True)


if __name__ == '__main__':
    main()
