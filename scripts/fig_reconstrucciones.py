"""Pares original / reconstruccion del VQ-VAE mad300 sobre el fold de validacion.

Cierra el TODO de 04_resultados.tex. Usa objetos del fold de validacion, que el
modelo no vio en entrenamiento, y reporta el MSE de cada par en el titulo.
"""
import glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, '.')
from src.models.vqvae.model import VQVAE

CKPT_DIR = 'results/enc_cmp_full/mad300/vqvae/checkpoints'
FOLDS = 'data/enc_cmp/mad300/folds_full.json'
LABELS = 'data/enc_cmp/mad300/labels.json'
SALIDA = 'results/figs_profe/reconstruccion_2grid.png'
N = 4

TIPOS = ['SNIa-SALT2', 'SNIa-91bg', 'SNIax', 'SNII-Templates', 'SNII-NMF',
         'SNII+HostXT_V19', 'SNIb-Templates', 'SNIb+HostXT_V19',
         'SNIc-Templates', 'SNIc+HostXT_V19', 'SNIcBL+HostXT_V19',
         'SNIIb+HostXT_V19', 'SNIIn-MOSFIT', 'SNIIn+HostXT_V19',
         'SLSN-I+host', 'SLSN-I_no_host', 'PISN']


def mejor_ckpt(patron):
    cands = glob.glob(f'{patron}/**/*.ckpt', recursive=True)
    cands = [c for c in cands if 'last' not in c]

    def loss_de(p):
        m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
        return float(m.group(1)) if m else float('inf')

    return min(cands, key=loss_de)


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = mejor_ckpt(CKPT_DIR)
print('vqvae:', ck, flush=True)
vq = VQVAE.load_from_checkpoint(ck, map_location=dev).eval().to(dev)

val = json.load(open(FOLDS))['0']['val']
labels = json.load(open(LABELS))

# Un objeto por tipo, buscando tipos morfologicamente distintos entre si.
quiero = [0, 4, 14, 16]
elegidos = []
for p in val:
    snid = re.match(r'(\d+)', p.split('/')[-1]).group(1)
    lab = labels.get(snid)
    if lab in quiero and lab not in [e[1] for e in elegidos]:
        elegidos.append((p, lab, snid))
    if len(elegidos) == N:
        break

tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

fig, axes = plt.subplots(2, N, figsize=(3.0 * N, 6.3))
for k, (p, lab, snid) in enumerate(elegidos):
    x = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = vq(x)
    xr = out[0] if isinstance(out, (tuple, list)) else out
    mse = float(torch.mean((x - xr) ** 2))

    def a_img(t):
        return np.clip(t[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, 0, 1)

    axes[0, k].imshow(a_img(x))
    axes[0, k].set_title(f'{TIPOS[lab]}\nSNID {snid}', fontsize=8.5)
    axes[1, k].imshow(a_img(xr))
    axes[1, k].set_xlabel(f'MSE = {mse:.5f}', fontsize=8.5)
    for r in (0, 1):
        axes[r, k].set_xticks([])
        axes[r, k].set_yticks([])

axes[0, 0].set_ylabel('original', fontsize=10)
axes[1, 0].set_ylabel('reconstruccion', fontsize=10)
fig.suptitle('VQ-VAE mad300: reconstruccion de imagenes 2grid del fold de validacion',
             fontsize=11)
fig.tight_layout()
fig.subplots_adjust(hspace=0.12)
fig.savefig(SALIDA, dpi=160, bbox_inches='tight')
print('escrita', SALIDA)
