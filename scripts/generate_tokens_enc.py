"""Tokeniza las imagenes de una rama con su propio VQ-VAE.

Cada rama usa el VQ-VAE entrenado sobre SUS imagenes, que es lo que hace justa
la comparacion: ninguna de las dos codificaciones queda fuera de distribucion.

Uso: python scripts/generate_tokens_enc.py <rama> [corrida]
     rama:    actual | mad100 | mad300
     corrida: vacio para el subconjunto, "full" para el dataset completo
"""
import sys
sys.path.insert(0, '.')
import glob
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset

RAMA = sys.argv[1]
# mad300t es mad300 con el eje temporal en escala absoluta en vez de normalizado
# por objeto: misma codificacion de flujo, otro renderizado.
assert RAMA in ('actual', 'mad100', 'mad300', 'mad300t')
CORRIDA = sys.argv[2] if len(sys.argv) > 2 else ''
SUF = f'_{CORRIDA}' if CORRIDA else ''

ROOT = f'data/enc_cmp/{RAMA}'
cfg = OmegaConf.load(f'configs/vqvae_enc_{RAMA}{SUF}.yaml').vqvae
CKPT_DIR = cfg.checkpoint_dir

# El nombre del checkpoint lleva "train/loss", y esa barra hace que Lightning
# escriba un subdirectorio: checkpoints/vqvae-epoch=39-train/loss=0.0328.ckpt
cands = glob.glob(f'{CKPT_DIR}/**/*.ckpt', recursive=True)
if not cands:
    raise SystemExit(f'sin checkpoints de VQ-VAE en {CKPT_DIR}')


def loss_de(p):
    m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
    return float(m.group(1)) if m else float('inf')


ckpt = min(cands, key=loss_de)
print(f'[{RAMA}] VQ-VAE: {ckpt}  (loss {loss_de(ckpt):.4f}, {len(cands)} candidatos)', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE.load_from_checkpoint(ckpt).to(device)
model.eval()
model.freeze()

image_paths = sorted(glob.glob(f'{ROOT}/images/*.png'))
print(f'[{RAMA}] imagenes: {len(image_paths):,}', flush=True)

loader = DataLoader(VQVAEDataset(image_paths, image_size=cfg.image_size),
                    batch_size=256, shuffle=False, num_workers=4, pin_memory=False)

all_tokens = []
with torch.no_grad():
    for batch in tqdm(loader):
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        z_e = model.encoder(x)
        _, _, indices, _, _, _ = model.codebook(z_e)
        all_tokens.append(indices.view(indices.size(0), -1).cpu().numpy().astype(np.uint16))

all_tokens = np.concatenate(all_tokens, axis=0)
print(f'[{RAMA}] tokens: {all_tokens.shape}', flush=True)

np.save(f'{ROOT}/tokens{SUF}.npy', all_tokens)
with open(f'{ROOT}/paths{SUF}.txt', 'w') as f:
    f.write('\n'.join(image_paths))
print(f'[{RAMA}] listo: {ROOT}/tokens{SUF}.npy', flush=True)
