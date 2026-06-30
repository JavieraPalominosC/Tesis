import sys
sys.path.insert(0, '.')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import glob

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset

# Config
cfg = OmegaConf.load('configs/vqvae_config.yaml').vqvae
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint del modelo con clasificación
ckpt_path = "results/vqvae/checkpoints_cls01/vqvae-epoch=07-train/loss=0.1188.ckpt"
print(f"Cargando checkpoint: {ckpt_path}", flush=True)

model = VQVAE.load_from_checkpoint(ckpt_path).to(device)
model.eval()
model.freeze()

# Dataset completo
image_paths = sorted(glob.glob('data/images/elasticc_1/2grid/*.png'))
print(f"Total imágenes: {len(image_paths):,}", flush=True)

dataset = VQVAEDataset(image_paths, image_size=cfg.image_size)
loader  = DataLoader(dataset, batch_size=256, shuffle=False,
                     num_workers=4, pin_memory=False)

# Generar tokens
os.makedirs('data/tokens', exist_ok=True)
all_tokens = []
all_paths  = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(loader)):
        x = batch.to(device)
        z_e = model.encoder(x)
        _, _, indices, _, _, _ = model.codebook(z_e)
        tokens = indices.view(indices.size(0), -1).cpu().numpy().astype(np.uint16)
        all_tokens.append(tokens)

        start = i * loader.batch_size
        end   = min(start + loader.batch_size, len(image_paths))
        all_paths.extend(image_paths[start:end])

all_tokens = np.concatenate(all_tokens, axis=0)
print(f"Tokens generados: {all_tokens.shape}", flush=True)

# Guardar con sufijo _cls
np.save('data/tokens/tokens_cls.npy', all_tokens)
with open('data/tokens/paths_cls.txt', 'w') as f:
    f.write('\n'.join(all_paths))

print("Listo! Guardado en data/tokens/tokens_cls.npy", flush=True)
