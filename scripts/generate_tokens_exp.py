import sys
sys.path.insert(0, '.')
import os, glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset

# Argumentos: ckpt_path, image_dir, out_suffix
ckpt_path = sys.argv[1]
image_dir = sys.argv[2]
suffix    = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = OmegaConf.load('configs/vqvae_config.yaml').vqvae

print(f"Cargando checkpoint: {ckpt_path}", flush=True)
model = VQVAE.load_from_checkpoint(ckpt_path).to(device)
model.eval()
model.freeze()

image_paths = sorted(glob.glob(f'{image_dir}/*.png'))
print(f"Total imagenes: {len(image_paths):,}", flush=True)

dataset = VQVAEDataset(image_paths, image_size=cfg.image_size)
loader  = DataLoader(dataset, batch_size=256, shuffle=False,
                     num_workers=4, pin_memory=False)

os.makedirs('data/tokens', exist_ok=True)
all_tokens = []
all_paths  = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(loader)):
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
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

np.save(f'data/tokens/tokens_{suffix}.npy', all_tokens)
with open(f'data/tokens/paths_{suffix}.txt', 'w') as f:
    f.write('\n'.join(all_paths))
print(f"Listo! data/tokens/tokens_{suffix}.npy", flush=True)
