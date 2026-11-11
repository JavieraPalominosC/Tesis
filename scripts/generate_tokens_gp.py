"""Tokeniza las superficies GP del piloto con el VQ-VAE entrenado.

Uso: python scripts/generate_tokens_gp.py [configs/vqvae_gp_pilot.yaml]
"""
import sys
sys.path.insert(0, ".")
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.vqvae.model_gp import VQVAEGP
from src.data.gp_dataset import GPSurfaceDataset

def loss_de(p):
    m = re.search(r"loss=([0-9.]+)\.ckpt$", p)
    return float(m.group(1)) if m else float("inf")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/vqvae_gp_pilot.yaml"
    cfg = OmegaConf.load(cfg_path).vqvae
    root = Path(cfg.folds_path).parent
    cands = glob.glob(f"{cfg.checkpoint_dir}/**/*.ckpt", recursive=True)
    if not cands:
        raise SystemExit(f"sin checkpoints en {cfg.checkpoint_dir}")
    ckpt = min(cands, key=loss_de)
    print(f"VQ-VAE GP: {ckpt}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    model = VQVAEGP.load_from_checkpoint(ckpt, map_location=device).to(device)
    model.eval()
    model.freeze()

    with open(cfg.folds_path) as f:
        folds = json.load(f)["0"]
    paths = folds["train"] + folds["val"]
    # las excluidas también se tokenizan, para poder puntuarlas después
    ho_path = root / "heldout_eval.json"
    if ho_path.exists():
        with open(ho_path) as f:
            for c, ps in json.load(f)["excluidas"].items():
                paths += ps
        print(f"incluye excluidas de {ho_path}", flush=True)
    # orden estable y sin duplicados
    paths = list(dict.fromkeys(paths))
    print(f"superficies: {len(paths):,}", flush=True)

    loader = DataLoader(
        GPSurfaceDataset(paths),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    all_tokens = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            z_e = model.encoder(x)
            _, _, indices, _, _, _ = model.codebook(z_e)
            all_tokens.append(
                indices.view(indices.size(0), -1).cpu().numpy().astype(np.uint16)
            )

    tokens = np.concatenate(all_tokens, axis=0)
    np.save(root / "tokens.npy", tokens)
    with open(root / "paths.txt", "w") as f:
        f.write("\n".join(paths))
    print(f"tokens: {tokens.shape} → {root / 'tokens.npy'}", flush=True)


if __name__ == "__main__":
    main()
