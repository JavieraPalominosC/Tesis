import sys
sys.path.insert(0, '.')
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Importar todo lo del script original (clases, modelo)
from scripts.train_maskgit_cls import TokenDataset, MaskGITWithViz

REP = sys.argv[1]  # 'filled' o 'overlay'
assert REP in ('filled', 'overlay')

VQVAE_CKPTS = {
    'filled':  'results/vqvae_exp/filled/checkpoints/vqvae-epoch=17-train/loss=0.1704.ckpt',
    'overlay': 'results/vqvae_exp/overlay/checkpoints/vqvae-epoch=18-train/loss=0.1816.ckpt',
}


def main():
    tokens = np.load(f'data/tokens/tokens_{REP}_exp.npy')

    with open(f'data/tokens/paths_{REP}_exp.txt') as f:
        paths = [l.strip() for l in f.readlines()]
    with open('data/images_exp/labels_subset.json') as f:
        labels_map = json.load(f)

    labels = np.array([
        labels_map.get(Path(p).stem, -1) for p in paths
    ], dtype=np.int64)
    print(f"[{REP}] Tokens: {tokens.shape}, labels unicos: {np.unique(labels[labels>=0]).shape[0]}", flush=True)

    n       = len(tokens)
    n_train = int(n * 0.8)
    n_val   = n - n_train

    dataset  = TokenDataset(tokens, labels)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=False)
    print(f"[{REP}] Train: {n_train:,} | Val: {n_val:,}", flush=True)

    model = MaskGITWithViz(
        image_dir     = f"data/images_exp/{REP}",
        vqvae_ckpt    = VQVAE_CKPTS[REP],
        n_viz         = 4,
        num_tokens    = 512,
        seq_len       = 1024,
        hidden_dim    = 256,
        n_layers      = 4,
        n_heads       = 4,
        ff_mult       = 4,
        dropout       = 0.1,
        learning_rate = 1e-4,
    )

    ckpt_dir = f'results/maskgit_exp/{REP}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath    = ckpt_dir,
        filename   = "maskgit-{epoch:02d}-{val/loss:.4f}",
        monitor    = "val/loss",
        mode       = "min",
        save_top_k = 2,
    )
    early_stop = EarlyStopping(
        monitor  = "val/loss",
        patience = 5,
        mode     = "min",
    )

    logger = WandbLogger(
        project  = "vqvae-supernovas",
        name     = f"maskgit-exp-{REP}",
        save_dir = f"results/maskgit_exp/{REP}/logs",
    )

    trainer = L.Trainer(
        max_epochs           = 30,
        accelerator          = "gpu",
        devices              = 1,
        precision            = "16-mixed",
        gradient_clip_val       = 0.5,
        gradient_clip_algorithm = "norm",
        callbacks            = [checkpoint, early_stop],
        logger               = logger,
        log_every_n_steps    = 100,
        val_check_interval   = 1.0,
        num_sanity_val_steps = 2,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"[{REP}] Mejor checkpoint: {checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
