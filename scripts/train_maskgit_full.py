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

from scripts.train_maskgit_cls import TokenDataset, MaskGITWithViz

REP = sys.argv[1]
assert REP in ('overlay_lines', 'points', 'waterfall', 'waterfall_global')

VQVAE_CKPTS = {
    'overlay_lines':    'results/vqvae_overlay_full/checkpoints/vqvae-epoch=17-train/loss=0.1378.ckpt',
    'points':           'results/vqvae_points_full/checkpoints/vqvae-epoch=52-train/loss=0.0658.ckpt',
    'waterfall':        'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt',
    'waterfall_global': 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt',
}
IMAGE_DIRS = {
    'overlay_lines':    'data/images/elasticc_1/overlay',
    'points':           'data/images/elasticc_1/overlay_points',
    'waterfall':        'data/images/elasticc_1/waterfall',
    'waterfall_global': 'data/images/elasticc_1/waterfall_global',
}
LABELS_PATHS = {
    'overlay_lines':    'data/labels_overlay_full.json',
    'points':           'data/labels_points_full.json',
    'waterfall':        'data/labels_waterfall_full.json',
    'waterfall_global': 'data/labels_waterfall_global_full.json',
}


def main():
    tokens = np.load(f'data/tokens/tokens_{REP}_full.npy')

    with open(f'data/tokens/paths_{REP}_full.txt') as f:
        paths = [l.strip() for l in f.readlines()]
    with open(LABELS_PATHS[REP]) as f:
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
        image_dir     = IMAGE_DIRS[REP],
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

    ckpt_dir = f'results/maskgit_full/{REP}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath    = ckpt_dir,
        filename   = "maskgit-{epoch:02d}-{val/loss:.4f}",
        monitor    = "val/loss",
        mode       = "min",
        save_top_k = 1,
    )
    early_stop = EarlyStopping(
        monitor  = "val/loss",
        patience = 5,
        mode     = "min",
    )

    logger = WandbLogger(
        project  = "vqvae-supernovas",
        name     = f"maskgit-full-{REP}",
        save_dir = f"results/maskgit_full/{REP}/logs",
    )

    trainer = L.Trainer(
        max_epochs           = 50,
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
