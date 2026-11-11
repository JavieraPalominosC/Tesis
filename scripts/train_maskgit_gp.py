"""MaskGIT sobre tokens del piloto GP.

Uso: python scripts/train_maskgit_gp.py [config.yaml] [aleatoria|ventana]
"""
import sys
sys.path.insert(0, ".")
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from scripts.mascara_ventana import enmascarar_ventana
from scripts.train_maskgit_cls import TokenDataset, MaskGITWithViz
from omegaconf import OmegaConf
from src.models.maskgit import MaskGIT

_args = sys.argv[1:]
CFG_PATH = next((a for a in _args if a.endswith(".yaml")), "configs/vqvae_gp_pilot.yaml")
MASCARA = next((a for a in _args if a in ("aleatoria", "ventana")), "aleatoria")
VEN = "_ventana" if MASCARA == "ventana" else ""
CFG = OmegaConf.load(CFG_PATH).vqvae
ROOT = Path(CFG.folds_path).parent


class MaskGITPlain(MaskGITWithViz):
    def __init__(self, **kwargs):
        MaskGIT.__init__(self, **kwargs)
        self.image_dir = Path(CFG.image_dir)
        self.n_viz = 0
        self.vqvae = None
        self.viz_snids = []
        self._val_scores = []
        self._val_labels = []

    def _mask_tokens(self, tokens):
        if MASCARA == "ventana":
            out = enmascarar_ventana(tokens, self.mask_token)
            if out is not None:
                return out
        return super()._mask_tokens(tokens)


def main():
    tokens = np.load(ROOT / "tokens.npy")
    with open(ROOT / "paths.txt") as f:
        paths = [l.strip() for l in f]
    with open(ROOT / "labels.json") as f:
        labels_map = json.load(f)
    with open(ROOT / "folds.json") as f:
        folds = json.load(f)["0"]

    idx_de = {p: i for i, p in enumerate(paths)}
    i_train = [idx_de[p] for p in folds["train"] if p in idx_de]
    i_val = [idx_de[p] for p in folds["val"] if p in idx_de]
    usados = i_train + i_val
    labels = np.array(
        [labels_map.get(Path(paths[i]).stem, -1) for i in usados], dtype=np.int64
    )
    dataset = TokenDataset(tokens[usados], labels)
    train_ds = torch.utils.data.Subset(dataset, list(range(len(i_train))))
    val_ds = torch.utils.data.Subset(
        dataset, list(range(len(i_train), len(usados)))
    )

    print(
        f"[gp] mascara {MASCARA} | tokens {tokens.shape} | "
        f"train {len(i_train):,} | val {len(i_val):,}",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = MaskGITPlain(
        num_tokens=512,
        seq_len=1024,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        ff_mult=4,
        dropout=0.1,
        learning_rate=1e-4,
    )

    base = Path(CFG.checkpoint_dir).parent.parent
    ckpt_dir = str(base / f"maskgit{VEN}" / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="maskgit-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
    logger = WandbLogger(
        project="vqvae-supernovas",
        name=f"{getattr(CFG, 'run_name', 'gp-pilot')}-maskgit{VEN}",
        save_dir=str(base / f"maskgit{VEN}" / "logs"),
    )

    # ACCELERATOR=mps (Mac) o cpu; 16-mixed solo en la GPU del clúster
    accelerator = os.environ.get("ACCELERATOR", "gpu")
    trainer = L.Trainer(
        max_epochs=int(os.environ.get("MAX_EPOCHS", 30)),
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
    )

    resume = f"{ckpt_dir}/last.ckpt"
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=resume if os.path.exists(resume) else None,
    )
    print(f"[gp] mejor: {checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
