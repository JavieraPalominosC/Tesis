"""Entrena VQ-VAE sobre superficies GP del piloto.

Uso: python scripts/train_vqvae_gp.py [configs/vqvae_gp_pilot.yaml]
     ACCELERATOR=cpu python scripts/train_vqvae_gp.py ...   # en la partición main

En CPU la precisión pasa a 32-true: 16-mixed requiere GPU.
"""
import sys
sys.path.insert(0, ".")
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.models.vqvae.model_gp import VQVAEGP
from src.data.gp_dataset import GPSurfaceDataModule


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/vqvae_gp_pilot.yaml"
    cfg = OmegaConf.load(cfg_path).vqvae

    num_classes = getattr(cfg, "num_classes", 0)
    classes_path = Path(cfg.folds_path).parent / "classes.json"
    if classes_path.exists():
        with open(classes_path) as f:
            num_classes = len(json.load(f)["classes"])
        print(f"num_classes desde classes.json: {num_classes}", flush=True)

    # Detecta canales del primer .npy cacheado (1 = mean, 2 = mean+std)
    in_channels = cfg.in_channels
    folds_path = Path(cfg.folds_path)
    if folds_path.exists():
        with open(folds_path) as f:
            sample_path = json.load(f)["0"]["train"][0]
        in_channels = int(np.load(sample_path).shape[0])
        print(f"in_channels desde cache: {in_channels}", flush=True)

    dm = GPSurfaceDataModule(
        folds_path=cfg.folds_path,
        fold=0,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        labels_path=getattr(cfg, "labels_path", None),
    )

    model = VQVAEGP(
        in_channels=in_channels,
        hidden_channels=list(cfg.hidden_channels),
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        learning_rate=cfg.learning_rate,
        num_classes=num_classes,
        cls_weight=getattr(cfg, "cls_weight", 0.1),
        cls_dropout=getattr(cfg, "cls_dropout", 0.1),
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename="vqvae-{epoch:02d}-{train/loss:.4f}",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="train/loss", patience=8, mode="min")

    logger = WandbLogger(
        project="vqvae-supernovas",
        name=getattr(cfg, "run_name", "vqvae-gp-pilot"),
        save_dir=cfg.log_dir,
    )

    accelerator = os.environ.get("ACCELERATOR", "gpu")
    if accelerator == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
        print(f"CPU con {torch.get_num_threads()} hilos", flush=True)

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=accelerator,
        devices=1,
        precision="32-true" if accelerator == "cpu" else "16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
    )

    resume = f"{cfg.checkpoint_dir}/last.ckpt"
    trainer.fit(model, dm, ckpt_path=resume if os.path.exists(resume) else None)
    print(f"Mejor checkpoint: {checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
