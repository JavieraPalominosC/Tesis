import sys
sys.path.insert(0, '.')
import os
import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataModule


def main():
    torch.cuda.empty_cache()

    cfg = OmegaConf.load('configs/vqvae_config.yaml').vqvae

    labels_path = getattr(cfg, 'labels_path', None)
    num_classes = getattr(cfg, 'num_classes',  0)
    cls_weight  = getattr(cfg, 'cls_weight',   1.0)
    cls_dropout = getattr(cfg, 'cls_dropout',  0.1)

    dm = VQVAEDataModule(
        folds_path  = 'data/folds/folds.json',
        fold        = 0,
        image_size  = cfg.image_size,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        labels_path = labels_path,
    )

    model = VQVAE(
        in_channels     = cfg.in_channels,
        hidden_channels = list(cfg.hidden_channels),
        embedding_dim   = cfg.embedding_dim,
        num_embeddings  = cfg.num_embeddings,
        commitment_cost = cfg.commitment_cost,
        learning_rate   = cfg.learning_rate,
        num_classes     = num_classes,
        cls_weight      = cls_weight,
        cls_dropout     = cls_dropout,
    )

    run_name = f"vqvae-cls{num_classes}" if num_classes > 0 else "vqvae-elasticc1-v6"

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath    = cfg.checkpoint_dir,
        filename   = "vqvae-{epoch:02d}-{train/loss:.4f}",
        monitor    = "train/loss",
        mode       = "min",
        save_top_k = 3,
    )
    early_stop = EarlyStopping(
        monitor  = "train/loss",
        patience = 10,
        mode     = "min",
    )

    logger = WandbLogger(
        project  = "vqvae-supernovas",
        name     = run_name,
        save_dir = cfg.log_dir,
    )

    trainer = L.Trainer(
        max_epochs              = cfg.num_epochs,
        accelerator             = "gpu",
        devices                 = 1,
        precision               = "16-mixed",
        gradient_clip_val       = 1.0,
        accumulate_grad_batches = 2,
        callbacks               = [checkpoint, early_stop],
        logger                  = logger,
        log_every_n_steps       = 200,
        val_check_interval      = 1.0,
        num_sanity_val_steps    = 2,
    )

    trainer.fit(model, dm)
    print(f"Mejor checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
