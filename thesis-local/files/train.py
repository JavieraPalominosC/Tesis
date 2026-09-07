"""
Script de entrenamiento del VQ-VAE para imágenes 2grid de supernovas.

Uso:
    python train_vqvae.py

O con argumentos:
    python train_vqvae.py --epochs 100 --batch_size 32 --latent_dim 256
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
from omegaconf import OmegaConf
import mlflow
import logging

from model import VQVAE
from dataset import SupernovaImageDataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_recon = total_vq = 0
    
    for batch_idx, (images, labels, snids) in enumerate(loader):
        images = images.to(device)
        
        optimizer.zero_grad()
        x_hat, vq_loss, indices = model(images)
        
        # Pérdida de reconstrucción (MSE)
        recon_loss = F.mse_loss(x_hat, images)
        
        # Pérdida total
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        
        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_vq    += vq_loss.item()
        
        if batch_idx % 50 == 0:
            log.info(f"  Batch {batch_idx}/{len(loader)} | "
                     f"loss={loss.item():.4f} | "
                     f"recon={recon_loss.item():.4f} | "
                     f"vq={vq_loss.item():.4f}")
    
    n = len(loader)
    return total_loss/n, total_recon/n, total_vq/n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = total_recon = total_vq = 0
    
    for images, labels, snids in loader:
        images = images.to(device)
        x_hat, vq_loss, indices = model(images)
        recon_loss = F.mse_loss(x_hat, images)
        loss = recon_loss + vq_loss
        
        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_vq    += vq_loss.item()
    
    n = len(loader)
    return total_loss/n, total_recon/n, total_vq/n


def save_reconstruction_samples(model, loader, device, save_path, n_samples=8):
    """Guarda comparaciones imagen original vs reconstruida."""
    model.eval()
    images, _, _ = next(iter(loader))
    images = images[:n_samples].to(device)
    
    with torch.no_grad():
        x_hat, _, _ = model(images)
    
    # Intercalar originales y reconstrucciones
    comparison = torch.cat([images, x_hat], dim=0)
    save_image(comparison, save_path, nrow=n_samples)
    log.info(f"  Imágenes guardadas en {save_path}")


def main(args):
    # ── Configuración ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Usando device: {device}")
    
    # Crear directorios
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'samples').mkdir(parents=True, exist_ok=True)
    
    # Cargar configs del repositorio VT_Model
    config = OmegaConf.to_container(
        OmegaConf.load('configs/online/run_config.yaml'), resolve=True
    )
    dataset_config = OmegaConf.to_container(
        OmegaConf.load('configs/datasets_config.yaml'), resolve=True
    )['elasticc_1']
    
    # ── Dataset y DataLoaders ───────────────────────────────────────
    log.info("Cargando datos...")
    
    train_dataset = SupernovaImageDataset(
        data_dir=args.data_dir,
        config=config,
        dataset_config=dataset_config,
        split='train',
        partition_path=args.partition_path,
        img_size=(args.img_size, args.img_size),
        save_dir=args.img_save_dir,
        max_samples=args.max_samples,
    )
    val_dataset = SupernovaImageDataset(
        data_dir=args.data_dir,
        config=config,
        dataset_config=dataset_config,
        split='val',
        partition_path=args.partition_path,
        img_size=(args.img_size, args.img_size),
        save_dir=args.img_save_dir,
        max_samples=args.max_samples,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # ── Modelo ──────────────────────────────────────────────────────
    model = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        hidden_dims=[64, 128, 256],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parámetros del modelo: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    # ── MLflow tracking ─────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file:{output_dir}/ml-runs")
    mlflow.set_experiment("vqvae_supernova")
    
    with mlflow.start_run(run_name="vqvae_2grid"):
        mlflow.log_params(vars(args))
        
        best_val_loss = float('inf')
        
        for epoch in range(1, args.epochs + 1):
            log.info(f"\n═══ Época {epoch}/{args.epochs} ═══")
            
            # Train
            train_loss, train_recon, train_vq = train_one_epoch(
                model, train_loader, optimizer, device
            )
            
            # Validation
            val_loss, val_recon, val_vq = validate(model, val_loader, device)
            scheduler.step(val_loss)
            
            log.info(f"Train → loss={train_loss:.4f} | recon={train_recon:.4f} | vq={train_vq:.4f}")
            log.info(f"Val   → loss={val_loss:.4f}   | recon={val_recon:.4f}   | vq={val_vq:.4f}")
            
            # MLflow logging
            mlflow.log_metrics({
                'train_loss': train_loss, 'train_recon': train_recon, 'train_vq': train_vq,
                'val_loss': val_loss,     'val_recon': val_recon,     'val_vq': val_vq,
            }, step=epoch)
            
            # Guardar muestras de reconstrucción cada 10 épocas
            if epoch % 10 == 0:
                save_reconstruction_samples(
                    model, val_loader, device,
                    output_dir / 'samples' / f'epoch_{epoch:04d}.png'
                )
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(args),
                }, output_dir / 'checkpoints' / 'best_model.pt')
                log.info(f"  ✅ Mejor modelo guardado (val_loss={val_loss:.4f})")
        
        log.info("\n🎉 Entrenamiento completado!")
        mlflow.log_artifact(str(output_dir / 'checkpoints' / 'best_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar VQ-VAE para supernovas')
    
    # Datos
    parser.add_argument('--data_dir',       default='data/lightcurves/elasticc_1/raw')
    parser.add_argument('--partition_path', default='data/lightcurves/elasticc_1/ATAT_partition/partitions_v1.parquet')
    parser.add_argument('--img_save_dir',   default=None,    help='Guardar imágenes generadas en disco')
    parser.add_argument('--img_size',       type=int, default=224)
    parser.add_argument('--max_samples',    type=int, default=None)
    
    # Modelo
    parser.add_argument('--latent_dim',       type=int,   default=256)
    parser.add_argument('--num_embeddings',   type=int,   default=512)
    parser.add_argument('--commitment_cost',  type=float, default=0.25)
    
    # Entrenamiento
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--output_dir',  default='results/vqvae')
    
    args = parser.parse_args()
    main(args)
