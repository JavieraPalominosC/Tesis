"""
Detección de outliers y generación de ejemplos contrafactuales
usando el VQ-VAE entrenado.

Basado en:
- TimeVQVAE-AD (arXiv:2311.12550): anomaly score via masked token prediction
- VQ-VAE-2 anomaly detection

Pipeline de anomaly detection:
    1. Encoder: imagen → tokens s = {s_1, ..., s_N}
    2. Calcular anomaly score: error de reconstrucción + log-likelihood bajo prior
    3. Umbral de decisión: percentil sobre el conjunto de validación normal

Pipeline de counterfactual:
    1. Identificar tokens anómalos (error local alto)
    2. Reemplazar tokens anómalos por los más cercanos del codebook
       que maximicen la verosimilitud bajo el prior
    3. Decodificar → imagen contrafactual
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


class AnomalyDetector:
    """
    Detecta outliers en imágenes 2grid usando el VQ-VAE entrenado.
    
    El anomaly score combina:
        1. Error de reconstrucción:  ||x - x_hat||^2
        2. (Opcional) Log-likelihood bajo el prior del codebook
    
    Args:
        model:      VQ-VAE entrenado
        device:     'cuda' o 'cpu'
        threshold:  Percentil para definir el umbral de anomalía (ej: 95)
    """
    
    def __init__(self, model, device='cuda', threshold_percentile=95):
        self.model = model
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None  # Se calcula con fit()
        
        self.model.eval()
        self.model.to(device)
    
    def fit(self, normal_loader):
        """
        Calcula el umbral de anomalía usando datos normales del conjunto de validación.
        
        Args:
            normal_loader: DataLoader con muestras normales (train o val)
        
        Returns:
            scores: array de anomaly scores para las muestras normales
        """
        scores = []
        with torch.no_grad():
            for images, labels, snids in normal_loader:
                images = images.to(self.device)
                batch_scores = self.model.reconstruction_error(images)
                scores.extend(batch_scores.cpu().numpy().tolist())
        
        scores = np.array(scores)
        self.threshold = np.percentile(scores, self.threshold_percentile)
        
        print(f"[AnomalyDetector] Umbral calculado ({self.threshold_percentile}°percentil): "
              f"{self.threshold:.6f}")
        print(f"[AnomalyDetector] Score medio normal: {scores.mean():.6f} ± {scores.std():.6f}")
        
        return scores
    
    def score(self, images):
        """
        Calcula el anomaly score para un batch de imágenes.
        
        Args:
            images: tensor (B, 3, H, W)
        
        Returns:
            scores: (B,) anomaly scores
        """
        images = images.to(self.device)
        with torch.no_grad():
            scores = self.model.reconstruction_error(images)
        return scores.cpu().numpy()
    
    def predict(self, images):
        """
        Predice si cada imagen es una anomalía.
        
        Returns:
            is_anomaly: (B,) bool array
            scores:     (B,) anomaly scores
        """
        if self.threshold is None:
            raise RuntimeError("Llamar fit() antes de predict()")
        
        scores = self.score(images)
        is_anomaly = scores > self.threshold
        return is_anomaly, scores
    
    def score_per_band(self, images):
        """
        Calcula el error de reconstrucción por banda (fila de la imagen 2grid).
        
        Las 6 bandas están apiladas verticalmente → dividir imagen en 6 franjas iguales.
        
        Args:
            images: (B, 3, H, W)
        
        Returns:
            band_scores: (B, 6) error por banda
        """
        images = images.to(self.device)
        with torch.no_grad():
            x_hat, _, _ = self.model(images)
        
        B, C, H, W = images.shape
        band_h = H // 6  # altura de cada franja de banda
        
        band_scores = torch.zeros(B, 6)
        for b in range(6):
            start = b * band_h
            end = (b + 1) * band_h
            error = F.mse_loss(
                x_hat[:, :, start:end, :],
                images[:, :, start:end, :],
                reduction='none'
            )
            band_scores[:, b] = error.mean(dim=[1, 2, 3]).cpu()
        
        return band_scores.numpy()
    
    def evaluate(self, test_loader, labels_map=None):
        """
        Evalúa el detector sobre el conjunto de test.
        
        Returns:
            dict con métricas: AUROC, precision, recall, f1
        """
        from sklearn.metrics import roc_auc_score, classification_report
        
        all_scores, all_labels = [], []
        
        with torch.no_grad():
            for images, labels, snids in test_loader:
                scores = self.score(images)
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.numpy().tolist())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Asumir que label != -1 y != clase_normal indica anomalía
        # Ajustar según la convención de tu dataset
        binary_labels = (all_labels != 0).astype(int)  # 0 = normal, resto = anomalía
        predictions = (all_scores > self.threshold).astype(int)
        
        auroc = roc_auc_score(binary_labels, all_scores)
        report = classification_report(binary_labels, predictions,
                                       target_names=['normal', 'anomaly'])
        
        print(f"\n[Evaluación] AUROC: {auroc:.4f}")
        print(report)
        
        return {'auroc': auroc, 'scores': all_scores, 'labels': all_labels}


class CounterfactualGenerator:
    """
    Genera ejemplos contrafactuales para anomalías detectadas.
    
    El contrafactual de una imagen anómala x es una imagen x_cf que:
        1. Es "normal" según el modelo (bajo anomaly score)
        2. Es lo más similar posible a x
    
    Estrategia (basada en 2311.12550):
        - Identificar tokens anómalos en el espacio del codebook
        - Reemplazar tokens anómalos por sus vecinos más cercanos "normales"
        - Decodificar la secuencia modificada
    
    Args:
        model:    VQ-VAE entrenado
        detector: AnomalyDetector ya ajustado con fit()
        device:   dispositivo
    """
    
    def __init__(self, model, detector, device='cuda'):
        self.model = model
        self.detector = detector
        self.device = device
        
        self.model.eval()
    
    def generate(self, image):
        """
        Genera un contrafactual para una imagen anómala.
        
        Args:
            image: tensor (1, 3, H, W) - imagen anómala individual
        
        Returns:
            x_cf:        (1, 3, H, W) - imagen contrafactual
            band_scores: (6,)         - error por banda (para visualización)
        """
        image = image.to(self.device)
        
        with torch.no_grad():
            # 1. Codificar → obtener tokens
            z_e = self.model.encoder(image)
            z_q, _, indices = self.model.vq(z_e)
            
            # 2. Reconstruir y calcular error por posición espacial
            x_hat = self.model.decoder(z_q)
            pixel_error = F.mse_loss(x_hat, image, reduction='none')  # (1, 3, H, W)
            
            # 3. Error por token (mapear error de píxeles al espacio de tokens)
            B, C, H, W = image.shape
            _, _, Hq, Wq = z_q.shape
            
            # Reducir error a espacio de tokens
            token_error = F.adaptive_avg_pool2d(
                pixel_error.mean(dim=1, keepdim=True),  # (1, 1, H, W)
                (Hq, Wq)
            ).squeeze()  # (Hq, Wq)
            
            # 4. Identificar tokens anómalos (error > percentil 75)
            threshold = torch.quantile(token_error, 0.75)
            anomalous_mask = token_error > threshold  # (Hq, Wq)
            
            # 5. Reemplazar tokens anómalos por el token del codebook
            #    más frecuente en imágenes normales (aproximación simple)
            #    → Usar el token mediano como "token normal"
            normal_token_idx = int(indices.median().item())
            
            new_indices = indices.clone()  # (1, Hq, Wq)
            new_indices[0, anomalous_mask] = normal_token_idx
            
            # 6. Decodificar desde tokens modificados
            x_cf = self.model.decode_from_indices(new_indices)
        
        # Error por banda para visualización
        band_scores = self.detector.score_per_band(image)
        
        return x_cf, band_scores[0]
    
    def visualize(self, image, x_cf, band_scores, snid=None, save_path=None):
        """
        Visualiza: original | reconstrucción | contrafactual | error por banda.
        """
        image = image.to(self.device)
        
        with torch.no_grad():
            x_hat, _, _ = self.model(image)
        
        # Pasar a numpy para visualización
        orig = image[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        recon = x_hat[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        cf   = x_cf[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        fig = plt.figure(figsize=(14, 5))
        fig.suptitle(f"Supernova {snid}" if snid else "Anomaly Analysis", fontsize=13)
        
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
        
        # Imagen original
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(orig)
        ax1.set_title('Original\n(anómala)')
        ax1.axis('off')
        
        # Reconstrucción
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(recon)
        ax2.set_title('Reconstrucción\n(VQ-VAE)')
        ax2.axis('off')
        
        # Contrafactual
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(cf)
        ax3.set_title('Contrafactual\n(normalizado)')
        ax3.axis('off')
        
        # Error por banda
        ax4 = fig.add_subplot(gs[3])
        bands = ['u', 'g', 'r', 'i', 'z', 'Y']
        colors = ['purple', 'green', 'red', 'darkred', 'brown', 'gray']
        bars = ax4.barh(bands, band_scores, color=colors, alpha=0.7)
        ax4.set_xlabel('Error de reconstrucción')
        ax4.set_title('Error por\nbanda')
        ax4.axvline(x=band_scores.mean(), color='black', linestyle='--',
                    linewidth=1, label='Media')
        ax4.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Guardado: {save_path}")
        
        return fig


def run_anomaly_detection(model_path, data_dir, output_dir, config, dataset_config,
                          batch_size=32, img_size=224, num_workers=4):
    """
    Pipeline completo: carga el modelo → detecta outliers → genera contrafactuales.
    
    Args:
        model_path:     Path al checkpoint del VQ-VAE entrenado
        data_dir:       Directorio de datos crudos
        output_dir:     Directorio para guardar resultados
        config:         Configuración del modelo
        dataset_config: Configuración del dataset
    """
    import sys
    sys.path.insert(0, '.')
    from model import VQVAE
    from dataset import SupernovaImageDataset
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    (output_dir / 'counterfactuals').mkdir(parents=True, exist_ok=True)
    
    # ── Cargar modelo ──────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']
    
    model = VQVAE(
        in_channels=3,
        latent_dim=args['latent_dim'],
        num_embeddings=args['num_embeddings'],
        commitment_cost=args['commitment_cost'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[Pipeline] Modelo cargado desde {model_path}")
    
    # ── DataLoaders ────────────────────────────────────────────────
    val_dataset  = SupernovaImageDataset(data_dir, config, dataset_config,
                                         split='val', img_size=(img_size, img_size))
    test_dataset = SupernovaImageDataset(data_dir, config, dataset_config,
                                          split='test', img_size=(img_size, img_size))
    
    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # ── Anomaly Detector ───────────────────────────────────────────
    detector = AnomalyDetector(model, device=device, threshold_percentile=95)
    normal_scores = detector.fit(val_loader)
    
    metrics = detector.evaluate(test_loader)
    
    # Guardar scores
    np.save(output_dir / 'anomaly_scores.npy', metrics['scores'])
    np.save(output_dir / 'labels.npy', metrics['labels'])
    print(f"[Pipeline] Scores guardados en {output_dir}")
    
    # ── Counterfactual Generator ───────────────────────────────────
    cf_gen = CounterfactualGenerator(model, detector, device=device)
    
    # Generar contrafactuales para las N anomalías más severas
    anomaly_indices = np.argsort(metrics['scores'])[::-1][:20]  # top 20 anomalías
    
    print(f"\n[Pipeline] Generando contrafactuales para las 20 anomalías más severas...")
    for rank, idx in enumerate(anomaly_indices):
        img, label, snid = test_dataset[idx]
        img = img.unsqueeze(0)  # (1, 3, H, W)
        
        x_cf, band_scores = cf_gen.generate(img)
        
        cf_gen.visualize(
            img, x_cf, band_scores, snid=snid,
            save_path=output_dir / 'counterfactuals' / f'rank{rank+1:02d}_{snid}.png'
        )
    
    print(f"\n🎉 Pipeline completado. Resultados en {output_dir}")
    return metrics
