import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('results/eda/figures', exist_ok=True)

df = pd.read_csv('results/vqvae/logs/vqvae/version_0/metrics.csv')

# Separar train y val
train = df[df['train/loss'].notna()][['step', 'epoch', 'train/loss', 'train/recon_loss', 'train/vq_loss']]
val = df[df['val/loss'].notna()][['step', 'epoch', 'val/loss']]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss total train vs val
axes[0].plot(train['step'], train['train/loss'], alpha=0.6, linewidth=0.8, color='steelblue', label='Train loss')
if len(val) > 0:
    axes[0].plot(val['step'], val['val/loss'], 'o-', linewidth=1.5, color='tomato', label='Val loss', markersize=4)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Evolución de la loss total')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale('log')

# Plot 2: Componentes de la loss de train
axes[1].plot(train['step'], train['train/recon_loss'], alpha=0.7, linewidth=0.8, color='green', label='Recon loss')
axes[1].plot(train['step'], train['train/vq_loss'], alpha=0.7, linewidth=0.8, color='orange', label='VQ loss')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Loss')
axes[1].set_title('Componentes de la loss (train)')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_yscale('log')

plt.suptitle('Entrenamiento VQ-VAE — elasticc_1', fontsize=13)
plt.tight_layout()
plt.savefig('results/eda/figures/vqvae_loss.png', dpi=150, bbox_inches='tight')
print('Figura guardada: results/eda/figures/vqvae_loss.png')
