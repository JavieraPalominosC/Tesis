#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --nodelist=gn004
#SBATCH --gres=gpu:1
#SBATCH --job-name=umap_tokens
#SBATCH --output=logs/umap_tokens_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/umap

echo "===== Activando venv ====="
source venv/bin/activate

echo "===== Verificando tokens ====="
python -c "
import numpy as np
from pathlib import Path

t = np.load('data/tokens/tokens.npy')
print(f'tokens.npy shape: {t.shape}, dtype: {t.dtype}')
print(f'min={t.min()}, max={t.max()}')

# Verificar si existe snids.npy
for p in ['data/tokens/snids.npy', 'data/tokens/snids.csv']:
    if Path(p).exists():
        print(f'Encontrado: {p}')
    else:
        print(f'NO encontrado: {p}')
"

echo "===== Instalando umap-learn si no está ====="
pip install umap-learn --quiet

echo "===== Ejecutando UMAP ====="
python umap_tokens.py \
    --tokens_dir  data/tokens \
    --raw_dir     data/lightcurves/elasticc_1/raw \
    --output_dir  results/umap \
    --max_samples 100000 \
    --n_neighbors 30 \
    --min_dist    0.1

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
