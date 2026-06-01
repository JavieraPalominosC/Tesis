#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=vqvae_train
#SBATCH --output=logs/vqvae_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=24:00:00
#SBATCH --nodelist=gn005

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/vqvae/checkpoints results/vqvae/logs

export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

source venv/bin/activate

echo "===== GPU ====="
python -c "import torch; print('torch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0))"

echo "===== Entrenando VQ-VAE ====="
python scripts/train_vqvae.py

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
