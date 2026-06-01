#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=maskgit
#SBATCH --output=logs/maskgit_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gn004

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/maskgit/checkpoints results/maskgit/logs

export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:$LD_LIBRARY_PATH

source venv/bin/activate

echo "===== Entrenando MaskGIT ====="
python scripts/train_maskgit.py

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
