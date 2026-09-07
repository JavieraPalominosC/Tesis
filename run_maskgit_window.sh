#!/bin/bash
# MaskGIT waterfall_global con mascara de ventana 2D contigua.
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=mgit_win
#SBATCH --output=logs/maskgit_window_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs results/maskgit_window/waterfall_global/checkpoints

export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/mpl-${USER}
export WANDB_MODE=disabled
mkdir -p "$MPLCONFIGDIR"

source venv/bin/activate
echo "===== MaskGIT ventana 2D (waterfall_global) ====="
python scripts/train_maskgit_window.py waterfall_global
echo "===== Job terminado: ${SLURM_JOB_ID:-local} ====="
