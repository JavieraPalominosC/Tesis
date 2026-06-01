#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=eda
#SBATCH --output=logs/eda_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/eda
source venv/bin/activate
python scripts/eda_dataset.py
