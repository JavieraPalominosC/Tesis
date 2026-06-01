#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=create_folds
#SBATCH --output=logs/create_folds_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/folds
source venv/bin/activate
python scripts/create_folds.py
