#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot_eda
#SBATCH --output=logs/plot_eda_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
python scripts/plot_eda.py
