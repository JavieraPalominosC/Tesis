#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot_cmp
#SBATCH --output=logs/plot_cmp_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
python scripts/plot_comparison.py
