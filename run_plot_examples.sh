#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot_ex
#SBATCH --output=logs/plot_examples_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
python scripts/plot_examples_by_type.py
python scripts/plot_comparison.py
