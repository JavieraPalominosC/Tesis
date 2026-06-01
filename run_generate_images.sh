#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_images
#SBATCH --output=logs/gen_images_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source venv/bin/activate
python scripts/generate_images.py
