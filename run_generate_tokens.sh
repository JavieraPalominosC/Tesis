#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=gen_tokens
#SBATCH --output=logs/gen_tokens_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nodelist=gn005

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/tokens
source venv/bin/activate
python scripts/generate_tokens.py
