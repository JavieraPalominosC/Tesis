#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=debug
#SBATCH --output=logs/debug_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
echo "PWD: $(pwd)"
echo "HOME: $HOME"
ls data/lightcurves/elasticc_1/raw/ | head -5
