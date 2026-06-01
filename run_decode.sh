#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=decode_cb
#SBATCH --output=logs/decode_cb_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/codebook

source venv/bin/activate

echo "===== Decodificando codebook ====="
python decode_codebook.py \
    --checkpoint  results/vqvae/checkpoints/vqvae-epoch=06-train/loss=0.0003.ckpt \
    --tokens_path data/tokens/tokens.npy \
    --output_dir  results/codebook \
    --grid_size   8 \
    --batch_size  16

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
