#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=vqvae_train
#SBATCH --output=logs/vqvae_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p results/vqvae/checkpoints
mkdir -p results/vqvae/samples

echo "===== Cargando módulos ====="
module purge
export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:$LD_LIBRARY_PATH

echo "===== Activando venv ====="
source venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "===== GPU ====="
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0))"

echo "===== Entrenando VQ-VAE ====="
python vqvae/train.py \
    --data_dir data/lightcurves/elasticc_1/raw \
    --partition_path data/lightcurves/elasticc_1/ATAT_partition/partitions_v1.parquet \
    --img_save_dir data/lightcurves/elasticc_1/images_2grid \
    --img_size 224 \
    --latent_dim 256 \
    --num_embeddings 512 \
    --commitment_cost 0.25 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_workers 8 \
    --output_dir results/vqvae

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
