#!/bin/bash
# Transientes: VQ-VAE y tokens en CPU (main), MaskGIT en GPU (mi210).
# En main con 32 hilos el VQ-VAE va de 2.5 a 9 min/epoca segun el nodo, asi que
# las 25 epocas pueden tomar hasta ~4 h: el limite queda en 8 h.
#
#   ./run_gp_pilot_tr_cpu.sh

set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-/home/jpalominos/VT_Model_for_LightCurves_Classification}
HIP_LIB=${HIP_LIB:-/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib}
PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"
CFG=configs/vqvae_gp_pilot_tr.yaml

mkdir -p logs

VQ=$(sbatch --parsable --partition=main --job-name="gp_tr_vqC" \
    --output="logs/gp_tr_vqC_%j.log" --cpus-per-task=32 --mem=48G --time=08:00:00 \
    --wrap="$PRE && ACCELERATOR=cpu python scripts/train_vqvae_gp.py $CFG")
echo "  vqvae (cpu):   $VQ"

TOK=$(sbatch --parsable --partition=main --job-name="gp_tr_tokC" \
    --output="logs/gp_tr_tokC_%j.log" --cpus-per-task=16 --mem=16G --time=00:40:00 \
    --dependency=afterok:$VQ \
    --wrap="$PRE && python scripts/generate_tokens_gp.py $CFG")
echo "  tokens (cpu):  $TOK"

MG=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_tr_mg" \
    --output="logs/gp_tr_mg_%j.log" --cpus-per-task=4 --mem=32G --time=04:00:00 \
    --dependency=afterok:$TOK \
    --wrap="$PRE && $GPU_ENV && MAX_EPOCHS=30 python scripts/train_maskgit_gp.py $CFG aleatoria")
echo "  maskgit (gpu): $MG"
