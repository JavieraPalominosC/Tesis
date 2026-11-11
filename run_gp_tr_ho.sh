#!/bin/bash
# GP transientes completos con seis clases excluidas (anomalías reales).
# Folds, tokens y evaluación en main (CPU); VQ-VAE y MaskGIT en mi210 (GPU).
# Los tokens incluyen las excluidas, para puntuarlas después con el MaskGIT.
#
# Límites de tiempo cortos a propósito: ayudan a que el scheduler los meta
# en huecos y a que terminen antes de la mantención del 16 de noviembre.
#
#   ./run_gp_tr_ho.sh [etapa_inicial]    etapas: folds | vqvae | tokens | maskgit

set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-/home/jpalominos/VT_Model_for_LightCurves_Classification}
HIP_LIB=${HIP_LIB:-/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib}
PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"
CFG=configs/vqvae_gp_tr_ho.yaml

DESDE=${1:-folds}
ETAPAS=(folds vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

mkdir -p logs
PREV=""

if activa folds; then
    PREV=$(sbatch --parsable --partition=main --job-name=gp_ho_fold \
        --output="logs/gp_ho_fold_%j.log" --cpus-per-task=2 --mem=8G --time=00:15:00 \
        --wrap="$PRE && python scripts/make_gp_heldout.py --out data/gp_tr_ho")
    echo "  folds   (cpu): $PREV"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=gp_ho_vq \
        --output="logs/gp_ho_vq_%j.log" --cpus-per-task=8 --mem=48G --time=04:00:00 $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_gp.py $CFG")
    echo "  vqvae   (gpu): $PREV"
    VQ=$PREV
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=main --job-name=gp_ho_tok \
        --output="logs/gp_ho_tok_%j.log" --cpus-per-task=16 --mem=16G --time=01:00:00 $(dep "$PREV") \
        --wrap="$PRE && python scripts/generate_tokens_gp.py $CFG")
    echo "  tokens  (cpu): $PREV"
    if [[ -n "${VQ:-}" ]]; then
        EV=$(sbatch --parsable --partition=main --job-name=gp_ho_ev \
            --output="logs/gp_ho_ev_%j.log" --cpus-per-task=8 --mem=16G --time=00:30:00 $(dep "$VQ") \
            --wrap="$PRE && python scripts/eval_vqvae_gp.py $CFG --split val")
        echo "  eval    (cpu): $EV"
    fi
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=gp_ho_mg \
        --output="logs/gp_ho_mg_%j.log" --cpus-per-task=4 --mem=32G --time=04:00:00 $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && MAX_EPOCHS=30 python scripts/train_maskgit_gp.py $CFG aleatoria")
    echo "  maskgit (gpu): $PREV"
fi
