#!/bin/bash
# Piloto GP → VQ-VAE → tokens → MaskGIT en Leftraru.
# Superficie GP as-is, 26 clases balanceadas (cv ~20.8k, test afuera).
#
#   ./run_gp_pilot.sh              # todo desde prepare
#   ./run_gp_pilot.sh vqvae        # desde el VQ-VAE
#   ./run_gp_pilot.sh tokens
#
# Tarde mas corta (opcional):
#   MAX_PER_CLASS=80 ./run_gp_pilot.sh

set -euo pipefail
cd "$(dirname "$0")"

DESDE=${1:-prepare}
ETAPAS=(prepare vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

REPO=${REPO:-/home/jpalominos/VT_Model_for_LightCurves_Classification}
HIP_LIB=${HIP_LIB:-/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib}
PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"

MAX_PER_CLASS=${MAX_PER_CLASS:-}
SURVEY=${SURVEY:-}
KIND=${KIND:-}
PREPARE_ARGS=""
[[ -n "$MAX_PER_CLASS" ]] && PREPARE_ARGS="$PREPARE_ARGS --max-per-class $MAX_PER_CLASS"
[[ -n "$SURVEY" ]] && PREPARE_ARGS="$PREPARE_ARGS --survey $SURVEY"
[[ -n "$KIND" ]] && PREPARE_ARGS="$PREPARE_ARGS --kind $KIND"

mkdir -p logs
PREV=""

if activa prepare; then
    PREV=$(sbatch --parsable --partition=main --job-name="gp_prep" \
        --output="logs/gp_prep_%j.log" --cpus-per-task=8 --mem=64G --time=04:00:00 \
        --wrap="$PRE && python scripts/prepare_gp_pilot.py $PREPARE_ARGS")
    echo "  prepare: $PREV"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_vqvae" \
        --output="logs/gp_vqvae_%j.log" --cpus-per-task=8 --mem=48G --time=06:00:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_gp.py")
    echo "  vqvae:   $PREV"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_tok" \
        --output="logs/gp_tok_%j.log" --cpus-per-task=4 --mem=32G --time=02:00:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_gp.py")
    echo "  tokens:  $PREV"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_mg" \
        --output="logs/gp_mg_%j.log" --cpus-per-task=8 --mem=48G --time=08:00:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && MAX_EPOCHS=30 python scripts/train_maskgit_gp.py aleatoria")
    echo "  maskgit: $PREV"
fi

echo "cadena GP piloto lanzada."
