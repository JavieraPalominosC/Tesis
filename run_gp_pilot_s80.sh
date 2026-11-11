#!/bin/bash
# Subset GP (4.160 objetos) → VQ-VAE → tokens → MaskGIT.
# No re-encodea ni toca data/gp_pilot. GPU en mi210.
#
#   ./run_gp_pilot_s80.sh
#   ./run_gp_pilot_s80.sh vqvae

set -euo pipefail
cd "$(dirname "$0")"

DESDE=${1:-subset}
ETAPAS=(subset vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

REPO=${REPO:-/home/jpalominos/VT_Model_for_LightCurves_Classification}
HIP_LIB=${HIP_LIB:-/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib}
PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"
CFG=configs/vqvae_gp_pilot_s80.yaml

mkdir -p logs
PREV=""

if activa subset; then
    PREV=$(sbatch --parsable --partition=main --job-name="gp_s80" \
        --output="logs/gp_s80_%j.log" --cpus-per-task=2 --mem=8G --time=00:20:00 \
        --wrap="$PRE && python scripts/make_gp_subset.py --per-fold 16 --out data/gp_pilot_s80")
    echo "  subset:  $PREV"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_s80_vq" \
        --output="logs/gp_s80_vq_%j.log" --cpus-per-task=4 --mem=32G --time=03:00:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_gp.py $CFG")
    echo "  vqvae:   $PREV"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_s80_tok" \
        --output="logs/gp_s80_tok_%j.log" --cpus-per-task=4 --mem=16G --time=00:40:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_gp.py $CFG")
    echo "  tokens:  $PREV"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="gp_s80_mg" \
        --output="logs/gp_s80_mg_%j.log" --cpus-per-task=4 --mem=32G --time=04:00:00 \
        $(dep "$PREV") \
        --wrap="$PRE && $GPU_ENV && MAX_EPOCHS=30 python scripts/train_maskgit_gp.py $CFG aleatoria")
    echo "  maskgit: $PREV"
fi

echo "cadena GP s80 lanzada."
