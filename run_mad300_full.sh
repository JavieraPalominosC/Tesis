#!/bin/bash
# Pipeline completo con la codificacion ganadora (mad300) sobre las 1.39M curvas.
#
# mad300 gano la comparacion en subconjunto de forma clara: recall 1.00 en df=500
# contra 0.38 de la codificacion actual, y monotonia +0.60 contra +0.09. Aca se
# entrena sobre el dataset entero para tener el modelo definitivo.
#
# Reusa data/enc_cmp/mad300/images, donde ya estan renderizadas 196k de las
# 1.39M; el renderizador saltea las que existen. Los resultados del subconjunto
# quedan intactos en results/enc_cmp/, los nuevos van a results/enc_cmp_full/.
#
#   ./run_mad300_full.sh           # todo
#   ./run_mad300_full.sh vqvae     # desde el VQ-VAE

set -euo pipefail
cd "$(dirname "$0")"

DESDE=${1:-render}
ETAPAS=(render folds vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

RAMA=mad300
ROOT="data/enc_cmp/$RAMA"
TODOS=999999999
REPO=/home/jpalominos/VT_Model_for_LightCurves_Classification
HIP_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib

SN_FILES=(
    "lc_SNIa-SALT2.parquet"       "lc_SNIa-91bg.parquet"
    "lc_SNIax.parquet"            "lc_SNII-Templates.parquet"
    "lc_SNII-NMF.parquet"         "lc_SNII+HostXT_V19.parquet"
    "lc_SNIb-Templates.parquet"   "lc_SNIb+HostXT_V19.parquet"
    "lc_SNIc-Templates.parquet"   "lc_SNIc+HostXT_V19.parquet"
    "lc_SNIcBL+HostXT_V19.parquet" "lc_SNIIb+HostXT_V19.parquet"
    "lc_SNIIn-MOSFIT.parquet"     "lc_SNIIn+HostXT_V19.parquet"
    "lc_SLSN-I+host.parquet"      "lc_SLSN-I_no_host.parquet"
    "lc_PISN.parquet"
)

PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"

mkdir -p logs
PREV=""
PREV_RENDER=""

if activa render; then
    DEP=""
    for i in "${!SN_FILES[@]}"; do
        jid=$(sbatch --parsable --partition=main --job-name="renF_${i}" \
              --output="logs/renF_${i}_%j.log" --cpus-per-task=2 --mem=24G \
              --wrap="$PRE && python scripts/render_subset_enc.py ${SN_FILES[$i]} $i $RAMA $TODOS $ROOT")
        DEP="${DEP}:${jid}"
    done
    echo "  render:  17 jobs (1.39M imagenes, ~30 min cada uno)"
    PREV_RENDER="afterok${DEP}"
fi

if activa folds; then
    D=$([[ -n "$PREV_RENDER" ]] && echo "--dependency=${PREV_RENDER}" || echo "")
    PREV=$(sbatch --parsable --partition=main --job-name=foldF \
           --output="logs/foldF_%j.log" --cpus-per-task=4 --mem=32G $D \
           --wrap="$PRE && python scripts/make_folds_enc.py $ROOT full")
    echo "  folds:   $PREV"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=vqF \
           --output="logs/vqF_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_exp.py configs/vqvae_enc_mad300_full.yaml")
    echo "  vqvae:   $PREV  (~22 h)"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=tokF \
           --output="logs/tokF_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_enc.py $RAMA full")
    echo "  tokens:  $PREV"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mgF \
           --output="logs/mgF_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && export MAX_EPOCHS=25 && python scripts/train_maskgit_enc.py $RAMA full")
    echo "  maskgit: $PREV  (~22 h)"
fi

echo
squeue -u "$USER" -o "%.10i %.12j %.9T %.22E"
