#!/bin/bash
# Experimento de clases excluidas, pedido en la reunion del 28 de septiembre.
#
# Reentrena la cadena mad300 dejando seis clases completamente fuera, para
# poder usarlas despues como anomalias reales en vez de perturbaciones
# sinteticas: SNIa-91bg, SNIax, SNIIb+HostXT_V19, SLSN-I+host, SLSN-I_no_host
# y PISN. Son 421.508 imagenes, el 30% del dataset.
#
# No se vuelve a renderizar: las imagenes mad300 ya existen para las 1.39M y la
# codificacion no cambia, asi que lo unico distinto es que subconjunto entra al
# entrenamiento.
#
# Los tokens SI se generan sobre las 1.39M, porque hacen falta los de las clases
# excluidas para poder puntuarlas despues. El prior se restringe a las clases
# vistas con FOLDS_SPLIT, que ademas le hace compartir la particion del VQ-VAE.
#
# Uso: ./run_mad300_heldout.sh [etapa_inicial]
#      etapas: folds (por defecto) | vqvae | tokens | maskgit
set -euo pipefail
cd "$(dirname "$0")"

REPO=/home/jpalominos/VT_Model_for_LightCurves_Classification
HIP_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
RAMA=mad300
CORRIDA=heldout
ROOT="data/enc_cmp/$RAMA"

DESDE=${1:-folds}
ETAPAS=(folds vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"

mkdir -p logs
PREV=""

echo "clases excluidas (desde: $DESDE)"

if activa folds; then
    PREV=$(sbatch --parsable --partition=main --job-name=foldHO \
           --output="logs/foldHO_%j.log" --cpus-per-task=4 --mem=32G \
           --wrap="$PRE && python scripts/make_folds_heldout.py $ROOT")
    echo "  folds:   $PREV"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=vqHO \
           --output="logs/vqHO_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_exp.py configs/vqvae_enc_${RAMA}_${CORRIDA}.yaml")
    echo "  vqvae:   $PREV  (~15 h, 967.639 imagenes, 11 clases)"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=tokHO \
           --output="logs/tokHO_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_enc.py $RAMA $CORRIDA")
    echo "  tokens:  $PREV  (~15 min, sobre las 1.39M incluidas las excluidas)"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mgHO \
           --output="logs/mgHO_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && export MAX_EPOCHS=25 \
                   && export FOLDS_SPLIT=$ROOT/folds_heldout.json \
                   && export LABELS_FILE=$ROOT/labels_heldout.json \
                   && python scripts/train_maskgit_enc.py $RAMA $CORRIDA aleatoria")
    echo "  maskgit: $PREV  (~13 h, mascara aleatoria)"
fi
