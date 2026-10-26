#!/bin/bash
# Ablacion del objetivo 3: rama mad300 SIN cabeza de clasificacion.
#
# Reentrena la cadena completa (VQ-VAE, tokens, prior) sobre las MISMAS imagenes
# y la MISMA particion que la rama mad300 del subconjunto, cambiando solo
# num_classes a 0. Comparando los contrafactuales de ambas ramas se responde si
# la organizacion semantica que induce la clasificacion supervisada mejora la
# interpretabilidad, que es la parte del objetivo 3 que quedo abierta.
#
# El par con clasificacion ya existe y no se vuelve a entrenar:
#   results/enc_cmp/mad300/{vqvae,maskgit}
#
# Uso: ./run_mad300_nocls.sh [etapa_inicial]
#      etapas: vqvae (por defecto) | tokens | maskgit
set -euo pipefail

REPO=/home/jpalominos/VT_Model_for_LightCurves_Classification
HIP_LIB=/opt/rocm/lib
RAMA=mad300
CORRIDA=nocls

DESDE=${1:-vqvae}
ETAPAS=(vqvae tokens maskgit)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"

mkdir -p logs
PREV=""

echo "rama mad300 sin clasificacion (desde: $DESDE)"

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=vqNC \
           --output="logs/vqNC_%j.log" --cpus-per-task=8 --mem=64G \
           --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_exp.py configs/vqvae_enc_${RAMA}_${CORRIDA}.yaml")
    echo "  vqvae:   $PREV  (~5 h)"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=tokNC \
           --output="logs/tokNC_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_enc.py $RAMA $CORRIDA")
    echo "  tokens:  $PREV  (~3 min)"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mgNC \
           --output="logs/mgNC_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/train_maskgit_enc.py $RAMA $CORRIDA aleatoria")
    echo "  maskgit: $PREV  (~2 h, mascara aleatoria como su par con clasificacion)"
fi
