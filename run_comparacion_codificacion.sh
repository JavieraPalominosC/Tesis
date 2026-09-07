#!/bin/bash
# Comparacion pareja de codificaciones de flujo.
#
# Tres ramas identicas en todo (mismas curvas, mismo layout, mismo subconjunto,
# mismos hiperparametros) salvo como se mapea el flujo a la altura del pixel:
#
#   actual   clip((f - p01) / (p99 - p01), 0, 1)
#   mad100   asinh((f - mediana) / MAD) / asinh(100), reescalado a [0,1]
#   mad300   igual pero con techo en 300 MAD
#
# Las dos ramas mad se diferencian en cuantos MAD entran en el carril, que es el
# compromiso entre resolucion para la forma normal (una SNIa ocupa 2.46 tokens
# con actual, 1.69 con mad100, 1.40 con mad300) y techo para el outlier (actual
# se satura en df=50, mad100 aguanta hasta 500, mad300 no se satura).
#
# Cada rama entrena su propio VQ-VAE y su propio MaskGIT, asi que ninguna queda
# fuera de distribucion y la comparacion mide la codificacion, no el desajuste.
#
# Cadena por rama: render (17 jobs CPU) -> folds -> VQ-VAE -> tokens -> MaskGIT
#
# Se puede reanudar desde cualquier etapa sin repetir lo ya hecho:
#   ./run_comparacion_codificacion.sh           # todo desde cero
#   ./run_comparacion_codificacion.sh tokens    # solo tokens y MaskGIT

set -euo pipefail
cd "$(dirname "$0")"

DESDE=${1:-render}
case "$DESDE" in
    render|folds|vqvae|tokens|maskgit) ;;
    *) echo "etapa desconocida: $DESDE (render|folds|vqvae|tokens|maskgit)"; exit 1 ;;
esac

N_POR_CLASE=${N_POR_CLASE:-12000}
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

ETAPAS=(render folds vqvae tokens maskgit)

indice() {
    local n=0
    for e in "${ETAPAS[@]}"; do
        [[ "$e" == "$1" ]] && { echo "$n"; return; }
        n=$((n + 1))
    done
}

etapa_activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }

# Cada etapa espera a la anterior solo si esa etapa tambien se relanzo.
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

for RAMA in actual mad100 mad300; do
    ROOT="data/enc_cmp/$RAMA"
    echo "=== rama $RAMA ==="
    PREV=""

    if etapa_activa render; then
        DEP=""
        for i in "${!SN_FILES[@]}"; do
            f="${SN_FILES[$i]}"
            jid=$(sbatch --parsable --partition=main --job-name="ren_${RAMA}_${i}" \
                  --output="logs/ren_${RAMA}_${i}_%j.log" --cpus-per-task=2 --mem=16G \
                  --wrap="$PRE && python scripts/render_subset_enc.py $f $i $RAMA $N_POR_CLASE $ROOT")
            DEP="${DEP}:${jid}"
        done
        echo "  render:  17 jobs"
        PREV_RENDER="afterok${DEP}"
    fi

    if etapa_activa folds; then
        D=$([[ -n "${PREV_RENDER:-}" ]] && echo "--dependency=${PREV_RENDER}" || echo "")
        PREV=$(sbatch --parsable --partition=main --job-name="fold_$RAMA" \
               --output="logs/fold_${RAMA}_%j.log" --cpus-per-task=2 --mem=16G $D \
               --wrap="$PRE && python scripts/make_folds_enc.py $ROOT")
        echo "  folds:   $PREV"
    fi

    if etapa_activa vqvae; then
        PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="vq_$RAMA" \
               --output="logs/vq_${RAMA}_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
               --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_exp.py configs/vqvae_enc_${RAMA}.yaml")
        echo "  vqvae:   $PREV"
    fi

    if etapa_activa tokens; then
        PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="tok_$RAMA" \
               --output="logs/tok_${RAMA}_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
               --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_enc.py $RAMA")
        echo "  tokens:  $PREV"
    fi

    if etapa_activa maskgit; then
        PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name="mg_$RAMA" \
               --output="logs/mg_${RAMA}_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
               --wrap="$PRE && $GPU_ENV && python scripts/train_maskgit_enc.py $RAMA")
        echo "  maskgit: $PREV"
    fi
    unset PREV_RENDER
done

echo
squeue -u "$USER" -o "%.10i %.14j %.9T %.20E"
