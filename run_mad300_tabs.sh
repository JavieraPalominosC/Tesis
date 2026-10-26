#!/bin/bash
# Ablacion del eje temporal: misma cadena de clases excluidas, otro renderizado.
#
# El diagnostico por rasgo de los baselines dejo una conclusion incomoda: la
# duracion en dias, un solo numero, separa SLSN-I_no_host de las clases normales
# con AUROC 0,895, por encima del 0,883 de la cadena completa. La razon es que
# render_subset_enc.py normaliza el tiempo por objeto, con (mjd - mjd_min)/span,
# de modo que toda curva ocupa el ancho entero de la imagen y la duracion se
# pierde antes de que el VQ-VAE la vea.
#
# Aca se renderiza con una escala de tiempo comun, T_ABS dias para todos, asi
# que la duracion queda escrita en el ancho que ocupa la curva. El precio es
# resolucion de forma para las curvas cortas, y esa es justamente la pregunta
# empirica: medido sobre las 1.39M curvas de ELAsTiCC-1, el span va de 123 d en
# el percentil 1 a 1104 d en el maximo, con mediana 532 d. Con T_ABS = 1110 no
# se recorta ninguna curva, la mediana ocupa el 48% del ancho y el decil mas
# corto baja al 15%.
#
# Todo lo demas es identico a run_mad300_heldout.sh (misma codificacion mad300,
# mismas seis clases fuera, mismos hiperparametros), de modo que la unica
# variable sea el eje temporal. Los folds salen de ordenar los mismos nombres de
# archivo con la misma semilla, asi que los objetos de evaluacion coinciden uno
# a uno con los de la rama principal y la comparacion queda pareada.
#
# Solo se entrena el prior de mascara aleatoria: la mejor combinacion medida es
# entrenar con aleatoria y puntuar con ventana, y el puntaje no necesita un
# prior propio.
#
# Uso: ./run_mad300_tabs.sh [etapa_inicial]
#      etapas: render (por defecto) | folds | vqvae | tokens | maskgit | baselines
set -euo pipefail
cd "$(dirname "$0")"

REPO=/home/jpalominos/VT_Model_for_LightCurves_Classification
HIP_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
RAMA=mad300t
CORRIDA=heldout
ROOT="data/enc_cmp/$RAMA"
T_ABS=1110
TODOS=999999999

DESDE=${1:-render}
ETAPAS=(render folds vqvae tokens maskgit baselines)
indice() { local n=0; for e in "${ETAPAS[@]}"; do [[ "$e" == "$1" ]] && { echo "$n"; return; }; n=$((n + 1)); done; }
[[ -n "$(indice "$DESDE")" ]] || { echo "etapa desconocida: $DESDE"; exit 1; }
activa() { [[ $(indice "$1") -ge $(indice "$DESDE") ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

# Mismo orden que run_mad300_full.sh: es el que fija las etiquetas 0..16, de las
# que make_folds_heldout.py excluye 1, 2, 11, 14, 15 y 16.
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

echo "eje temporal absoluto, T = $T_ABS d (desde: $DESDE)"

if activa render; then
    DEP=""
    for i in "${!SN_FILES[@]}"; do
        jid=$(sbatch --parsable --partition=main --job-name="renT_${i}" \
              --output="logs/renT_${i}_%j.log" --cpus-per-task=2 --mem=24G \
              --wrap="$PRE && export T_ABS=$T_ABS \
                      && python scripts/render_subset_enc.py ${SN_FILES[$i]} $i mad300 $TODOS $ROOT")
        DEP="${DEP}:${jid}"
    done
    PREV="afterok${DEP}"
    echo "  render:  17 jobs en paralelo (1.39M imagenes, ~30 min cada uno, 22 GB)"
fi

if activa folds; then
    D=$([[ "$PREV" == afterok* ]] && echo "--dependency=$PREV" || dep "$PREV")
    PREV=$(sbatch --parsable --partition=main --job-name=foldT \
           --output="logs/foldT_%j.log" --cpus-per-task=4 --mem=32G $D \
           --wrap="$PRE && python scripts/make_folds_enc.py $ROOT full \
                   && python scripts/make_folds_heldout.py $ROOT")
    echo "  folds:   $PREV  (labels.json y despues el split de clases excluidas)"
fi

if activa vqvae; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=vqT \
           --output="logs/vqT_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/train_vqvae_exp.py configs/vqvae_enc_${RAMA}_${CORRIDA}.yaml")
    echo "  vqvae:   $PREV  (~15 h, 967.639 imagenes, 11 clases)"
fi

if activa tokens; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=tokT \
           --output="logs/tokT_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && python scripts/generate_tokens_enc.py $RAMA $CORRIDA")
    echo "  tokens:  $PREV  (~15 min, sobre las 1.39M incluidas las excluidas)"
fi

if activa maskgit; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mgT \
           --output="logs/mgT_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && export MAX_EPOCHS=25 \
                   && export FOLDS_SPLIT=$ROOT/folds_heldout.json \
                   && export LABELS_FILE=$ROOT/labels_heldout.json \
                   && python scripts/train_maskgit_enc.py $RAMA $CORRIDA aleatoria")
    echo "  maskgit: $PREV  (~13 h, mascara aleatoria)"
fi

if activa baselines; then
    PREV=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=baseT \
           --output="logs/baseT_%j.log" --cpus-per-task=8 --mem=64G $(dep "$PREV") \
           --wrap="$PRE && $GPU_ENV && export RAMA=$RAMA \
                   && python scripts/baselines_heldout.py curva \
                   && python scripts/baselines_heldout.py modelo \
                   && python scripts/baselines_heldout.py resumen")
    echo "  baselines: $PREV  (~1 h, salidas en results/enc_cmp_heldout/eval_$RAMA/)"
fi

echo
squeue -u "$USER" -o "%.10i %.12j %.9T %.22E"
