#!/bin/bash
# Reejecucion de la evaluacion con los priors que comparten particion con el VQ-VAE.
#
# Los priors originales de la rama completa se entrenaron con un random_split
# propio de 80/20, distinto del folds_full.json que uso el VQ-VAE. La seleccion
# de objetos ya se habia corregido exigiendo pertenecer a las dos validaciones,
# pero el prior que los puntuaba seguia siendo el viejo. Con mgFa y mgFv
# reentrenados sobre folds_full.json la inconsistencia desaparece.
#
# PRIOR_TAG=_folds hace tres cosas a la vez: apunta a los checkpoints nuevos,
# desactiva el filtro de split_maskgit, que con particion compartida sobra y solo
# recortaria los candidatos a un quinto, y sufija las salidas para no pisar las
# cifras viejas.
#
# Uso: ./run_eval_folds.sh [id_dependencia[:id2...]]
set -euo pipefail
cd "$(dirname "$0")"

REPO=/home/jpalominos/VT_Model_for_LightCurves_Classification
HIP_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib

DEP=${1:-}
[[ -n "$DEP" ]] && DEP_ARG="--dependency=afterok:$DEP" || DEP_ARG=""

PRE="cd $REPO && source venv/bin/activate && export MPLCONFIGDIR=/tmp/mpl-\$USER"
GPU_ENV="export LD_LIBRARY_PATH=$HIP_LIB:\$LD_LIBRARY_PATH && export PYTORCH_ALLOC_CONF=expandable_segments:True"

mkdir -p logs

JOB=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=evalFO \
      --output="logs/evalFO_%j.log" --cpus-per-task=8 --mem=64G $DEP_ARG \
      --wrap="$PRE && $GPU_ENV && export PRIOR_TAG=_folds \
              && python scripts/eval_final_mad300.py \
              && python scripts/barrido_extendido.py \
              && python scripts/fig_barrido_extendido.py")

echo "evaluacion + barrido + figuras: $JOB"
[[ -n "$DEP" ]] && echo "  encolado tras $DEP"
echo "  salidas: final_mad300_folds.json, barrido_extendido_folds.json,"
echo "           barrido_amplitud_folds.png, barrido_duracion_folds.png"
