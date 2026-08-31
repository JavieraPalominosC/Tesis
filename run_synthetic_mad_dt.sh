#!/bin/bash
# MAD (df, dt) sobre puntos existentes, waterfall_global.
# CPU: imagenes. GPU: mapas NLL + F1 del maximo.
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

GEN=$(sbatch --parsable --partition=main --job-name=mad_dt_gen \
  --output=logs/mad_dt_gen_%j.log --cpus-per-task=4 --mem=16G --time=01:30:00 \
  --wrap="${WRAP} && python scripts/generate_synthetic_mad_dt.py 40 waterfall_global")
# df extra 20,50,100,200: las imagenes 1-10 ya existen y se saltan.
echo "generacion: job ${GEN} (main)"

EVAL=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mad_dt_eval \
  --dependency=afterok:${GEN} \
  --output=logs/mad_dt_eval_%j.log --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --wrap="${WRAP} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_synthetic_mad_dt.py waterfall_global")
echo "eval:       job ${EVAL} (mi210, afterok ${GEN})"
echo "logs: logs/mad_dt_gen_${GEN}.log  logs/mad_dt_eval_${EVAL}.log"
