#!/bin/bash
# CF 5 paneles del experimento MAD (imagenes ya generadas).
# Las imagenes viven en data/images/elasticc_1/synthetic_mad/
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

JOB=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mad_cf \
  --output=logs/mad_cf_%j.log --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --wrap="${WRAP} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/cf_synthetic_mad.py r 10")
echo "cf mad: job ${JOB} (mi210)"
echo "log:    logs/mad_cf_${JOB}.log"
