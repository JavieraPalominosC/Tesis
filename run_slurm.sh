#!/bin/bash
#SBATCH --partition=mi210
#SBATCH --gres=gpu:1
#SBATCH --job-name=vt_train
#SBATCH --output=logs/vt_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p sample_images

echo "===== Cargando módulos ====="
module purge
export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:$LD_LIBRARY_PATH

echo "===== Activando venv ====="
source venv/bin/activate

# Deshabilitar migración de cache de Transformers
export HUGGINGFACE_HUB_VERBOSITY=debug

echo "===== Python ====="
which python
python --version

echo "===== GPU ====="
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip)"

echo "===== Guardando imágenes de ejemplo ====="
python - << 'PYEOF'
import sys
sys.path.insert(0, '.')
import yaml
import pandas as pd
from omegaconf import OmegaConf
from src.data.processing.create_images import create_2grid_images

# Cargar configs
config = OmegaConf.load('configs/online/run_config.yaml')
dataset_config = OmegaConf.load('configs/datasets_config.yaml')['elasticc_1']

# Cargar algunos datos
df = pd.read_parquet('data/lightcurves/elasticc_1/raw/chunk_01.parquet')
ids = df['SNID'].unique()[:5]

for i, snid in enumerate(ids):
    obj_df = df[df['SNID'] == snid]
    img = create_2grid_images(obj_df, OmegaConf.to_container(config, resolve=True), OmegaConf.to_container(dataset_config, resolve=True))
    img.save(f'sample_images/sample_{i}_{snid}.png')
    print(f'Saved sample_{i}_{snid}.png')

print('Done! Images saved in sample_images/')
PYEOF

echo "===== Verificando modelo ====="
ls /home/jpalominos/.cache/huggingface/hub/models--microsoft--swinv2-tiny-patch4-window16-256/snapshots/f4d3075206f2ad5eda586c30d6b4d0500f312421/
echo "===== Lanzando entrenamiento ====="
python -m scripts.run_online \
  ft_classification.loader.name_dataset='elasticc_1' \
  ft_classification.loader.spc=500 \
  ft_classification.imgs_params.input_type='2grid'

echo "===== Job terminado: ${SLURM_JOB_ID} ====="
