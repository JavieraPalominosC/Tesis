# Pipeline de Generación de Imágenes 2grid — elasticc_1

**Proyecto:** Detección de outliers en curvas de luz de supernovas con VQ-VAE  
**Cluster:** Leftraru HPC (NLHPC Chile), partición `mi210`, nodos gn004/gn005 (AMD MI210)  
**Stack:** Python 3.10, PyTorch ROCm 6.3, PyTorch Lightning 2.4, SLURM

---

## 1. Contexto

El objetivo de esta etapa es convertir las curvas de luz del dataset `elasticc_1` en imágenes 2grid de 256×256 píxeles RGB, que servirán como input al modelo VQ-VAE para detección de outliers.

El dataset `elasticc_1` contiene curvas de luz fotométricas en 6 bandas (u, g, r, i, z, Y) para distintos tipos de objetos astronómicos. Para la tesis solo se usan los archivos de supernovas (17 de los 32 archivos totales).

---

## 2. Formato de las imágenes 2grid

Las imágenes 2grid son gráficos de curvas de luz convertidos a imágenes:

- **Dimensiones:** 256×256 píxeles RGB
- **Origen del tamaño:** `figsize=[2.56, 2.56]` en matplotlib con DPI=100 por defecto
- **Layout:** 6 subplots verticales (6×1), uno por banda fotométrica
- **Eje X:** tiempo MJD, con `xlim` fijo para que todas las supernovas sean comparables
- **Eje Y:** flujo, con `ylim` fijo normalizado entre objetos
- **Banda vacía:** rectángulo blanco
- **Tensor para el modelo:** `(B, 3, 256, 256)`

El código que las genera está en `src/data/processing/create_images.py`, función `create_2grid_images`.

### Bug corregido

La función `create_2grid_images` tenía un error en la línea 65:

```python
# Antes (bug):
color=fig_params['colors'][j+2]

# Después (correcto):
color=fig_params['colors'][j]
```

El offset `+2` causaba un `KeyError: 6` al procesar la banda Y (índice 4), ya que `4+2=6` estaba fuera del rango del diccionario de colores.

---

## 3. Estructura de datos del dataset

Los datos están en el filesystem de los nodos de cómputo (no montados en el nodo de login):

```
data/lightcurves/elasticc_1/raw/
    lc_AGN.parquet
    lc_SNIa-SALT2.parquet     # 211,892 supernovas
    lc_SNIa-91bg.parquet      # 53,414 supernovas
    lc_SNIax.parquet
    lc_SNII-Templates.parquet
    lc_SNII-NMF.parquet
    lc_SNII+HostXT_V19.parquet
    lc_SNIb-Templates.parquet
    lc_SNIb+HostXT_V19.parquet
    lc_SNIc-Templates.parquet
    lc_SNIc+HostXT_V19.parquet
    lc_SNIcBL+HostXT_V19.parquet
    lc_SNIIb+HostXT_V19.parquet
    lc_SNIIn-MOSFIT.parquet
    lc_SNIIn+HostXT_V19.parquet
    lc_SLSN-I+host.parquet
    lc_SLSN-I_no_host.parquet
    lc_PISN.parquet
    ... (otros tipos no-SN)
```

Las imágenes generadas se guardan en:
```
data/images/elasticc_1/2grid/{SNID}.png
```

---

## 4. Scripts de generación

### 4.1 `scripts/generate_one.py`

Script que procesa un solo archivo parquet. Recibe el nombre del archivo como argumento.

```python
import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, '.')
import os
import pandas as pd
from omegaconf import OmegaConf
from src.data.processing.create_images import create_2grid_images
from tqdm import tqdm

parquet_file = f'data/lightcurves/elasticc_1/raw/{sys.argv[1]}'
full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml')['elasticc_1'])

output_dir = 'data/images/elasticc_1/2grid'
os.makedirs(output_dir, exist_ok=True)

print(f'Procesando {sys.argv[1]}...', flush=True)
df = pd.read_parquet(parquet_file)
snids = df['SNID'].unique()
print(f'Total SNIDs: {len(snids)}', flush=True)

for snid in tqdm(snids):
    out_path = f'{output_dir}/{snid}.png'
    if os.path.exists(out_path):
        continue  # retoma desde donde quedó
    obj_df = df[df['SNID'] == snid]
    img = create_2grid_images(obj_df, config, dataset_config)
    img.save(out_path)

print('Listo!', flush=True)
```

### 4.2 `run_generate_images_parallel.sh`

Script bash que lanza un job SLURM independiente por cada archivo de supernovas:

```bash
#!/bin/bash

SN_FILES=(
    "lc_SNIa-SALT2.parquet"
    "lc_SNIa-91bg.parquet"
    "lc_SNIax.parquet"
    "lc_SNII-Templates.parquet"
    "lc_SNII-NMF.parquet"
    "lc_SNII+HostXT_V19.parquet"
    "lc_SNIb-Templates.parquet"
    "lc_SNIb+HostXT_V19.parquet"
    "lc_SNIc-Templates.parquet"
    "lc_SNIc+HostXT_V19.parquet"
    "lc_SNIcBL+HostXT_V19.parquet"
    "lc_SNIIb+HostXT_V19.parquet"
    "lc_SNIIn-MOSFIT.parquet"
    "lc_SNIIn+HostXT_V19.parquet"
    "lc_SLSN-I+host.parquet"
    "lc_SLSN-I_no_host.parquet"
    "lc_PISN.parquet"
)

for f in "${SN_FILES[@]}"; do
    sbatch --partition=mi210 \
           --gres=gpu:1 \
           --job-name=gen_${f%.parquet} \
           --output=logs/gen_${f%.parquet}_%j.log \
           --cpus-per-task=8 \
           --mem=32G \
           --time=12:00:00 \
           --wrap="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate && python scripts/generate_one.py $f"
    echo "Lanzado: $f"
done
```

---

## 5. Problemas encontrados y soluciones

### 5.1 Nodos de cómputo sin internet

**Problema:** Los nodos gn004/gn005 no tienen acceso a internet. El modelo SwinV2 (`microsoft/swinv2-tiny-patch4-window16-256`) intentaba descargarse desde HuggingFace en tiempo de ejecución y fallaba.

**Solución:** Descargar el modelo desde el nodo de login (que sí tiene internet), y apuntar el config a la ruta local del snapshot:

```bash
# Descargar desde el nodo de login
python -c "
from transformers import Swinv2Model, AutoImageProcessor
Swinv2Model.from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')
AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')
"

# Copiar a carpeta local del proyecto
mkdir -p models/swinv2
cp -L ~/.cache/huggingface/hub/models--microsoft--swinv2-tiny-patch4-window16-256/snapshots/f4d3075206f2ad5eda586c30d6b4d0500f312421/* models/swinv2/
```

Luego en `configs/online/run_config.yaml`:
```yaml
pretrained_model:
  path: "models/swinv2"
```

### 5.2 Datos no montados en el nodo de login

**Problema:** Los datos parquet solo están disponibles en los nodos de cómputo. El nodo de login no tiene acceso a `data/lightcurves/`.

**Solución:** Todo el procesamiento de datos debe lanzarse como jobs SLURM en la partición `mi210`. La partición `mi210` requiere obligatoriamente `--gres=gpu:1` aunque no se use GPU.

### 5.3 Particiones `general` y `largemem` incompatibles con el venv

**Problema:** El venv fue compilado para la arquitectura de los nodos `gn004/gn005` (AMD EPYC con instrucciones específicas). Al correr en nodos de otras particiones (`general`, `largemem`) fallaba con `Illegal instruction (core dumped)`.

**Solución:** Usar siempre la partición `mi210` para cualquier código que use este venv.

### 5.4 Velocidad de generación

**Problema inicial:** Generación a ~1 it/s con un solo job secuencial → estimado de 50+ horas.

**Solución:** Lanzar un job SLURM independiente por cada archivo SN. Con jobs en paralelo en gn004 y gn005, la velocidad sube a ~3-4 it/s por job, y múltiples archivos se procesan simultáneamente.

**Nota importante:** El script incluye `if os.path.exists(out_path): continue`, lo que permite relanzar el job si se corta a las 12 horas sin perder el progreso.

---

## 6. Resultado

Al momento de redactar esta documentación se han generado ~856,000 imágenes 2grid de supernovas en:
```
data/images/elasticc_1/2grid/
```

Para el VQ-VAE se estima que con 50,000–100,000 imágenes es suficiente para entrenar un modelo representativo. Se puede usar un subconjunto balanceado entre tipos de SN.

---

## 7. Próximos pasos: VQ-VAE

Con las imágenes generadas, la arquitectura del VQ-VAE a implementar es:

```
Input: (B, 3, 256, 256)
    ↓ Encoder (Conv×3 + ResBlock×2)
    → (B, 256, 32, 32)
    ↓ Codebook (512 embeddings, dim=256)
    → 1024 índices enteros por imagen
    ↓ Decoder (ConvTranspose×3)
    → (B, 3, 256, 256)
    ↓ Transformer prior sobre secuencias de índices
    → anomaly score: -log p(s)
    ↓ Counterfactual: mask + resample
```

Archivos a implementar:
- `src/models/vqvae/encoder.py`
- `src/models/vqvae/decoder.py`
- `src/models/vqvae/codebook.py`
- `src/models/vqvae/model.py` (LightningModule)
- `src/data/vqvae_dataset.py`
- `scripts/train_vqvae.py`
- `configs/vqvae_config.yaml`
