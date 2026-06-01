# Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE

Tesis de Magíster — Javiera Palominos C.
Universidad de Chile

## Descripción

Pipeline de detección de anomalías y generación de contrafactuales para curvas de luz de supernovas del dataset ELAsTiCC-1, usando un VQ-VAE con representación 2grid y un prior MaskGIT.

## Pipeline

Curvas de luz (6 bandas) → Imágenes 2grid (256x256) → VQ-VAE → Tokens discretos → MaskGIT prior → Anomaly score + Contrafactuales

## Estructura

src/models/vqvae/       - VQ-VAE con EMA, codebook reset, clasificación supervisada
src/models/maskgit/     - Prior MaskGIT para anomaly detection
src/data/               - Dataset y preprocesamiento
scripts/                - Entrenamiento, generación de tokens, análisis
configs/                - Configuraciones

## Dataset

ELAsTiCC-1: ~895,000 supernovas de 17 tipos, 6 bandas fotométricas (u, g, r, i, z, Y).

## Modelos

VQ-VAE: encoder CNN + codebook EMA (512 embeddings, dim=256) + decoder CNN.
Opcionalmente con cabeza de clasificación supervisada sobre z_q (17 clases).

MaskGIT: prior sobre secuencias de 1024 tokens para anomaly scoring
y generación de contrafactuales.

## Entrenamiento

sbatch run_vqvae.sh
sbatch run_maskgit.sh

## Cluster

Leftraru HPC (NLHPC Chile), partición mi210, AMD Instinct MI210.
Stack: Python 3.10, PyTorch ROCm 6.3, PyTorch Lightning 2.4.
