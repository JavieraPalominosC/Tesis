# Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE y Generación de Ejemplos Contrafactuales

Tesis de Magíster — Javiera Palominos Clarke
Universidad de Chile, Facultad de Ciencias Físicas y Matemáticas

## Descripción

Pipeline de detección de anomalías y generación de contrafactuales para curvas de luz de supernovas del dataset ELAsTiCC-1. Las curvas se renderizan como imágenes 2grid con codificación de flujo `mad300`, un VQ-VAE las cuantiza en tokens, y un prior MaskGIT asigna el score de anomalía y reescribe la región anómala.

## Skill de escritura

Cursor redacta y revisa la tesis con la skill en `.cursor/skills/magister-tesis/`. El paquete instalable, con el mismo contenido, está en `magister-tesis.skill` en la raíz del repo.

| Archivo | Para qué |
|---------|----------|
| `SKILL.md` | Cuándo usarla, en qué capítulo va cada contenido, y el flujo de redacción |
| `references/estilo-humano.md` | Reglas de prosa y voz |
| `references/muestra-javiera.md` | Muestra verificada de su escritura |
| `references/registro-memoria-fcfm.md` | Cómo se discute un resultado y cómo se cierran las conclusiones en una memoria FCFM |
| `references/notacion.md` | Términos y símbolos que ya están fijados |
| `references/dominio.md` | Contexto astronómico, arquitectura y estado del experimento |

Los capítulos en LaTeX viven en `thesis-local/Tesis docs/`. Overleaf no está sincronizado con este repo: las figuras que un `.tex` referencia hay que copiarlas a `thesis-local/Tesis docs/img/` y subirlas a Overleaf aparte.

## Pipeline

Curvas de luz (6 bandas) → imágenes 2grid (256×256, codificación `mad300`) → VQ-VAE → tokens discretos → prior MaskGIT → score de anomalía + contrafactuales

La codificación `mad300` está en `scripts/enc_flujo.py`: el flujo se centra en la mediana, se escala por el MAD y se pasa por `asinh`, con recorte en ±300·MAD.

## Estructura

```
.cursor/skills/magister-tesis/   skill de escritura y sus referencias
thesis-local/Tesis docs/         capítulos LaTeX y figuras usadas en la tesis
src/models/vqvae/                VQ-VAE con EMA, codebook reset y clasificación supervisada
src/models/maskgit/              prior MaskGIT para detección y contrafactuales
src/data/                        dataset y preprocesamiento
scripts/                         entrenamiento, tokens, evaluación y figuras
configs/                         configuraciones de entrenamiento
imagenes/                        figuras y JSON que producen los scripts
```

## Dataset

ELAsTiCC-1: cerca de 1.39 millones de curvas de luz de 17 tipos de supernovas, en seis bandas fotométricas (u, g, r, i, z, Y). El render 2grid produjo 895.354 imágenes.

## Modelos

VQ-VAE: encoder CNN, codebook EMA (512 embeddings, dimensión 256) y decoder CNN. La cabeza de clasificación supervisada sobre el mapa cuantizado usa peso λ = 0.1 (`cls_weight` en los YAML de `configs/`).

MaskGIT: prior sobre secuencias de 1024 tokens (32×32). El score de anomalía es el máximo del NLL por token, con 24 rondas de enmascaramiento.

## Entrenamiento

```
sbatch run_vqvae.sh
sbatch run_maskgit.sh
```

## Cluster

Leftraru HPC (NLHPC Chile), partición mi210, AMD Instinct MI210.
Stack: Python 3.10, PyTorch ROCm 6.3, PyTorch Lightning 2.4.
