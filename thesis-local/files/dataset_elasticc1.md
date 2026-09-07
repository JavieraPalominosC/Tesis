# Descripción del Dataset: ELAsTiCC-1

## 3.1 Origen y contexto

El dataset utilizado en este trabajo corresponde al Extended LSST Astronomical Time Series Classification Challenge (ELAsTiCC), específicamente su primera versión (elasticc_1). Este dataset fue desarrollado en el contexto del Large Synoptic Survey Telescope (LSST), actualmente conocido como el Vera C. Rubin Observatory, cuya cámara fotométrica es la más grande jamás construida para astronomía y se encuentra en operación en el Cerro Pachón, Chile.

ELAsTiCC fue diseñado como un desafío de clasificación de series de tiempo astronómicas, con el objetivo de desarrollar y evaluar algoritmos capaces de identificar automáticamente distintos tipos de objetos transitorios y variables a partir de sus curvas de luz fotométricas. El dataset es de acceso público y fue distribuido por la colaboración DESC (Dark Energy Science Collaboration) del Rubin Observatory.

## 3.2 Características del dataset

El dataset elasticc_1 contiene curvas de luz simuladas de objetos astronómicos observados en el sistema fotométrico de LSST, que comprende seis bandas espectrales:

| Banda | Rango espectral aproximado |
|-------|---------------------------|
| u     | Ultravioleta (~320–400 nm) |
| g     | Verde (~400–550 nm)        |
| r     | Rojo (~550–700 nm)         |
| i     | Infrarrojo cercano (~700–820 nm) |
| z     | Infrarrojo cercano (~820–920 nm) |
| Y     | Infrarrojo cercano (~920–1000 nm) |

Cada objeto en el dataset es descrito por una serie temporal de observaciones en estas seis bandas, donde cada observación registra el tiempo de observación en días julinianos modificados (MJD), el flujo fotométrico y su incertidumbre asociada.

## 3.3 Tipos de objetos

El dataset incluye 32 tipos de objetos astronómicos distintos, abarcando tanto objetos transitorios como variables. Para esta tesis se utilizan exclusivamente los 17 tipos correspondientes a supernovas y eventos relacionados:

| Archivo | Tipo | Objetos |
|---------|------|---------|
| lc_SNIa-SALT2.parquet | Supernova tipo Ia (modelo SALT2) | ~211,892 |
| lc_SNIa-91bg.parquet | Supernova tipo Ia 91bg-like | ~53,414 |
| lc_SNIax.parquet | Supernova tipo Iax | — |
| lc_SNII-Templates.parquet | Supernova tipo II (templates) | — |
| lc_SNII-NMF.parquet | Supernova tipo II (NMF) | — |
| lc_SNII+HostXT_V19.parquet | Supernova tipo II con extinción host | — |
| lc_SNIb-Templates.parquet | Supernova tipo Ib (templates) | — |
| lc_SNIb+HostXT_V19.parquet | Supernova tipo Ib con extinción host | — |
| lc_SNIc-Templates.parquet | Supernova tipo Ic (templates) | — |
| lc_SNIc+HostXT_V19.parquet | Supernova tipo Ic con extinción host | — |
| lc_SNIcBL+HostXT_V19.parquet | Supernova tipo Ic BL con extinción host | — |
| lc_SNIIb+HostXT_V19.parquet | Supernova tipo IIb con extinción host | — |
| lc_SNIIn-MOSFIT.parquet | Supernova tipo IIn (MOSFiT) | — |
| lc_SNIIn+HostXT_V19.parquet | Supernova tipo IIn con extinción host | — |
| lc_SLSN-I+host.parquet | Supernova superluminosa tipo I (con host) | — |
| lc_SLSN-I_no_host.parquet | Supernova superluminosa tipo I (sin host) | — |
| lc_PISN.parquet | Supernova de inestabilidad de par (PISN) | — |

En total, el dataset procesado contiene aproximadamente **895,000 supernovas únicas**, identificadas por su campo `SNID`.

## 3.4 Preprocesamiento: imágenes 2grid

Las curvas de luz del dataset no se utilizan directamente como series de tiempo, sino que se convierten en imágenes RGB de 256×256 píxeles mediante una representación denominada **2grid**. Esta representación fue elegida porque permite al modelo VQ-VAE operar sobre imágenes en lugar de series de tiempo irregulares, lo que simplifica la arquitectura y aprovecha las capacidades de las redes convolucionales para detectar patrones espaciales.

### Construcción de la imagen 2grid

Para cada supernova, se genera una imagen con las siguientes características:

- **Dimensiones:** 256×256 píxeles, 3 canales RGB
- **Layout:** 6 subplots verticales (uno por banda), apilados de arriba hacia abajo en el orden u, g, r, i, z, Y
- **Eje X:** tiempo MJD, con límites fijos (`xlim`) para que todas las supernovas sean comparables entre sí
- **Eje Y:** flujo fotométrico, con límites fijos (`ylim`) normalizados
- **Banda vacía:** se representa como un rectángulo blanco
- **Barras de error:** opcionales, controladas por el parámetro `use_err`

El tamaño de 256×256 píxeles surge de `figsize=[2.56, 2.56]` pulgadas con el DPI por defecto de matplotlib (100 DPI).

### Ejemplo de imagen 2grid

```
┌──────────────────────┐
│  banda u (violeta)   │  
├──────────────────────┤
│  banda g (verde)     │  
├──────────────────────┤
│  banda r (rojo)      │  
├──────────────────────┤
│  banda i (naranja)   │  
├──────────────────────┤
│  banda z (azul)      │  
├──────────────────────┤
│  banda Y (celeste)   │  
└──────────────────────┘
     ← tiempo MJD →
```

### Generación en paralelo

Dado el volumen del dataset (~895,000 imágenes), la generación se realizó en paralelo en el cluster Leftraru HPC, lanzando un job SLURM independiente por cada tipo de supernova en la partición `mi210` (nodos AMD MI210). Las imágenes generadas se almacenan en:

```
data/images/elasticc_1/2grid/{SNID}.png
```

## 3.5 Partición train/validación

Para el entrenamiento del VQ-VAE, el dataset de imágenes se divide aleatoriamente en:

- **Training:** 90% (~805,000 imágenes)
- **Validación:** 10% (~90,000 imágenes)

La división se realiza de forma aleatoria sin estratificación por tipo de supernova, ya que el objetivo del VQ-VAE es aprender la distribución general de curvas de luz sin supervisión, no clasificar por tipo.

## 3.6 Normalización

Las imágenes se normalizan al rango [-1, 1] antes de ser procesadas por el VQ-VAE:

```python
T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

Esta normalización es consistente con la función de activación Tanh del decoder, que también produce valores en [-1, 1].
