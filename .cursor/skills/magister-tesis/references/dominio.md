# Conocimiento del dominio — Tesis Javiera Palominos

## El problema astronómico

**Observatorio Vera C. Rubin / LSST**: Telescopio en construcción en Chile que producirá ~10 millones de alertas
fotométricas por noche en seis bandas (u, g, r, i, z, Y). El volumen hace inviable la inspección manual.

**ELAsTiCC-1**: Dataset de simulación para desarrollar y testear algoritmos antes de que el telescopio opere.
Contiene ~1.39 millones de curvas de luz de 17 tipos de supernovas, simuladas con el cadence real de LSST.

**Curvas de luz**: Registro de la variación del flujo fotométrico de un objeto en el tiempo.
Características problemáticas: muestreo irregular (no observaciones en todos los días),
gaps (períodos sin observación), errores heterocedásticos, 6 bandas con cadence distinta.

**Por qué anomalías**: En astronomía de dominio temporal, los objetos raros o desconocidos
son frecuentemente los más interesantes científicamente. Identificarlos requiere saber
qué es "normal" primero.

---

## Decisiones de diseño ya fijadas (no proponer alternativas sin que Javiera lo pida)

### Representación 2grid
- 256×256 píxeles RGB, 6 bandas apiladas verticalmente
- Normalización por objeto al rango [0,1] (tiempo MJD y flujo)
- Normalización para el modelo a [-1,1] (consistente con Tanh del decoder)
- Implementada con matplotlib: figsize=[2.56, 2.56], DPI=100
- Resultado: ~10% de píxeles con contenido (vs ~4% sin normalización)
- 895.354 imágenes generadas en Leftraru HPC (SLURM, un job por tipo)

### VQ-VAE
- Encoder: 3× Conv2d stride=2 (256→128→64→32) + 2× ResBlock
- Codebook: K=512 embeddings de dimensión D=256
- Actualización: EMA con γ=0.99 (no gradientes sobre codebook)
- Codebook reset: cada 10 steps, embeddings con N_k < 1.0 se reinicializan
- Decoder: espejo del encoder con ConvTranspose2d + Tanh
- ~7.7M parámetros entrenables
- Entrenamiento: batch=32, lr=2e-4, Adam, β=0.25, float16 mixta, grad clip=1.0

### Clasificación supervisada
- Cabeza MLP sobre z_q (GlobalAvgPool → Linear(256→128) → ReLU → Dropout → Linear(128→17))
- λ=1.0 para el peso de la clasificación
- Motivación: inducir organización semántica en el codebook

### Prior MaskGIT
- Opera sobre secuencias de 1024 tokens (32×32)
- Score de anomalía: -log p(s)
- Generación de contrafactuales: enmascarar tokens más improbables, remuestrear condicionalmente

---

## Estado del experimento al momento de crear la skill

- **Stage 1 (VQ-VAE base)**: completado, 16 epochs. train/loss=0.0003, recon_loss=0.000216, perplexity≈162-196
- **Stage 1 extendido (con clasificación)**: en curso al crear la skill. Epoch 0 en progreso: acc sube de 5.9% a 31% en 2000 steps, perplexity≈335
- **Stage 2 (MaskGIT)**: pendiente
- **Detección de anomalías y contrafactuales**: pendiente

---

## Exploración previa descartada

Se intentó VQ-VAE directamente sobre series de tiempo (esquema TimeVQVAE).
Problemas: perplexity=1.00 (colapso total) con codebook_dim=8, artefactos en reconstrucciones.
La representación en imágenes resuelve el problema del muestreo irregular y habilita arquitecturas CNN estándar.
Esto ya está documentado en la tesis y no debe proponerse como alternativa futura sin que Javiera lo pida.

---

## Literatura relevante (citas ya en uso)

- van den Oord et al. (2017): VQ-VAE original
- Chang et al. (2022): MaskGIT
- Lee et al. (2023): TimeVQVAE (`\cite{lee2023timevqvae}`)
- ELAsTiCC-1: dataset del Observatorio Vera C. Rubin

---

## Infraestructura de entrenamiento

- Cluster HPC Leftraru (NLHPC, Chile)
- Jobs SLURM
- PyTorch con float16 mixta
- Almacenamiento: `data/images/elasticc_1/2grid/{SNID}.png`
- Split: 5-fold CV estratificado por tipo (StratifiedKFold, random_state=42), fold 0 para entrenamiento principal
  - Train: 718.673 imágenes (80%)
  - Val: 179.669 imágenes (20%)
