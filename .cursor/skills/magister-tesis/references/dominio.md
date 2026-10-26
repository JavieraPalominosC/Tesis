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
- λ=0.1 para el peso de la clasificación (`cls_weight` en los YAML de `configs/`)
- Motivación: inducir organización semántica en el codebook

### Codificación del flujo (decisión posterior, reemplaza la normalización por percentiles)
- La normalización lineal p01–p99 satura: el outlier fija el p99 y por lo tanto su propia escala.
  Medido: p99 se multiplica por 60 entre Δf=10 y Δf=1000; el MAD solo 1.4x.
- Adoptada `mad300`: u = (asinh((f − mediana)/MAD) / asinh(300) + 1) / 2, con recorte en ±300·MAD.
- Se eligió comparando tres ramas (lineal, mad100, mad300) con VQ-VAE y prior propios sobre 196k
  imágenes. `mad100` falla; su VQ-VAE reconstruye y clasifica peor.
- Código único de la codificación: `scripts/enc_flujo.py`.

### Prior MaskGIT
- Opera sobre secuencias de 1024 tokens (32×32)
- Score de anomalía: **máximo** del NLL por token, con R=24 rondas de enmascaramiento (ρ=0.5).
  El promedio NO sirve: diluye una anomalía localizada entre 1024 posiciones.
- Dos priors entrenados, idénticos salvo la forma de la máscara: aleatoria y ventana (rectángulo
  contiguo). Sus val/loss no son comparables entre sí porque cada uno se evalúa con su máscara.
- Hallazgo: lo que manda es la forma con que se **puntúa**, no con la que se entrenó. El prior
  aleatorio puntuado con ventana es la mejor combinación y no requiere reentrenar.

### Contrafactuales
- Región a reescribir: rectángulo de un estadístico de barrido sobre el mapa de NLL, dilatado 2 tokens.
  El criterio por percentil 95 cubre solo el 25% de la anomalía y es peor, salvo en Δf pequeños.
- Remuestreo: 10 pasos de revelado progresivo, muestreo restringido a top-k=10.
- Evaluación en FLUXCAL invirtiendo el render, con extractor de trazas calibrado en 0.73 (el umbral
  más alto que no borra ninguna época de una curva limpia reconstruida).
- El borrado de épocas se cuenta como fracaso, no se excluye. Excluirlo infla la reparación de 36% a >60%.

---

## Estado del experimento (septiembre 2026)

- **VQ-VAE base**: 16 epochs. train/loss=0.0003, recon_loss=0.000216, perplexity≈162–196
- **VQ-VAE con clasificación**: 7 epochs, val acc 55.6%, acc sobre validación completa 38.8%
- **VQ-VAE `mad300` sobre 1.39M**: epoch 23, train/loss=0.0781. macro-F1 0.387
- **Prior MaskGIT `mad300`**: aleatoria val/loss=3.1837 (epoch 16), ventana val/loss=3.6200 (epoch 23)
- **Detección**: recall 1.00 en Δf=500, AUROC 0.997 (prior aleatorio puntuado con ventana),
  sobre 40 SNIa del fold de validación
- **Contrafactuales**: reducción del residuo en FLUXCAL 36.0%, borrado 29.4%, oráculo 42.4%
- **Capítulos escritos**: 3, 4 y 5 actualizados con todo lo anterior

### Pendiente
- Prior entrenado solo con SNIa
- Evaluación sobre clases excluidas del entrenamiento (lo que falta para hablar de utilidad real)
- Verificar por qué la localización cae en Δf=500 (hipótesis: el pico invade la banda vecina)

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
