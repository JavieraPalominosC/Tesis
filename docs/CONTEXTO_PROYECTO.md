# Contexto del proyecto — Tesis de Magíster

**Autora:** Javiera Palominos Clarke — FCFM, Universidad de Chile
**Título:** Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE y Generación de Ejemplos Contrafactuales
**Profesor guía:** Francisco Förster
**Última actualización:** 19 agosto 2026

---

## 1. Resumen del pipeline

curvas de luz (parquet) -> imágenes 256x256 RGB -> VQ-VAE -> tokens discretos (1024 por imagen) -> prior MaskGIT -> anomaly score (NLL enmascarado) + contrafactuales

**Dataset:** ELAsTiCC-1, **1.389.147** supernovas, 17 tipos, 6 bandas (u, g, r, i, z, Y).
OJO: textos antiguos de la tesis dicen 895k, hay que actualizarlos a 1.389.147.

**Agrupación en 6 clases para análisis de anomalías:**

| Grupo | Tipos (índices) | Objetos |
|---|---|---|
| SNIa | 0, 1, 2 | 319.033 |
| SNIbc | 6, 7, 8, 9, 10 | 310.328 |
| SNII | 3, 4, 5, 11 | 412.659 |
| SNIIn | 12, 13 | 136.443 |
| SLSN | 14, 15 | 105.238 |
| PISN | 16 | 105.446 |

---

## 2. Infraestructura

**Cluster:** Leftraru HPC (NLHPC). Acceso: `ssh -p 4603 jpalominos@leftraru2.nlhpc.cl` (o leftraru1).

- **Partición `mi210`** (GPU AMD MI210, nodos gn004/gn005): entrenamiento VQ-VAE, MaskGIT, evals. Se pueden correr 3 o más jobs en paralelo.
- **Partición `main`** (CPU, nodos mn0xx): SOLO generación de imágenes (matplotlib no usa GPU). Si generas en mi210, el monitor de GPU-underuse cancela el job.
- venv Python 3.10 (27GB, no borrar), PyTorch ROCm 6.3, Lightning 2.4.
- WandB: proyecto `vqvae-supernovas`, equipo `fforster-uchile`.
- El `LD_LIBRARY_PATH` de ROCm se exporta en cada `--wrap` de sbatch (ver sección 8).

**Cuota de disco:** ampliada a 300GB (temporal, expira con la cuenta el 2026-08-26). Uso actual ~122GB. Al renovar la cuenta hay que pedir explícitamente el almacenamiento en el formulario, si no vuelve al default de 100GB. La renovación la hace el INVESTIGADOR RESPONSABLE (el profe), no la estudiante.

**Gotchas de NLHPC:**

- Cancelar jobs en estado RUNNING drena el nodo por ~1 día. Cancelar PENDING es seguro.
- Reintentar SSH muchas veces tras una caída bloquea tu IP. Hay que esperar o cambiar de red.
- Los parquets solo están montados en nodos de cómputo, no siempre en el nodo de login.
- El monitor de GPU no detecta cargas ROCm, así que puede cancelar jobs que SÍ usan GPU. Evidencia para reclamar: `srun --jobid=XXXX --pty rocm-smi` muestra la GPU al 100%.

---

## 3. Las 4 representaciones de imagen

Todas 256x256 RGB, normalización temporal por objeto, generadas con la técnica rápida (`df.groupby('SNID')` una sola vez más reuso de la figura matplotlib con `ax.clear()`, lo que da ~50 it/s frente a ~1 it/s de la versión ingenua). Los SNID son **strings** en el parquet, no enteros.

| Rep | Layout | Normalización de flujo | Directorio | Script |
|---|---|---|---|---|
| overlay_lines | 6 bandas superpuestas, `'o-'` ms=2 lw=1.0 | GLOBAL (p01-p99 de todas las bandas juntas) | `data/images/elasticc_1/overlay/` | `generate_overlay_fast.py` |
| points | 6 bandas superpuestas, `'o'` ms=3 (sin líneas) | GLOBAL | `overlay_points/` | `generate_overlay_points_fast.py` |
| waterfall | 6 carriles apilados, STEP=1.15 AMP=1.0 | POR BANDA (cada banda su propio p01-p99) | `waterfall/` | `generate_waterfall_fast.py` |
| waterfall_global | 6 carriles apilados, STEP=2.0 AMP=1.8 | GLOBAL | `waterfall_global/` | `generate_waterfall_global_fast.py` |

**Notas importantes:**

- En waterfall, la banda de índice MAYOR (Y) queda ARRIBA y u abajo. Al mapear filas del grid 32x32 a bandas hay que INVERTIR el índice. Cada banda ocupa ~5.33 filas del grid.
- La waterfall fue idea de Javiera, no estaba en la lista original del profe (él pidió overlay, solo-puntos, y grillas separadas).
- Las grillas separadas (el 2grid original) se descartaron del experimento.
- El profe objetó la normalización por-banda porque pierde el flujo relativo entre bandas. Por eso se generó waterfall_global como control.

---

## 4. Checkpoints (rutas exactas)

OJO con las rutas: el `filename` de Lightning incluía `train/loss` o `val/loss`, y ese `/` crea un SUBDIRECTORIO. Además hay `=`. Siempre usar comillas simples al pasarlas por shell.

**VQ-VAE** (todos con config idéntica: hidden [64,128,256], embedding_dim 256, K=512, commitment 0.25, lr 2e-4, batch 32, cls_weight 0.1, num_classes 17, save_top_k=1):

    overlay:          'results/vqvae_overlay_full/checkpoints/vqvae-epoch=17-train/loss=0.1378.ckpt'
    points:           'results/vqvae_points_full/checkpoints/vqvae-epoch=52-train/loss=0.0658.ckpt'
    waterfall:        'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
    waterfall_global: 'results/vqvae_waterfall_global_full/checkpoints/vqvae-epoch=26-train/loss=0.0597.ckpt'

Nota: el de points paró por límite de tiempo (48h) en la época 54; se usa el mejor guardado (época 52). Los otros convergieron por EarlyStopping.

**MaskGIT** (num_tokens 512, seq_len 1024, hidden 256, 4 capas, 4 heads, lr 1e-4, max_epochs 50, EarlyStopping patience 5, save_top_k=1). Son 3.7M params entrenables más 7.7M del VQ-VAE congelado:

    overlay:          'results/maskgit_full/overlay_lines/checkpoints/maskgit-epoch=14-val/loss=2.8207.ckpt'   (18h31)
    points:           'results/maskgit_full/points/checkpoints/maskgit-epoch=13-val/loss=2.9244.ckpt'          (17h37)
    waterfall:        'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'       (1d04h53)
    waterfall_global: 'results/maskgit_full/waterfall_global/checkpoints/maskgit-epoch=20-val/loss=3.6256.ckpt'  (1d00h51, job 11577250)

**Tokens:** `data/tokens/tokens_{rep}_full.npy` (1389147 x 1024, uint16, ~2.8GB cada uno) y `paths_{rep}_full.txt`.

**Para cargar checkpoints del MaskGIT:** `MaskGITWithViz.load_from_checkpoint(ckpt, strict=False)`. El `strict=False` es necesario porque los pesos del VQ-VAE van guardados junto al MaskGIT.

---

## 5. Resultados principales

### 5.1 Separación de anomaly scores (la métrica que decide)

Método: media del NLL sobre tokens enmascarados, mask_ratio 0.5, 10 rondas (estándar de TimeVQVAE-AD). Métrica de separación = (media_max menos media_min) dividido por la dispersión media intra-grupo.

| Rep | Separación | Jerarquía de grupos |
|---|---|---|
| overlay | 0.381 | SNII > PISN > SLSN > SNIIn > SNIa > SNIbc (desordenada) |
| points | 0.421 | PISN > SNII > SLSN > SNIIn > SNIa > SNIbc (desordenada) |
| waterfall | 0.792 | SLSN(3.286) > PISN(3.174) > SNIIn(3.087) > SNII(3.068) > SNIa(3.048) > SNIbc(3.026) |
| **waterfall_global** | **0.858** | SLSN(3.235) > PISN(3.083) > SNIIn(3.058) > SNII(3.052) > SNIa(2.995) > SNIbc(2.980) |

Eval completo 1.389.146 objetos, job 11668859 (4h50, 19 ago 2026). Overlay y points no ordenan los tipos raros arriba. Las dos waterfall sí. Waterfall_global gana la separación y además conserva el flujo relativo entre bandas, que era la objeción del profe.

### 5.2 Accuracy de clasificación (17 clases, validación, muestra de 30k)

Calculada aparte con `scripts/eval_classification.py`. IMPORTANTE: el `val/acc` del entrenamiento salía 0.000 por un artefacto de logging (las métricas están con `on_step=True, on_epoch=False` en `src/models/vqvae/model.py`, así que nunca se agrega la métrica por época). Los números reales son:

| Rep | Accuracy |
|---|---|
| overlay | 0.486 |
| waterfall_global | 0.431 |
| points | 0.392 |
| waterfall (local) | 0.377 |

**Trade-off actualizado:** overlay sigue siendo el mejor clasificador (0.486) pero no separa anomalías. Waterfall local separa bien (0.792) y clasifica peor (0.377). Waterfall_global queda en el medio en clasificación (0.431) y **primero** en separación de anomalías (0.858). Para el objetivo de la tesis (outliers), waterfall_global es la representación a defender: responde la objeción del profe y gana la métrica que decide.

### 5.3 Hallazgo transversal: el loss del MaskGIT no predice la calidad

Waterfall_global tiene el loss MÁS ALTO (3.63), waterfall 3.18, overlay 2.82, y el ranking de separación es el inverso. Ya había pasado en el prototipo (filled tenía menor loss que overlay pero peores contrafactuales). Una representación con más estructura morfológica real es más difícil de predecir, y eso resulta bueno para el prior.

---

## 6. Validación con anomalías sintéticas (ground truth)

Objetivo: inyectar anomalías conocidas y ver si el pipeline las detecta y las localiza.

### 6.1 Lo que NO funcionó

- **Perturbaciones anchas** (gaussiana de ancho 0.20 del span temporal): deforman la curva completa, así que el prior las considera otra curva plausible. Δscore ~0.02, o sea ruido. El profe lo intuyó: "la perturbación es muy larga, tiene un ancho comparable a la curva de luz".
- **Perturbaciones angostas sobre puntos existentes**: el muestreo es irregular, y con ancho 0.02 varios objetos tienen CERO puntos cerca del centro temporal, así que no se modifica nada.

### 6.2 Lo que SÍ funcionó: insertar observaciones nuevas

`scripts/generate_synthetic_insert.py` inserta 3 observaciones nuevas en el centro temporal de la banda r, con amplitudes 2x, 5x y 15x el rango p01-p99 de la banda. Directorio: `data/images/elasticc_1/synthetic_insert/`. Evaluación: `eval_synthetic_insert.py`.

| Amplitud | Δscore | Δ NLL banda r | Δ centro/fuera | Tokens distintos |
|---|---|---|---|---|
| 2x | 0.0431 | 0.1290 | 0.1197 | 316 |
| 5x | 0.0791 | 0.1453 | 0.1151 | 340 |
| 15x | 0.1365 | 0.2904 | 0.1087 | 357 |

Conclusiones:

- **SÍ detecta:** el Δscore crece monótonamente con la amplitud.
- **SÍ atribuye a la banda correcta:** el Δ NLL en banda r es el doble del Δ global en amp15.
- **NO localiza temporalmente:** la razón centro/fuera no escala con la amplitud.
- **Sensibilidad baja:** 0.137 es solo ~0.45 sigmas (la dispersión intra-grupo es ~0.3), y eso con un pulso que visualmente aplasta toda la banda.
- Los contrafactuales SÍ suavizan el pico insertado (ver `cf_synthetic_insert.py`, resultados en `results/synthetic_eval/insert_cf/`).

### 6.3 Protocolo sistemático: un punto × k × banda (pedido del profe)

Los inserts anteriores eran ad hoc (solo banda r, amplitudes 2x/5x/15x del rango). Protocolo:

1. 40 SNIa-SALT2 con ≥8 obs en las 6 bandas.
2. En cada banda, un punto **existente** al azar. El mismo punto se reusa para todos los k.
3. Dos escalas: MAD de la banda, y FLUXCALERR de ese punto.
4. `f ← f + k·escala`, k = 1…10.
5. Normalización del waterfall congelada con p01-p99 del control.
6. Hit = Δscore MaskGIT > 0.3 (subida respecto al control de ESA misma supernova).

Resultado (jobs 11668861 y 11671197, 19 ago 2026): **hit rate = 0 en todas las bandas y todos los k**, con ambas escalas. El Δscore medio llega a ~0.02 en k=10, un orden de magnitud bajo el umbral. Incluso con TAU=0.15 el hit máximo es 0.025. Un solo punto corrido no existe para el prior: el VQ-VAE tokeniza parches de 8×8 píxeles y el punto se diluye. Detectaba el insert de 3 puntos gordos; este protocolo muestra el piso de sensibilidad.

Figuras: `results/synthetic_eval/mad/hitrate_by_band.png` y `results/synthetic_eval/err/hitrate_by_band.png`.

### 6.4 Segundo peak (respuesta al “suaviza otros spikes”)

El insert en el centro se confunde con el máximo de la SN. Aquí se insertan 3 puntos en banda r **después** del máximo real (delay 0.40 del span). 8 SNIa, waterfall + MaskGIT waterfall.

| Amplitud | Δscore | Δ NLL r | Δ peak | Δ 2nd | prec_t | tok dist |
|---|---|---|---|---|---|---|
| 2x | 0.169 | 0.504 | 0.071 | 2.650 | 0.840 | 335 |
| 5x | 0.194 | 0.533 | 0.123 | 2.665 | 0.785 | 358 |
| 15x | 0.221 | 0.524 | −0.113 | 2.802 | 0.799 | 378 |

Contrafactuales (quantile 0.85, 154 tokens editados):

- prec_cf medio = 0.910 (masa |recon−CF| en el extra / (extra+peak))
- fracción del cambio en el peak original = 0.055
- fracción del cambio en el segundo peak = 0.474

Cuando el extra está separado en el tiempo, el CF **no** se pone a suavizar el peak real. El reclamo del profe aplica al caso en que el pulso se parece al máximo canónico, no a un segundo peak tardío.

Scripts: `generate_synthetic_second_peak.py`, `eval_synthetic_second_peak.py`, `cf_synthetic_second_peak.py` (aceptan `waterfall` o `waterfall_global`). Replica en waterfall_global: `run_synthetic_second_peak_wfg.sh`.

---

## 7. Aprendizajes negativos (todos reportables en la tesis)

1. **Nueve variantes del método de score** (mask_ratio 0.15/0.3/0.5 por agregación mean/max/p90, en `explore_score_methods.py`): ninguna supera sustancialmente a mean con ratio 0.5. Las dos mejores empatan alrededor de 0.44 pero con jerarquías OPUESTAS entre sí, o sea es ruido. El método `max` es el peor (0.11 a 0.15) porque el token más improbable de cualquier supernova tiene un NLL alto similar.

2. **Filtrar el fondo blanco del score EMPEORA** la separación (0.555 frente a 0.813). El patrón espacial del vacío aporta información discriminativa. Waterfall tiene 41.5% de tokens con contenido y 58.5% de fondo.

3. **El cuello de botella es el PRIOR, no el VQ-VAE.** Entre control y perturbada cambian entre 27% y 39% de los 1024 tokens, así que el VQ-VAE sí captura la anomalía. El MaskGIT es el que no la considera improbable.

4. **La atribución por banda sobre supernovas reales sale PLANA** (~0.19 por banda frente a 0.167 esperado por azar), y el NLL por token es un moteado uniforme sin estructura espacial. Solo con anomalías sintéticas insertadas aparece atribución real.

5. **Muestrear 12k basta** para comparar representaciones: la separación calculada sobre 12k (0.813) coincide con la del 1.39M completo (0.792). El eval completo tarda ~5h; el de 12k, minutos.

6. **Un punto fotométrico no alcanza.** k=1…10 en MAD o en FLUXCALERR deja Δscore ~0.02. El prior reacciona a deformaciones extendidas (insert de varios puntos, segundo peak), no a un outlier de un dato.

---

## 8. Comandos tipo

Entrenar VQ-VAE:

    sbatch --partition=mi210 --gres=gpu:1 --job-name=vqvae_X \
      --output=logs/vqvae_X_%j.log --cpus-per-task=8 --mem=64G --time=48:00:00 \
      --wrap="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate && export LD_LIBRARY_PATH=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib:\$LD_LIBRARY_PATH && python scripts/train_vqvae_exp.py configs/vqvae_X_full.yaml"

Generar tokens:

    python scripts/generate_tokens_exp.py '<ckpt_vqvae>' '<image_dir>' <sufijo>

MaskGIT, scores y contrafactuales (todos parametrizados por representación):

    python scripts/train_maskgit_full.py {overlay_lines|points|waterfall|waterfall_global}
    python scripts/eval_anomaly_scores_full.py <rep>
    python scripts/generate_counterfactuals_full.py <rep>
    python scripts/compare_representations.py

Generación de imágenes (siempre en `main`, nunca en mi210):

    bash launch_waterfall_full.sh

Verificar que el dataset esté completo antes de entrenar. Esta es una lección aprendida a la mala: 290.178 imágenes quedaron en 0 bytes por jobs cortados, y casi entrenamos sobre un dataset sesgado.

    ls data/images/elasticc_1/<rep> | wc -l
    find data/images/elasticc_1/<rep> -name "*.png" -size 0 | wc -l

El primero debe dar 1389147 y el segundo 0.

---

## 9. Scripts del experimento

Generación de imágenes:

- `generate_overlay_fast.py`, `generate_overlay_points_fast.py`
- `generate_waterfall_fast.py`, `generate_waterfall_global_fast.py`
- `launch_*_full.sh` (17 jobs paralelos, uno por parquet)
- `make_folds_*_full.py` (folds y labels por representación)

Entrenamiento y evaluación:

- `train_vqvae_exp.py` (recibe config YAML)
- `train_maskgit_full.py` (recibe rep)
- `generate_tokens_exp.py`
- `eval_anomaly_scores_full.py`
- `eval_classification.py`
- `compare_representations.py`

Diagnóstico e interpretabilidad:

- `explore_score_methods.py` (9 variantes del score)
- `eval_scores_content_only.py` (score filtrando fondo)
- `interpret_anomaly_waterfall.py` (atribución por banda y perfil temporal)
- `diagnose_content.py` (fracción de tokens con contenido)

Contrafactuales:

- `generate_counterfactuals_full.py` (4 paneles clásicos)
- `generate_counterfactuals_viz.py` (superpone recon frente a CF por banda, heatmap con contraste realzado, quantile configurable; q=0.70 se lee mejor que 0.85)

Anomalías sintéticas:

- `generate_synthetic_insert.py` (inserta pulso, el que funciona)
- `eval_synthetic_insert.py`
- `cf_synthetic_insert.py`
- `generate_synthetic_second_peak.py`, `eval_synthetic_second_peak.py`, `cf_synthetic_second_peak.py`
- `generate_synthetic_mad.py`, `eval_synthetic_mad.py` (hit rate vs k·MAD por banda)
- `generate_synthetic_err.py`, `eval_synthetic_err.py` (igual, escala = FLUXCALERR del punto)

---

## 10. Pendientes

1. Evaluar las **anomalías sintéticas también sobre waterfall_global** (insert, segundo peak, MAD/err). El eval de scores ya está: separación 0.858.
2. **Escribir la tesis:** `04_resultados.tex` y `05_discusion_conclusiones.tex` tienen TODOs. Actualizar todas las referencias de 895k a 1.389.147. Incluir la tabla de 4 representaciones y el piso de sensibilidad (un punto no se detecta; un segundo peak sí se localiza).
3. **Idea abierta:** condicionar el prior POR CLASE, algo como "esto es una SNIa, debería verse así". Tensión metodológica: la tesis es de detección NO SUPERVISADA. Alternativa: usar la clase PREDICHA por la cabeza del VQ-VAE. Conversar con el profe.
4. **Sin resolver:** sensibilidad a outliers de un solo punto, y localización temporal cuando el pulso se superpone al peak. Hipótesis: MaskGIT chico (3.7M, 4 capas), masking al 50% destructivo, o el NLL global no ve cambios locales.

---

## 11. Convenciones de escritura de la tesis

- Español, registro informal. Se tutea al profe y se le dice "profe".
- NO usar em-dashes como sustituto de paréntesis. Es un patrón de escritura de IA a evitar.
- Evitar en la prosa de tesis: bullets, negritas dentro de párrafos, transiciones esquemáticas, aperturas formulaicas.
- LaTeX: usar `\includegraphics` más `\caption` más `\label`. NUNCA `\insertimage`, que tiene un bug de resolución de labels en el template ppizarror v3.4.0.
- El capítulo de Dataset va como sección dentro de Metodología, no como capítulo aparte.
- Template: ppizarror v3.4.0. Archivos: `capitulos/01_introduccion.tex` hasta `05_discusion_conclusiones.tex`.
- Bibliografía en `library.bib`: vqvae2017, maskgit2022, lee2023timevqvae, lee2023timevqvae_ad, moreno2025swinv2.
- Compilación local: MacTeX más LaTeX Workshop en Cursor.
