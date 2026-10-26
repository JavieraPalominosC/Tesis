# Contexto del proyecto de tesis

Documento de traspaso autocontenido. Resume la tesis, el pipeline, el código
relevante, los resultados numéricos y el estado actual del trabajo.

---

## 1. La tesis

**Título:** Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE y
Generación de Ejemplos Contrafactuales.

**Autora:** Javiera Palominos Clarke, Magíster, Universidad de Chile, FCFM.

**Idea central.** Las curvas de luz de supernovas se renderizan como imágenes,
se comprimen con un VQ-VAE a una grilla de tokens discretos, y se entrena un
modelo generativo (MaskGIT) sobre esas secuencias de tokens. Ese segundo modelo
aprende qué configuraciones de tokens son plausibles, y un objeto es anómalo
cuando el modelo no logra predecirlo. Además se generan ejemplos
contrafactuales: la mínima modificación que vuelve normal a un objeto anómalo.

**Datos.** ELAsTiCC. El dataset completo son 1.389.147 curvas de luz, en 6
bandas fotométricas (u, g, r, i, z, Y), con 17 clases de transientes.

---

## 2. Arquitectura del pipeline

Es un modelo de dos etapas, el esquema clásico de VQ-VAE + prior.

### Etapa 1: VQ-VAE

Comprime cada imagen de 256×256×3 a una grilla de 32×32 tokens discretos.

- Encoder `E_φ`, decoder `G_θ`, y una cabeza clasificadora auxiliar `f_ψ`.
- Codebook de **K = 512** vectores, dimensión **D = 256**.
- Grilla latente **H = W = 32**, o sea **T = 1024 tokens** por imagen.
- `commitment_cost` β = 0.25, EMA γ = 0.99.
- La cabeza clasificadora pesa `cls_weight` λ = 0.1. Es auxiliar: empuja al
  codebook a separar clases, pero no es el objetivo.
- 25 épocas, batch 32, lr 2e-4.

Config de referencia: `configs/vqvae_enc_mad300_full.yaml`.
Código: `src/models/vqvae/{model,encoder,decoder,codebook}.py`.

### Etapa 2: prior MaskGIT

Modela `p(s)` sobre las secuencias de 1024 tokens. Transformer bidireccional
estilo BERT: tapa una fracción de los tokens y aprende a predecirlos a partir
de los visibles.

- 8 capas, hidden 512, 8 heads, ff_mult 4, dropout 0.1, lr 1e-4.
- `mask_token = num_tokens = 512`, o sea un símbolo 513 extra. La cabeza de
  salida tiene solo 512 logits, así que MASK nunca es predecible.
- La pérdida se calcula **solo sobre las posiciones enmascaradas**.

Código: `src/models/maskgit/maskgit.py`.

---

## 3. Codificación de las imágenes: `mad300`

Cada curva se dibuja como una imagen de 256×256 px con las 6 bandas apiladas
verticalmente (`u` abajo, `Y` arriba), tiempo en el eje horizontal.

Parámetros del renderizador: `AMP = 1.8`, `STEP = 2.0`, `N_BANDS = 6`. La banda
`k` se dibuja centrada en `k*STEP + AMP/2`, sobre un eje que va de `-0.2` a
`(N_BANDS-1)*STEP + AMP + 0.2`.

### Eje de flujo

```
u = ½ [ clip( arcsinh((f − med(f)) / MAD(f)) / arcsinh(U), −1, 1 ) + 1 ]
```

con **U = 300**. De ahí el nombre `mad300`.

**Por qué arcsinh sobre MAD y no reescalado lineal entre percentiles.** El
reescalado lineal se satura justo donde importa, porque el outlier arrastra la
escala con la que debería medirse. Entre Δf = 10 y Δf = 1000 el `Q_0.99` se
multiplica por 60 mientras el MAD crece 1.4 veces. Con la escala congelada al
control, las imágenes de Δf = 20, 50, 100 y 200 salen idénticas píxel a píxel.
Sin congelar, triplicar la amplitud cambia el 0.44% de la imagen.

**Por qué U = 300 y no U = 100.** Empírico, entrenando el pipeline completo por
separado para cada variante sobre 196.000 imágenes:

| Codificación | Δf=10 | Δf=50 | Δf=100 | Δf=300 | Δf=500 | Spearman |
|---|---|---|---|---|---|---|
| Lineal | 0.10 | 0.41 | 0.33 | 0.23 | 0.38 | +0.09 |
| `mad100` | 0.05 | 0.36 | 0.39 | 0.10 | 0.08 | +0.16 |
| `mad300` | **0.31** | **0.46** | **0.69** | **0.95** | **1.00** | **+0.60** |

`mad100` es peor que la lineal en el extremo alto porque satura igual, solo que
más arriba. U fija cuántos MAD entran en el carril antes de tocar el techo.

**El costo.** Subir U comprime la curva normal. Una SNIa usa el 84% del carril
con U = 5 y solo el **32% con U = 300**. Y U se eligió optimizando el barrido
sintético, donde las anomalías son picos de flujo grandes; para las anomalías
reales (diferencias de forma y duración) nadie probó qué U conviene.

### Eje temporal

Hay dos variantes, y la diferencia resultó importante:

- **`mad300`** (rama principal): `(mjd − mjd_min) / span` por objeto. Toda curva
  ocupa el ancho completo, así que la duración absoluta desaparece de la imagen.
- **`mad300t`** (ablación): `(mjd − mjd_min) / T_ABS` con `T_ABS = 1110` días.
  Escala común a todos los objetos, así que la duración se conserva.

`T_ABS = 1110` se eligió apenas por encima del máximo observado (1104 días) para
que nada se recorte. Un T más chico aprovecharía mejor el ancho pero recortaría
los objetos más largos, que son justo las SLSN que queremos detectar.

Distribución de duraciones medida con `scripts/medir_span.py`: mediana 532 días,
máximo 1104, p90/p10 = 4.64. Ese cociente bajo es lo que hace viable un eje
lineal sin necesidad de compresión logarítmica.

---

## 4. Cómo se puntúa una anomalía

El prior da una NLL por token. Hay que convertir 1024 números en uno.

```python
N_RONDAS, RATIO = 24, 0.5

def nll(t, forma, g):
    """NLL por token promediada sobre N_RONDAS de enmascarado."""
    B, T = t.shape
    n = int(T * RATIO)
    total = torch.zeros(B, T, device=dev)
    veces = torch.zeros(B, T, device=dev)
    for _ in range(N_RONDAS):
        if forma == 'aleatoria':
            mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
            for i in range(B):
                mask[i, torch.randperm(T, device=dev, generator=g)[:n]] = True
        else:
            mask = torch.cat([mascara_ventana(T, n, dev, g) for _ in range(B)])
        masked = t.clone()
        masked[mask] = mg.mask_token
        lp = F.log_softmax(mg(masked), dim=-1)
        total += (-lp.gather(2, t.unsqueeze(-1)).squeeze(-1)) * mask.float()
        veces += mask.float()
    return total / veces.clamp(min=1.0)
```

Hay **dos decisiones independientes**, y pesan mucho:

### Forma de la máscara al puntuar

```python
def mascara_ventana(T, n, dev, g):
    """Rectangulo contiguo de area n, igual que en el entrenamiento."""
    H = W = int(round(math.sqrt(T)))
    w = int(torch.randint(1, W + 1, (1,), device=dev, generator=g).item())
    h = min(H, max(1, (n + w - 1) // w))
    w = min(W, max(1, (n + h - 1) // h))
    r0 = 0 if h == H else int(torch.randint(0, H - h + 1, (1,), ...).item())
    c0 = 0 if w == W else int(torch.randint(0, W - w + 1, (1,), ...).item())
    grid = torch.zeros(H, W, dtype=torch.bool, device=dev)
    grid[r0:r0 + h, c0:c0 + w] = True
    ...
```

Con `T = 1024` y `RATIO = 0.5` se tapan `n = 512` tokens, o sea **la mitad de la
imagen en un único rectángulo contiguo** por ronda. Por la aritmética del área,
las formas posibles van de 32×16 a 16×32. Físicamente: o tapás todas las bandas
en la mitad del rango temporal, o la mitad de las bandas en todo el rango.

- **Aleatoria:** celdas dispersas. Cada token tapado conserva vecinos visibles,
  así que el prior interpola, incluida la anomalía.
- **Ventana:** rectángulo contiguo. El prior tiene que reconstruir sin apoyo
  local y la anomalía sobrevive como sorpresa.

### Estadístico sobre la grilla

- **Máximo** (`m.max(1)`): el token más sorprendente. Correcto cuando la
  anomalía es localizada (el pico sintético inyectado en la banda r).
- **Promedio** (`m.mean(1)`): los 1024. Correcto cuando lo raro es el objeto
  entero (una PISN no tiene un pedazo anómalo, es anómala completa).

### Umbral

τ = percentil 95 de los controles. Eso fija la tasa de falsos positivos en 5%.
En el barrido sintético con 40 objetos eso son FP = 2 siempre, lo que implica
que el **macro F1 tiene piso 0.32 (recall 0) y techo 0.975 (recall 1)**, no 0 y 1.

```python
def piso_macro_f1(n_ctrl, q):
    """Macro F1 cuando el recall es 0, que es el minimo alcanzable."""
    fp = round(n_ctrl * (1 - q))
    tn, fn = n_ctrl - fp, n_ctrl
    p, r = tn / (tn + fn), tn / (tn + fp)
    return 0.5 * (2 * p * r / (p + r))
```

**Ruido estadístico:** con 40 objetos, se(recall) ≈ 0.06, o sea ≈ ±0.04 en macro
F1. Diferencias menores a 0.08 entre celdas no significan nada.

---

## 5. Infraestructura

- Cluster **Leftraru** (NLHPC), acceso por `ssh leftraru`.
- Raíz del proyecto en el cluster:
  `/home/jpalominos/VT_Model_for_LightCurves_Classification`
- Entorno: `source venv/bin/activate` (Python 3.10).
- SLURM: partición `mi210` con `--gres=gpu:1` para GPU, partición `main` para
  CPU. Encadenamiento con `--dependency=afterok:`.
- Cuidado con la memoria: `tokens_heldout.npy` es grande, hay que cargarlo con
  `np.load(..., mmap_mode='r')` o el job muere por OOM. Pedir `--mem=48G`.
- El repo local y Overleaf **no están sincronizados**: las figuras hay que
  subirlas a mano.

---

## 6. La pauta del profe (reunión 28 de septiembre)

Seis puntos. Todos cerrados. Se documenta todo en `docs/semana-28-septiembre.md`
y el informe para el profe está en `docs/avance-pauta-28sep.pdf` (9 páginas,
fuente en `docs/avance-pauta-28sep.tex`).

1. Reentrenar VQ-VAE y MaskGIT **sin** las clases: SNIIb, SLSN-I (2 clases),
   SNIax, SNIa-91bg, PISN.
2. Evaluar MaskGIT con las categorías faltantes.
3. Agregar perturbaciones más grandes en Δt.
4. Ver perturbaciones más chicas en Δf.
5. Agregar baseline de detección de anomalías.
6. Enviar ejemplos de imágenes normalizadas variando n MAD entre 5 y 300
   logarítmicamente.

### Corrección previa importante

Se detectó que **los folds del VQ-VAE y los del MaskGIT no eran los mismos**.
Se corrigió y los puntos 3 y 4 se rehicieron con los priors de folds
compartidos. Los resultados viejos quedaron obsoletos.

### Punto 1 y 2: clases excluidas (held-out)

Se entrenó todo de nuevo sobre las 11 clases vistas (967.639 imágenes, ~15 h de
VQ-VAE), reservando 6 clases como anomalías reales. Evaluación sobre 1996
normales de validación y ~500 objetos de cada clase excluida.

AUROC con el mejor puntaje (`prior_media_ven`, rama `mad300`):

| clase | AUROC | recall @ τ |
|---|---|---|
| SLSN-I sin host | 0.883 | 0.59 |
| SLSN-I con host | 0.794 | 0.44 |
| PISN | 0.679 | 0.27 |
| SNIax | 0.495 | 0.07 |
| SNIa-91bg | 0.499 | 0.05 |
| SNIIb | 0.453 | 0.01 |

Las tres de abajo están en el azar y su recall es la propia tasa de FP del
umbral: el detector las trata igual que a un objeto normal.

### Punto 5: baselines

Promedio de AUROC sobre las 6 clases excluidas:

| método | mad300 | mad300t |
|---|---|---|
| `mad_max` (desviación máx en MAD, una línea de código) | — | 0.458 |
| `iforest` (Isolation Forest sobre estadísticos resumen) | — | 0.626 |
| `recon` (error de reconstrucción del VQ-VAE) | 0.612 | 0.600 |
| `prior_max` | — | 0.578 |
| `prior_media` | 0.609 | 0.638 |
| `prior_max_ven` | 0.587 | 0.633 |
| **`prior_media_ven`** (el método) | **0.634** | **0.657** |

`recon` es la comparación importante: mismo VQ-VAE, mismas imágenes, mismo
codebook, y lo único que cambia es la pregunta. Es la ablación más limpia.

**Dato incómodo:** la **duración de la curva en días, sola, sin modelo**, da
AUROC **0.895** en SLSN-I sin host. Un rasgo trivial gana.

### Tabla de la elección del puntaje

Los cuatro números del prior salen del **mismo checkpoint entrenado**, solo
cambia el scoring. El prior se entrenó con máscara aleatoria; la ventana se usa
únicamente al puntuar.

|  | máximo + aleatoria | promedio + ventana |
|---|---|---|
| las 3 exóticas | 0.548 | **0.785** |
| las 3 sutiles | 0.494 | 0.482 |

Spread entre peor y mejor configuración: **0.113**. Spread entre el prior y el
mejor baseline: **0.022**. **Cómo preguntás pesa cinco veces más que qué modelo
usás**, y es gratis.

### Punto 3: Δt más grande

Barrido extendido hasta 320 días, que cubre ~62% del largo de la curva.
Cobertura real: 58.3% para 160 días, 61.6% para 320.

### Punto 4: Δf más chico

El barrido original partía en Δf = 10. Se agregaron 1, 2, 3, 5, 7.
Máscara de ventana, 1D:

| Δf | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| recall | 0.125 | 0.275 | 0.325 | 0.450 | 0.525 | 0.550 |
| macro F1 | 0.443 | 0.563 | 0.598 | 0.680 | 0.725 | 0.740 |
| AUROC | 0.493 | 0.667 | 0.715 | 0.776 | 0.794 | 0.810 |

Serie completa de macro F1 (ventana): 0.44, 0.56, 0.60, 0.68, 0.73, 0.74, 0.86,
0.90, 0.91, 0.92, 0.95, 0.94, 0.95, 0.97, 0.97 para Δf = 1…500.
Con máscara aleatoria: 0.46, 0.49, 0.53, 0.49, 0.53, 0.53, 0.63, 0.56, 0.58,
0.65, 0.66, 0.81, 0.82, 0.89, 0.94.

**Resultado: el piso de sensibilidad está en Δf ≈ 2 MAD, no en 10.** En Δf = 1
el AUROC es 0.493, azar exacto. Ojo: ahí el macro F1 da 0.44, que parece señal
pero no lo es, porque un detector sin señal daría ~0.37 con este umbral.

---

## 7. Hallazgo adicional: la normalización temporal borra la señal

El resultado que ordena todo lo demás. Que un rasgo trivial (la duración) gane
0.895 no es una derrota sino un diagnóstico: la normalización temporal por
objeto estaba borrando la duración antes de que el modelo la viera.

Se hizo la ablación completa (`mad300t`, eje absoluto, ~30 h de cómputo):

- Prior: **0.634 → 0.657**
- Baseline de reconstrucción: **0.612 → 0.600** (va para el otro lado)

Que el baseline empeore mientras el prior mejora **descarta** la explicación
"las imágenes nuevas son más fáciles". Si lo fueran, subirían los dos.

**Confound que se testeó antes de lanzar:** la ganancia podría venir solo de
contar tokens en blanco. El test era si las tres clases sutiles también
mejoraban. Mejoraron las tres, dos cruzando 0.5, así que no es solo eso.

**Por qué NO se cambió la codificación principal.** La ablación solo cubre la
tarea de held-out, y la representación se degradó en todas las métricas
intrínsecas: perplejidad −64%, reconstrucción +15% peor, accuracy del
clasificador −25%. Eso plausiblemente daña el barrido sintético y los
contrafactuales, que son el resultado principal de la tesis.

**Experimento pendiente que lo decidiría:** correr el barrido sintético Δf/Δt
con los dos ejes. Los dos modelos ya están entrenados y tarda minutos.

---

## 8. Estructura espacial del mapa de NLL

Verificado con `scripts/fig_maskgit_residual.py` sobre 200 normales y 200
SLSN-I sin host.

Los mapas de NLL individuales están llenos de puntos calientes, tanto en
normales como en anómalos. Pero **ese moteado desaparece al promediar**: la
mediana por token sobre 200 objetos es suave. O sea que el moteado no está en
posiciones fijas de la grilla, es variabilidad objeto a objeto.

Las dos medianas comparten la misma estructura gruesa: franjas oscuras (NLL
baja) en las bandas **g** y **u**, que son las peor muestreadas, y en los bordes
izquierdo y derecho. El prior acierta fácil ahí porque el carril está vacío.

La diferencia entre clases es una banda vertical en el centro del rango temporal
(la zona del pico), presente en todas las bandas. Perfil promedio por banda:
u +0.68, g +0.28, r +0.44, i +0.46, z +0.47, Y +0.70.
Amplitud: σ = 0.81 contra σ = 1.20 de la estructura basal.

Correlación del mapa de cada objeto con el patrón basal: **0.62 en normales,
0.46 en SLSN**, con histogramas casi sin solapamiento. Eso es un puntaje
alternativo no explotado todavía.

**Consecuencia práctica:** si el moteado es ruido por objeto y la señal es un
corrimiento suave y global, entonces el máximo está leyendo literalmente el
ruido. Por eso `prior_max_ven` = 0.587 y `prior_media_ven` = 0.634.

---

## 9. Archivos clave

### Scripts (los relevantes de esta etapa)

| archivo | qué hace |
|---|---|
| `scripts/render_subset_enc.py` | Renderiza curvas a imágenes. Variable de entorno `T_ABS` activa el eje temporal absoluto. |
| `scripts/make_folds_enc.py` | Folds normales. |
| `scripts/make_folds_heldout.py` | Folds excluyendo las 6 clases; define `NOMBRES`. |
| `scripts/generate_tokens_enc.py` | Tokeniza el dataset con el VQ-VAE entrenado. |
| `scripts/train_maskgit_enc.py` | Entrena el prior. |
| `scripts/baselines_heldout.py` | Puntos 2 y 5. Tres fases: `curva`, `modelo`, `resumen`. Variable `RAMA`. |
| `scripts/barrido_extendido.py` | Puntos 3 y 4. Define `DFS_1D`, `DFS_2D`, `DTS`. |
| `scripts/fig_barrido_extendido.py` | Figuras del barrido. Genera 4: amplitud, amplitud_f1, duración, duración_f1. |
| `scripts/fig_barrido_umad.py` | Punto 6, figura variando U. |
| `scripts/medir_span.py` | Distribución de duraciones. |
| `scripts/fig_eje_temporal.py` | Comparación visual de los dos renderizados. |
| `scripts/fig_heldout_resumen.py` | Figura resumen de las clases excluidas. |
| `scripts/fig_maskgit_ejemplos.py` | Ejemplos individuales: imagen + mapa de NLL. |
| `scripts/fig_maskgit_residual.py` | Patrón basal vs residuo de clase. |
| `scripts/eval_mascara_cruzada.py` | Define `mascara_ventana` y `mascara_aleatoria`. |
| `scripts/eval_enc_cmp.py` | Utilidades: `mejor_ckpt`, `tf`, `RAW_DIR`, `TARGET_BAND`. |

### Lanzadores

`run_mad300_heldout.sh`, `run_mad300_tabs.sh`, `run_mad300_nocls.sh`,
`run_eval_folds.sh`. El de `tabs` tiene 6 etapas encadenadas
(render / folds / vqvae / tokens / maskgit / baselines) con `T_ABS=1110` y
`RAMA=mad300t`, y entrena solo el prior aleatorio para ahorrar 13 h.

### Configs

`configs/vqvae_enc_mad300_heldout.yaml` y `configs/vqvae_enc_mad300t_heldout.yaml`
(idénticos salvo rutas de imágenes, folds y checkpoints).

### Resultados en el cluster

```
results/enc_cmp_heldout/{mad300,mad300t}/vqvae/checkpoints/
results/enc_cmp_heldout/{mad300,mad300t}/maskgit/checkpoints/
results/enc_cmp_heldout/eval/            # mad300
results/enc_cmp_heldout/eval_mad300t/    # ablacion
    baselines_auroc.json
    rasgos_curva.csv                     # columnas: snid, clase, grupo, rasgos
    puntajes_{recon,prior_max,prior_media,prior_max_ven,prior_media_ven}.json
results/enc_cmp_full/eval/barrido_extendido_folds.json
results/enc_cmp_full/eval/spans.json
```

**Ojo con `rasgos_curva.csv`:** la columna `clase` tiene el nombre
astrofísico real (SNIa-SALT2, SNIIn-MOSFIT, etc.), y la columna `grupo` es la
que vale para filtrar (`normal` o el nombre de la clase excluida).

---

## 10. Estado de la tesis en LaTeX

Directorio: `thesis-local/Tesis docs/`. Compilación:
`pdflatex → bibtex → pdflatex → pdflatex`, **sin `-halt-on-error`**, porque el
`.cls` de la plantilla tiene un `\AtBeginShipout` indefinido en la línea 929 que
es preexistente y no fatal.

Capítulos:
- `capitulos/01_introduccion.tex`
- `capitulos/02_marco_teorico.tex`
- `capitulos/03_metodologia.tex` — agregadas las subsecciones
  `sec:protocolo_heldout` y `sec:baselines`.
- `capitulos/04_resultados.tex` — agregadas `sec:res_piso` y `sec:res_heldout`.
  Contiene `sec:res_codificacion` con la comparación de codificaciones.
- `capitulos/05_discusion_conclusiones.tex` — agregadas "Detección sobre
  anomalías reales" y "La normalización temporal descarta la señal".

Las figuras van en `thesis-local/Tesis docs/img/`.

---

## 11. Pendientes

1. **Subir a Overleaf las figuras regeneradas** (manual, el repo no está
   sincronizado). Son siete, incluyendo `barrido_amplitud.png` y
   `heldout_resumen.png`.
2. **Agregar el resultado de la ablación del eje temporal al capítulo 4 y a la
   discusión.** Hoy vive solo en el informe y en la bitácora. El encuadre ya
   está escrito, falta la tabla y promover el ítem de trabajo futuro a
   resultado.
3. **Opcional, el único experimento que vale la pena:** barrido sintético con
   los dos ejes temporales, para decidir si conviene cambiar la codificación
   principal. Los dos modelos ya existen, tarda minutos.
4. **Evaluar el score de "forma del mapa"** (correlación con el patrón basal)
   como una fila más de la tabla de baselines.
5. Contrafactuales para las clases excluidas (postergado).
6. Varios menores: `capitulos/main.tex` duplicado huérfano, TODOs de dedicatoria
   y agradecimientos, intervalos de confianza por bootstrap.

---

## 12. Lectura general del trabajo

El método funciona donde fue diseñado y falla donde la representación le quita
la información.

- **El piso de sensibilidad está caracterizado:** aguanta hasta Δf ≈ 2 MAD y
  hasta Δt = 320 días, más del 60% del largo de la curva.
- **Sobre anomalías reales el resultado es modesto y hay que decirlo así:**
  0.634 contra 0.612 del mejor baseline no sobrevive al ruido. Las tres clases
  sutiles están en el azar para todos los métodos, incluidos los baselines, así
  que probablemente no sean separables con esta información.
- **El hallazgo que ordena todo:** la normalización temporal borraba la
  duración. Lo mostró un baseline trivial que ganó, y lo confirmó la ablación.
- **El hallazgo metodológico, gratis:** la elección del puntaje pesa 0.113 y la
  del modelo 0.022, sobre el mismo checkpoint.

Una tesis no entrega un número más alto, entrega un método con sus límites
conocidos. Eso está cubierto.
