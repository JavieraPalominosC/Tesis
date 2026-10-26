# Semana del 28 de septiembre

Bitácora de lo que pidió el profe en la reunión, cómo se tradujo a experimentos concretos, qué
decisiones se tomaron y dónde quedó cada resultado. La idea es que esto se pueda leer en frío
dentro de un mes sin tener que reconstruir nada.

**Estado general:** en curso. Última actualización al final del documento.

---

## Lo que pidió el profe

Textual, tal como llegó:

1. Reentrenar VQ-VAE y MaskGIT sin las clases: Supernova 2b, SLSN (son 2 clases), SNIax,
   SNIa-91bg, PISN.
2. Evaluar MaskGIT con las categorías faltantes.
3. Agregar perturbaciones más grandes en $\Delta t$.
4. Ver perturbaciones más chicas en $\Delta f$.
5. Agregar baseline de detección de anomalías.
6. Enviar ejemplos de imágenes normalizadas variando n MAD entre 5 y 300 logarítmicamente.

### Por qué importa

Los puntos 1 y 2 juntos son el experimento que la tesis venía declarando como el principal
pendiente. Hasta ahora todas las anomalías evaluadas eran sintéticas, un pico inyectado en la cola
de la banda *r*, por lo que lo medido era la sensibilidad del método frente a una perturbación
controlada y no su capacidad de encontrar objetos genuinamente raros. Dejar seis clases fuera del
entrenamiento y puntuarlas después convierte eso en una evaluación de detección de anomalías real,
sin necesidad de datos nuevos.

Los puntos 3 y 4 extienden el barrido hacia los dos extremos que hoy no están cubiertos: hacia
anomalías más largas en duración y más sutiles en amplitud, que es donde debería estar el límite
de sensibilidad.

El punto 5 es la respuesta a una limitación que la tesis reconoce de forma explícita: no hay
comparación contra ningún otro detector, así que los números no tienen contra qué contrastarse.

---

## 1. Reentrenamiento sin seis clases

### Mapeo de lo que pidió el profe a las clases del dataset

El orden de las etiquetas lo fija `SN_FILES` en `run_mad300_full.sh`. Los nombres que dio el profe
se traducen así:

| Pidió | Clase en ELAsTiCC-1 | Etiqueta | Imágenes |
|---|---|---|---|
| Supernova 2b | `SNIIb+HostXT_V19` | 11 | 103.683 |
| SLSN (2 clases) | `SLSN-I+host` | 14 | 100.977 |
| SLSN (2 clases) | `SLSN-I_no_host` | 15 | 4.261 |
| SNIax | `SNIax` | 2 | 53.727 |
| SNIa-91bg | `SNIa-91bg` | 1 | 53.414 |
| PISN | `PISN` | 16 | 105.446 |

**Excluidas: 421.508 imágenes (30,3% del dataset).**

Las once clases que quedan para entrenar suman **967.639 imágenes**:

| Etiqueta | Clase | Imágenes |
|---|---|---|
| 0 | `SNIa-SALT2` | 211.892 |
| 3 | `SNII-Templates` | 103.536 |
| 4 | `SNII-NMF` | 102.251 |
| 5 | `SNII+HostXT_V19` | 103.189 |
| 6 | `SNIb-Templates` | 40.065 |
| 7 | `SNIb+HostXT_V19` | 102.883 |
| 8 | `SNIc-Templates` | 20.617 |
| 9 | `SNIc+HostXT_V19` | 103.885 |
| 10 | `SNIcBL+HostXT_V19` | 42.878 |
| 12 | `SNIIn-MOSFIT` | 31.162 |
| 13 | `SNIIn+HostXT_V19` | 105.281 |

Total del dataset: 1.389.147. Verificado contra `data/enc_cmp/mad300/labels.json`.

### Decisiones de diseño

- **No se vuelve a renderizar.** Las imágenes `mad300` ya existen para las 1.39M y la codificación
  no cambia, así que lo único que cambia es qué subconjunto entra al entrenamiento. Esto ahorra las
  ~8 horas de render y garantiza que las imágenes de las clases excluidas sean idénticas a las que
  ya se usaron.
- **Las clases excluidas no aparecen ni en train ni en validación.** Quedan enteramente reservadas
  como conjunto de anomalías. Si aparecieran en validación, el prior las habría visto de forma
  indirecta a través de la selección del mejor checkpoint.
- **Las etiquetas se remapean a 0–10** para la cabeza de clasificación, ya que los índices
  originales quedan con huecos. El mapeo se guarda junto con los folds para poder volver atrás.
- **Se mantiene `mad300` y todos los demás hiperparámetros** idénticos a la corrida completa, de
  modo que la única variable sea qué clases se vieron durante el entrenamiento.

### Archivos

- Folds: `data/enc_cmp/mad300/folds_heldout.json`
- Config VQ-VAE: `configs/vqvae_enc_mad300_heldout.yaml`
- Lanzador: `run_mad300_heldout.sh`
- Resultados: `results/enc_cmp_heldout/mad300/`

### Costo estimado

Escalando desde la corrida completa (1.39M) por el factor 0,70 de tamaño:

| Etapa | Estimado |
|---|---|
| VQ-VAE | ~15 h |
| Tokens | ~12 min |
| MaskGIT (máscara aleatoria) | ~13 h |

---

## 2. Evaluación con las categorías faltantes

Es el experimento que da sentido al punto 1. En vez de inyectar un pico sintético, se puntúan
objetos de las seis clases excluidas y se comparan contra objetos del fold de validación de las
once clases vistas.

Métricas previstas, una por clase excluida además del agregado:

- AUROC entre la distribución de scores de la clase excluida y la de los objetos vistos.
- Recall con el umbral fijado en el percentil 95 de los objetos vistos, que mantiene la tasa de
  falsos positivos en 5% por construcción, igual que en la evaluación sintética.
- Macro F1 sobre las clases normal y anómala, para poder comparar con las cifras que ya se
  reportaron.

Se puntúa con **máscara de ventana**, que es lo que ya se mostró que domina sin importar con qué
máscara se entrenó el prior.

Una prevención para leer los resultados: las seis clases excluidas no son igual de raras. PISN y
SLSN-I son morfológicamente muy distintas del resto y el codebook ya les reservaba vocabulario
propio, mientras que SNIa-91bg y SNIax son subtipos de SNIa y deberían ser mucho más difíciles.
Reportar el promedio sobre las seis escondería justamente lo interesante, así que va a ir siempre
desglosado por clase.

---

## 3 y 4. Barrido extendido en duración y en amplitud

**Hecho.** Los dos puntos se resolvieron en una sola corrida, `scripts/barrido_extendido.py`, con
figuras en `scripts/fig_barrido_extendido.py`. Resultados en
`results/enc_cmp_full/eval/barrido_extendido.json` y figuras en `imagenes/barrido_amplitud.png` y
`imagenes/barrido_duracion.png`. Usa ya los 40 objetos limpios.

**Cifras definitivas**, con los priors que comparten partición con el VQ-VAE (`mgFa` y `mgFv`,
reejecutado en el trabajo 13543024). Las tablas de más abajo son las buenas; las de la versión
con priors de partición propia quedaron en `barrido_extendido.json`, sin sufijo.

### Amplitudes chicas

El barrido anterior arrancaba en $\Delta f = 10$, donde el recall ya era 0,35, de modo que el piso
de sensibilidad quedaba fuera del rango medido. Se bajó hasta 1 MAD, puntuando con ventana:

| $\Delta f$ | 1 | 2 | 3 | 5 | 7 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| recall | 0,12 | 0,28 | 0,33 | 0,45 | 0,53 | 0,55 | 0,78 | 0,85 |
| AUROC | 0,493 | 0,667 | 0,715 | 0,776 | 0,794 | 0,810 | 0,929 | 0,959 |

**El piso quedó acotado.** En $\Delta f = 1$ el detector está exactamente en azar, con AUROC
0,493, y la separación se construye entre 2 y 3 MAD, donde el AUROC pasa de 0,667 a 0,715. El
recall cruza 0,5 alrededor de 7 MAD.

Puntuando con máscara aleatoria, en cambio, el detector se queda en azar hasta $\Delta f = 150$,
lo que vuelve a mostrar que la forma de la máscara al puntuar es lo que más pesa.

### Duraciones largas

El barrido anterior llegaba a $\Delta t = 80$ días, que sobre estas curvas cubre apenas el 31,8%
de los puntos de la banda *r*, cuyo span mediano es de 499 días. Se extendió a 160 y 320 días,
que cubren 58,3% y 61,6%.

Recall, puntuando con ventana:

| $\Delta f$ \ $\Delta t$ | 0 | 5 | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0,05 | 0,07 | 0,20 | 0,07 | 0,23 | 0,15 | 0,38 | 0,35 |
| 10 | 0,38 | 0,38 | 0,35 | 0,35 | 0,40 | 0,50 | 0,65 | 0,62 |
| 30 | 0,72 | 0,78 | 0,82 | 0,62 | 0,85 | 0,82 | 0,80 | 0,82 |
| 70 | 0,60 | 0,78 | 0,75 | 0,78 | 0,72 | 0,85 | 0,95 | 0,95 |
| 150 | 0,68 | 0,62 | 0,75 | 0,68 | 0,75 | 0,97 | 0,97 | 0,95 |
| 300 | 0,85 | 0,95 | 0,88 | 0,90 | 0,93 | 0,93 | 1,00 | 0,95 |
| 500 | 0,85 | 0,93 | 0,90 | 0,85 | 0,93 | 0,97 | 0,97 | 0,97 |

La detección crece en los dos ejes, pero **el límite es de amplitud y no de extensión**. Con
$\Delta f = 3$ el recall no pasa de 0,38 por más que la anomalía dure 320 días, mientras que con
$\Delta f \geq 300$ se detecta prácticamente siempre incluso sin extensión temporal.

---

## 5. Baseline de detección de anomalías

**Parcialmente hecho.** Script en `scripts/baselines_heldout.py`, con tres fases: `curva` extrae
estadísticos de la curva sin necesidad de GPU, `modelo` puntúa con el VQ-VAE y el prior, y
`resumen` arma la tabla de AUROC.

### Decisión de diseño: solo sobre clases excluidas

Se evalúa sobre las seis clases reales que quedaron fuera, y **no** sobre el barrido sintético.
La razón es que la perturbación sintética se define en la línea 150 de `eval_final_mad300.py`
como `flux + d * mad_r`, de modo que un estadístico que mida la desviación máxima en MAD es casi
un filtro adaptado a ella y ganaría por construcción, sin que eso signifique nada. Las clases
excluidas no están definidas en nuestras unidades, así que la comparación es limpia.

Esto además une los puntos 2 y 5, que comparten toda la maquinaria.

### Montaje

Normales: 1.996 objetos muestreados de la validación de las once clases vistas. Anomalías: unos
500 por cada clase excluida, 2.991 en total. El Isolation Forest se ajusta solo con los normales,
que es el protocolo correcto en detección no supervisada. Los rasgos van todos normalizados por el
MAD del propio objeto, para que el detector no aprenda a separar por brillo aparente.

### Resultados de los dos baselines sin modelo

AUROC de cada clase excluida contra el conjunto de normales:

| Clase excluida | `mad_max` | Isolation Forest |
|---|---|---|
| PISN | 0,272 | 0,681 |
| SLSN-I+host | 0,580 | 0,788 |
| SLSN-I_no_host | 0,401 | 0,803 |
| SNIIb+HostXT_V19 | 0,487 | 0,462 |
| SNIa-91bg | 0,495 | 0,511 |
| SNIax | 0,512 | 0,513 |
| **promedio** | **0,458** | **0,626** |

**El estadístico de una línea está bajo el azar**, con 0,458 de promedio. En PISN da 0,272, o sea
que tienen excursiones más chicas respecto de su propio MAD que una supernova normal, algo
esperable en eventos largos y suaves sin pico pronunciado. Confirma que era un filtro adaptado a
la perturbación sintética y nada más. No se invierte el signo, porque elegir la dirección mirando
las etiquetas de test sería hacer trampa, y además invertirlo ayuda en PISN pero perjudica en SLSN.

**El Isolation Forest parte las seis clases en dos grupos.** Funciona con las exóticas, 0,79 y
0,80 en las SLSN y 0,68 en PISN, y queda en el azar con las sutiles, 0,513 en SNIax, 0,511 en
SNIa-91bg y 0,462 en SNIIb. El corte es físicamente coherente: SNIax y SNIa-91bg son variantes
subluminosas de las Ia, y las SNIIb transicionan a Ib, así que son las que se confunden con las
normales.

### Baseline principal: error de reconstrucción del VQ-VAE

Es el que se reporta como baseline en la tesis. Los dos anteriores pasan a segundo plano: se
dejan calculados por si hace falta una comparación que no sea de red neuronal, pero no son el
punto de comparación.

La idea es la habitual en detección de anomalías con autoencoders. El VQ-VAE se entrena solo con
las once clases vistas, así que aprende a redibujar supernovas normales. Se le pasa una imagen,
la codifica a tokens y el decoder la reconstruye; el puntaje de anomalía es el error cuadrático
medio entre original y reconstrucción. Una clase que nunca vio debería reconstruirse peor.

Se prefiere sobre el Isolation Forest por tres razones. Es un modelo y no una receta de
estadísticos elegidos a mano, de modo que desaparece la objeción de que los rasgos estaban mal
escogidos. Sale gratis, porque el modelo ya está entrenado por el punto 1. Y sobre todo, **es la
ablación más limpia posible del método nuestro**: mismo VQ-VAE, mismas imágenes, mismo codebook,
y lo único que cambia es la pregunta. El baseline le pregunta al decoder si puede redibujar la
imagen, y nosotros le preguntamos al prior qué tan improbable es la secuencia de tokens. Toda
diferencia de desempeño se atribuye entonces al prior.

Hipótesis previa, tomada de TimeVQVAE-AD: los métodos de reconstrucción fallan con anomalías
sutiles, así que debería andar en SLSN y PISN y caerse en SNIax, SNIa-91bg y SNIIb.

Advertencia: puede resultar un detector flojo en todas las clases, y no por un error de montaje.
Los VQ-VAE reconstruyen bien incluso lo que nunca vieron, porque el codebook es expresivo y el
encoder convolucional; el nuestro tiene error de reconstrucción de 0,0002. Es un problema conocido
de los autoencoders en detección de anomalías y, si ocurre, es un resultado citable.

### Resultados finales, con los cuatro métodos

AUROC de cada clase excluida contra los 1.996 normales:

| Clase | `mad_max` | iforest | `recon` | `prior_max` | `prior_media` | `prior_max_ven` | **`prior_media_ven`** |
|---|---|---|---|---|---|---|---|
| PISN | 0,272 | 0,681 | 0,588 | 0,543 | 0,562 | 0,629 | **0,679** |
| SLSN-I+host | 0,580 | 0,788 | 0,675 | 0,542 | 0,730 | 0,673 | **0,794** |
| SLSN-I_no_host | 0,401 | 0,803 | 0,847 | 0,558 | 0,855 | 0,704 | **0,883** |
| SNIIb+HostXT_V19 | 0,487 | 0,462 | 0,461 | 0,486 | 0,472 | 0,485 | 0,453 |
| SNIa-91bg | 0,495 | 0,511 | 0,577 | 0,526 | 0,514 | 0,545 | 0,499 |
| SNIax | 0,512 | 0,513 | 0,521 | 0,470 | 0,522 | 0,484 | 0,495 |
| **promedio** | 0,458 | 0,626 | 0,612 | 0,521 | 0,609 | 0,587 | **0,634** |

Dos decisiones de puntaje resultaron importantes, y las dos en la dirección esperada.

**La NLL promedio supera al máximo**, 0,609 contra 0,521. El máximo es lo correcto cuando la
anomalía está localizada en un tramo, que es el caso del barrido sintético, pero acá lo raro es
el objeto entero y el promedio es el puntaje natural de un estimador de densidad.

**La máscara de ventana supera a la aleatoria**, 0,634 contra 0,609, igual que en todo el resto
del trabajo. Con esa combinación el prior pasa a ser el mejor de los cuatro métodos, por encima
del Isolation Forest con 0,626 y del error de reconstrucción con 0,612. La ventaja es chica pero
consistente, y aparece en las tres clases donde algún método detecta algo.

### Diagnóstico por rasgo: la duración es lo que importa

Se midió cuánto separa cada rasgo por sí solo, para entender de dónde sale la señal:

| Rasgo | PISN | SLSN+host | SLSN_no_host | fuerza media |
|---|---|---|---|---|
| **`span`, en días** | 0,662 | 0,741 | **0,895** | **0,146** |
| `n`, número de puntos | 0,648 | 0,736 | 0,886 | 0,139 |
| `cadencia` | 0,634 | 0,691 | 0,811 | 0,122 |
| `abs_log_pico` | 0,431 | 0,664 | 0,409 | 0,063 |
| `abs_log_mad` | 0,694 | 0,625 | 0,522 | 0,068 |

**La variable discriminativa es la duración, no el brillo.** El span en días, un solo número, da
0,895 en SLSN-I_no_host, por encima del pipeline completo que da 0,883. Las otras dos variables
fuertes, número de puntos y cadencia, son derivadas de la misma cosa. El brillo absoluto llega
apenas a 0,664 en el mejor caso, así que la hipótesis de que las SLSN se detectan por ser
superluminosas queda descartada.

Esto señala un problema de diseño concreto y acotado: el renderizado normaliza el eje de tiempo
al span de cada objeto, con `(mjd - t0) / rango` en `eval_enc_cmp.render`, de modo que la imagen
borra justamente la variable más discriminativa. No es la normalización del flujo la que estorba,
que era la sospecha inicial, sino la del tiempo.

Una variante que conserve la duración absoluta requiere reentrenar la cadena completa, porque el
VQ-VAE y el prior quedarían fuera de distribución con otra normalización. Son unos dos días.

### Límite real: las tres clases sutiles

SNIax, SNIa-91bg y SNIIb no las detecta **ningún** método, y el diagnóstico por rasgo muestra que
tampoco las separa ningún rasgo individual: SNIIb queda bajo el azar en todas las columnas. No es
una falla del prior sino un límite de los datos en esta representación. Es coherente con la
física, porque SNIax y SNIa-91bg son variantes subluminosas de las Ia y las SNIIb transicionan a
Ib, así que son las que más se confunden con las normales.

El objetivo queda bien definido: el mérito no está en superar 0,80 en SLSN, donde cualquier método
razonable funciona, sino en sacar algo sobre 0,51 en las tres clases sutiles.

---

## 6. Imágenes variando n MAD entre 5 y 300

El parámetro es el $U$ de la codificación:

$$
u = \frac{1}{2}\left[\mathrm{clip}\left(\frac{\operatorname{arcsinh}\left((f - \operatorname{med}(f))/\operatorname{MAD}(f)\right)}{\operatorname{arcsinh}(U)}, -1, 1\right) + 1\right]
$$

$U$ fija cuántos MAD entran en el carril de la banda. Se piden ejemplos con $U$ espaciado
logarítmicamente entre 5 y 300, que es justo el rango donde se movió la decisión original, tomada
comparando solo $U = 100$ y $U = 300$.

**Hecho.** Script `scripts/fig_barrido_umad.py`, salidas en `imagenes/umad_ejemplos.png`,
`imagenes/umad_curvas.png` y `imagenes/umad_resumen.json`.

Se usan siete valores, $U \in \{5, 10, 20, 39, 77, 152, 300\}$, y cuatro objetos de tipos
distintos, eligiendo en cada clase el primero que tiene fotometría en las seis bandas para que la
comparación entre valores de $U$ no dependa de que falte una banda. El render reproduce exactamente
la geometría del renderizador de entrenamiento, o sea 256×256 píxeles, $A = 1.8$ y $S = 2.0$.

Para cuantificar el compromiso se mide qué fracción del carril ocupa realmente la curva, tomando el
rango del 90% central de $u$:

| $U$ | SNIa-SALT2 | SNII-NMF | SLSN-I+host | PISN |
|---|---|---|---|---|
| 5 | 0,838 | 0,919 | 0,666 | 0,809 |
| 10 | 0,679 | 0,711 | 0,516 | 0,680 |
| 20 | 0,553 | 0,579 | 0,420 | 0,554 |
| 39 | 0,466 | 0,489 | 0,354 | 0,467 |
| 77 | 0,403 | 0,422 | 0,306 | 0,404 |
| 152 | 0,355 | 0,372 | 0,270 | 0,356 |
| 300 | 0,317 | 0,332 | 0,241 | 0,318 |

En $U = 300$, que es el valor adoptado, la curva normal ocupa alrededor de un tercio del carril, y
en el caso de SLSN-I baja a menos de un cuarto. Traducido a píxeles, con un carril útil de 37,8 px,
son unos 12 px para toda la morfología de la curva, o sea **1,5 tokens de los 4,7 que mide la banda
de alto**. Es la contracara del margen que se gana para el outlier, y da una explicación
cuantitativa del comentario de que las imágenes `mad300` se ven planas.

---

## Hallazgo: los folds del VQ-VAE y los del MaskGIT no eran los mismos

Surgió al preparar el punto 1 y afecta a resultados ya reportados, así que va documentado aparte.

### Qué pasaba

El VQ-VAE se entrenaba con `folds_full.json`, partición 90/10 generada por `make_folds_enc.py`.
El prior MaskGIT, en cambio, **no leía ese archivo**: hacía su propio `random_split` 80/20 sobre el
arreglo completo de tokens, con semilla 42, sin ninguna relación con la partición del VQ-VAE.

Las dos particiones son independientes, por lo que un objeto podía estar en validación del VQ-VAE
y en entrenamiento del MaskGIT al mismo tiempo. Medido sobre la corrida completa:

- 1.389.147 tokens, que el MaskGIT parte en 1.111.317 train y 277.830 val.
- De los 138.914 objetos de validación del VQ-VAE, **110.950 (79,9%) estaban en el entrenamiento
  del MaskGIT**.
- Solo 27.964 (20,1%) caían en validación de ambos modelos.

### Consecuencia sobre la evaluación

Los 40 SNIa de la evaluación definitiva se eligen del fold de validación del VQ-VAE. Verificado
reproduciendo el split con la misma semilla: **32 de los 40 objetos de evaluación fueron vistos por
el MaskGIT durante el entrenamiento.**

Esto contradice lo que afirma la tesis en el capítulo de resultados:

> "Los 40 SNIa de evaluación provienen del fold de validación, de modo que ninguno fue visto por el
> VQ-VAE ni por los priors."

La parte del VQ-VAE es correcta. La de los priors no lo es para 32 de los 40 objetos.

### Dirección del sesgo

El prior vio la versión **limpia** de esos objetos, nunca la perturbada. Memorizar la curva limpia
baja la NLL del control, y como el umbral $\tau$ es el percentil 95 de los controles, un control
memorizado empuja el umbral hacia abajo y hace que más perturbadas lo superen. **El sesgo es
optimista**, o sea que el recall y el AUROC reportados están probablemente inflados.

La magnitud no es obvia, porque los 8 objetos no memorizados tienen NLL de control más alta y son
justamente los que pueden estar fijando el percentil 95. Hay que medirlo, no estimarlo.

### Corrección

Elegir los objetos de evaluación en la **intersección** de las dos validaciones. Hay 20.912 SNIa en
el fold de validación del VQ-VAE y alrededor del 20% cae también en validación del MaskGIT, así que
quedan unos 4.200 candidatos, de sobra para elegir 40.

Para las corridas nuevas el problema ya está resuelto: se agregó la variable `FOLDS_SPLIT` a
`train_maskgit_enc.py`, que hace que el prior tome la partición desde el archivo de folds en vez de
sortearla. La cadena de clases excluidas lanzada hoy la usa, por lo que ahí validación significa lo
mismo en toda la cadena.

### Magnitud real del sesgo, medida

Se reejecutó la evaluación eligiendo los 40 objetos en la intersección de las dos validaciones,
donde hay 4.191 SNIa candidatos. Los dos conjuntos de 40 resultaron disjuntos, así que la
comparación mezcla el efecto de la fuga con el de estar mirando supernovas distintas.

Celda recomendada, prior entrenado con máscara aleatoria y puntuado con ventana:

| $\Delta f$ | recall con fuga | recall limpio | AUC con fuga | AUC limpio |
|---|---|---|---|---|
| 10 | 0,35 | 0,42 | 0,751 | 0,834 |
| 20 | 0,62 | 0,68 | 0,888 | 0,924 |
| 30 | 0,68 | 0,70 | 0,908 | 0,929 |
| 50 | 0,82 | 0,62 | 0,958 | 0,904 |
| 70 | 0,78 | 0,72 | 0,950 | 0,940 |
| 100 | 0,80 | 0,75 | 0,941 | 0,953 |
| 150 | 0,88 | 0,75 | 0,961 | 0,952 |
| 200 | 0,93 | 0,93 | 0,979 | 0,981 |
| 300 | 0,90 | 0,95 | 0,981 | 0,991 |
| 500 | 1,00 | 1,00 | 0,997 | 0,996 |

**El sesgo no resultó material.** Las diferencias van en las dos direcciones, el recall sube en
$\Delta f$ de 10, 20, 30 y 300 y baja en 50, 70 y 150, que es el comportamiento esperable al
sortear 40 objetos de entre 4.191 y no el de un sesgo sistemático. Las cifras de titular se
mantienen: recall 1,00 y AUROC 0,996 contra el 0,998 reportado.

La excepción es la monotonía, que baja de 0,545 a 0,376 y sí merece corregirse en el texto.

Respaldo de las cifras contaminadas en `results/enc_cmp_full/eval/final_mad300_CONFUGA.json`.

### Qué habría que regenerar

Todo lo que dependa de los 40 objetos contaminados: el cruce de máscaras, las curvas de recall y
AUROC contra $\Delta f$, las figuras en flujo absoluto, la matriz de confusión y el macro F1
preparados para la reunión, y las cifras de recall 1,00 y AUROC 0,998 que están en las
conclusiones. La evaluación de contrafactuales usa 300 SNIa elegidos igual, así que tiene el mismo
problema.

---

## Referencia: TimeVQVAE-AD (arXiv 2311.12550)

Surgió al resolver una duda sobre si el MaskGIT enmascara la imagen real o los tokens. El paper
es la referencia más cercana a lo que hacemos y conviene citarlo, porque la arquitectura coincide
casi punto por punto aunque ellos trabajen con series de tiempo y nosotros con imágenes.

### Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Diagrama de las dos etapas, con copos de nieve sobre el VQ-VAE congelado | Figura 4 |
| Objetivo de entrenamiento del prior | Sección 4.1, Ec. 3 |
| Definición de token como índice del codebook | Sección 4, primer párrafo |
| Softmax sobre el tamaño del codebook | Ec. 1a y 1b |
| Enmascaramiento aleatorio en entrenamiento | Sección 4, párrafo 2 |
| Puntaje de anomalía | Sección 4, párrafo 2 |

### Qué coincide con lo nuestro

La etapa 2 maximiza $\mathbb{E}_{x,\, z_q \sim VQ(E(\mathrm{STFT}(x)))}[p_\theta(s\mid s_M)]$, con
$E$ y $D$ *"set to be untrainable in the second stage"*. El enmascaramiento es
$s_M = s \odot m + \texttt{[MASK]} \odot (1-m)$ y el puntaje de anomalía es
$a = -\log p_\theta(s \mid s_M)$, que es exactamente nuestra NLL por token. El enmascaramiento en
entrenamiento es aleatorio y de fracción aleatoria, igual que el nuestro.

Vale aclarar un matiz que se prestó a confusión: el prior usa los **índices** del codebook, no sus
vectores. Los embeddings del VQ-VAE solo reaparecen al decodificar tokens a imagen, o sea en el
contrafactual. Para el transformer el codebook es un alfabeto de 512 símbolos, y el símbolo
`[MASK]` es un 513.º que se agrega aparte y que la cabeza nunca puede predecir, porque tiene 512
salidas.

### Tres puntos que conectan con la pauta del profe

**La ventana valida el punto de las máscaras.** Ellos usan *"a sliding masking window with an
arbitrary window size"* que abarca toda la dimensión de frecuencia y se desliza en el tiempo. Es
el análogo directo de nuestro hallazgo de que puntuar con ventana domina sobre puntuar con celdas
aleatorias. Sugiere además una variante que no probamos: enmascarar la columna completa de las
seis bandas en vez de una ventana dentro de una banda.

**Sus tamaños de ventana múltiples cubren el eje $\Delta t$.** Agregan varios $\alpha$ y combinan
los puntajes, lo que es más elegante que nuestro barrido manual de duraciones y podría reemplazarlo.

**Su argumento sobre reconstrucción anticipa el baseline.** Sostienen que los métodos basados en
reconstrucción *"inherently struggle to detect anomalies with small amplitude"*, lo que motiva el
baseline del punto 5 y predice dónde debería perder: justo en el régimen de $\Delta f$ chico que
acabamos de medir en el punto 4, donde el AUROC nuestro cae a 0,493.

---

## Referencias: predicción por escalas (VAR y NCM)

Dos papers que usan la misma receta de dos etapas que esta tesis, pero con un codebook
multiescala en vez de una grilla plana. Agregados a `library.bib` como `tian2024var` y
`kumbong2026ncm`.

### Qué proponen

**VAR** (`tian2024var`, arXiv 2404.02905, NeurIPS 2024) es el que aplica a imágenes y por lo tanto
el relevante acá. Reemplaza la predicción token a token en orden de barrido por **predicción por
escalas**: el autoencoder cuantiza el mapa de rasgos en $K$ mapas de tokens
$(r_1, \ldots, r_K)$ de resolución creciente $h_k \times w_k$, terminando en la resolución
original, y el transformer genera un mapa completo por paso, en paralelo dentro de cada escala,
condicionado en las escalas anteriores. Las resoluciones siguen una progresión geométrica
$n_k = a^{k-1}$ y **el codebook es compartido entre todas las escalas**, así que el vocabulario
sigue siendo uno solo.

**NCM** (`kumbong2026ncm`, arXiv 2610.14162) lleva la misma idea a texto, con cuantización
residual: el nivel $k$ codifica el error que dejaron los anteriores, siguiendo
$\hat{z}^{(k)} = \hat{z}^{(k-1)} + e_k(r_k)$ con $\ell_1 < \cdots < \ell_K = L$, y ahí sí cada
nivel tiene su propio codebook. Ellos mismos señalan que adaptarlo a texto costó porque no hay
análogo natural del submuestreo de imágenes, de modo que para esta tesis VAR es la referencia
directa y NCM la evidencia de que la idea se está extendiendo.

### Por qué conectan con lo que se midió

El barrido en $\Delta t$ es, en el fondo, una pregunta sobre la **escala** de la anomalía: un pico
en una sola época y una meseta de 320 días son fenómenos de escalas distintas, y la grilla plana
de $32 \times 32$ los representa a los dos con la misma resolución. Con una jerarquía multiescala
una anomalía larga caería en los niveles gruesos y una puntual en los finos, y el score se podría
descomponer por nivel, lo que diría no solo que un objeto es anómalo sino a qué escala lo es.

Cumple el mismo papel que los tamaños de ventana múltiples de TimeVQVAE-AD, pero metido en la
representación en vez de en la puntuación.

VAR además muestra *in-painting*, *out-painting* y edición sin entrenamiento adicional, que es
estructuralmente la misma operación que la generación de contrafactuales de esta tesis.

### Qué no resuelve

Nada de esto recupera la duración absoluta. El renderizado normaliza el eje temporal al span de
cada objeto, así que la información se pierde **antes** de que el VQ-VAE vea la imagen, y ningún
codebook la puede reconstruir. Es un problema ortogonal al de la escala de representación.

---

## Ablación del eje temporal

Es la continuación directa del diagnóstico por rasgo del punto 5: si la duración es la variable
más discriminativa y el renderizado la borra, la pregunta obvia es qué pasa si no se borra.

Va planteada como **ablación y no como reemplazo**. La codificación principal sigue siendo
`mad300` con eje por objeto, de modo que ninguna cifra de la tesis se mueve. Esto entra al lado de
la comparación de codificaciones del capítulo 4, que ya tiene ese formato.

### Qué cambia exactamente

Una línea de `render_subset_enc.py`. El eje pasa de normalizarse por el span de cada objeto a
usar una escala común `T_ABS` en días:

$$x = \frac{\mathrm{mjd} - \mathrm{mjd}_{\min}}{\mathrm{span}} \quad\longrightarrow\quad
x = \frac{\mathrm{mjd} - \mathrm{mjd}_{\min}}{T}$$

Todo lo demás queda idéntico: misma codificación de flujo con $U = 300$, mismas seis clases
excluidas, mismos hiperparámetros, misma semilla de folds. Como los folds salen de ordenar los
mismos nombres de archivo, **los objetos de evaluación coinciden uno a uno** con los de la rama
principal y la comparación queda pareada.

### Por qué $T = 1110$ días

Se midió la distribución de duraciones sobre las 1.389.147 curvas con `scripts/medir_span.py`,
guardada en `results/enc_cmp_full/eval/spans.json`:

| percentil | 1 | 5 | 10 | 25 | 50 | 75 | 90 | 95 | 99 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| span, días | 123 | 154 | 170 | 230 | 532 | 560 | 787 | 902 | 1094 | 1104 |

La razón p90/p10 es **4,64**, o sea menos de un orden de magnitud. Eso es lo que hace viable un
eje lineal: si la dispersión hubiera sido de varios órdenes, habría hecho falta comprimir en log y
el resultado sería mucho más difícil de interpretar.

$T = 1110$ está apenas por encima del máximo observado, así que **no se recorta ninguna curva**.
Un $T$ más chico aprovecharía mejor el ancho, pero recortaría justo a los objetos más largos, que
son las SLSN que interesa detectar, de modo que destruiría la señal que se quiere agregar.

Por clase, las medianas confirman de dónde saldría la separación: SLSN-I sin host 895 d y
SLSN-I con host 808 d, contra 513 a 540 d de todas las clases vistas. PISN queda en 558 d, apenas
por encima, así que ahí no debería ganar nada.

### El precio, verificado antes de gastar el cómputo

`scripts/fig_eje_temporal.py` renderiza los mismos objetos de las dos formas, en
`imagenes/eje_temporal.png`:

| Caso | span | ancho que ocupa | tokens de los 32 |
|---|---|---|---|
| SNIa-SALT2, duración mediana | 532 d | 48% | 15,3 |
| SNIa-SALT2, decil más corto | 170 d | 15% | 4,9 |
| SLSN-I sin host (excluida) | 895 d | 81% | 25,8 |
| PISN (excluida) | 558 d | 50% | 16,1 |

La señal aparece: 81% contra 15% es una diferencia imposible de no ver. El costo también es real,
porque el decil más corto queda en **4,9 tokens de ancho** y la mediana pierde la mitad de la
resolución de forma que tenía.

### Qué esperar, y una advertencia para leer el resultado

Con la mitad de la imagen en blanco para un objeto mediano, buena parte de los tokens pasan a ser
fondo trivialmente predecible, y la fracción de tokens en blanco **es** la duración. Como el mejor
puntaje medido es la NLL promedio sobre la secuencia, es muy posible que la mejora llegue por esa
vía y no porque el prior entienda mejor la morfología.

Eso no invalida el resultado, que es que la representación ahora sí lleva la información, pero hay
que describir el mecanismo y no venderlo como otra cosa. La forma de distinguirlo es mirar si
mejora también en las tres clases sutiles, que tienen duración normal: si solo mejoran SLSN y PISN,
la ganancia es de conteo de fondo.

Riesgo en la otra dirección: perder la mitad del ancho podría degradar la detección de anomalías
sintéticas en $\Delta f$, que es el resultado principal de la tesis. Por eso la ablación se evalúa
**solo** sobre la tarea de clases excluidas y no se rehace el barrido sintético.

### Montaje

- Renderizado: `T_ABS=1110`, salida en `data/enc_cmp/mad300t/`
- Config VQ-VAE: `configs/vqvae_enc_mad300t_heldout.yaml`
- Lanzador: `run_mad300_tabs.sh`
- Resultados: `results/enc_cmp_heldout/mad300t/` y `results/enc_cmp_heldout/eval_mad300t/`

Se entrena **solo el prior de máscara aleatoria**, porque la mejor combinación medida es entrenar
con aleatoria y puntuar con ventana, y eso no necesita un prior propio. Ahorra 13 h.

| Etapa | Trabajo | Estimado |
|---|---|---|
| render, 17 en paralelo | 17702371–17702387 | ~1,5 h |
| folds | 17702388 | min |
| VQ-VAE | 17702389 | 15 h 16 |
| tokens | 17702394 | 17 min |
| MaskGIT | 17702395 | 13 h 13 |
| baselines | 17702396 | 7 min |

### Resultados

La cadena terminó el 23-10 a las 20:12. Los folds salieron idénticos a los de la rama
principal, 967.639 para entrenar y 421.508 excluidas, así que la comparación quedó pareada.

**La representación empeoró por todas las métricas intrínsecas**, que era el riesgo previsto:

| | eje por objeto | eje absoluto |
|---|---|---|
| perplexity del codebook | 154 | 56 |
| error de reconstrucción | 0,00223 | 0,00257 |
| exactitud del clasificador | 0,296 | 0,222 |
| `val/loss` del MaskGIT | 3,3151 | 2,2299 |

La pérdida del MaskGIT **no es comparable**: con la mitad de la imagen en blanco hay muchos
tokens de fondo trivialmente predecibles, así que la caída de 3,32 a 2,23 es casi toda
dilución y no mejor modelo.

**La detección mejoró igual:**

| Clase | eje por objeto | eje absoluto | dif |
|---|---|---|---|
| SLSN-I_no_host | 0,883 | **0,949** | +0,066 |
| SLSN-I+host | 0,794 | 0,800 | +0,006 |
| PISN | 0,679 | 0,681 | +0,002 |
| SNIa-91bg | 0,499 | 0,527 | +0,028 |
| SNIax | 0,495 | 0,527 | +0,032 |
| SNIIb+HostXT_V19 | 0,453 | 0,460 | +0,007 |
| **promedio** | 0,634 | **0,657** | +0,023 |
| baseline de reconstrucción | 0,612 | 0,600 | −0,012 |

Tres cosas que conviene separar al contarlo.

**El baseline de reconstrucción empeoró en paralelo**, de 0,612 a 0,600. Eso es lo que
descarta la explicación aburrida de que las imágenes nuevas sean más fáciles en general: es
específicamente el prior el que aprovecha la información agregada. Sin ese dato, la mejora
del prior sería ambigua.

**El pipeline por fin supera a la duración sola.** En SLSN-I_no_host pasa de 0,883 a 0,949
contra los 0,895 del rasgo suelto, que es lo que cierra el diagnóstico: antes el número de
una línea le ganaba al modelo, ahora no.

**Las tres clases sutiles también subieron**, poco pero las tres, y dos cruzaron 0,5 por
primera vez. Era la prueba fijada de antemano: si solo hubieran mejorado SLSN y PISN, la
ganancia se explicaría por contar fondo en blanco. Que suban clases de duración normal
indica que el prior usa algo más que el ancho. Aun así siguen cerca del azar, en 0,527, de
modo que el límite de las clases sutiles sigue intacto.

La lectura general es que **la información descartada valía más que la capacidad de codebook
que cuesta conservarla**. No era obvio de antemano y por eso había que medirlo.

---

## Macro F1, matriz de confusión y curvas de entrenamiento

Pedido en la reunión anterior y rehecho con los priors de partición compartida, con
`PRIOR_TAG=_folds python scripts/fig_maskgit_final.py`. Las versiones del 21 de septiembre salían
de los priors viejos y los objetos con fuga, así que quedaron obsoletas.

Salidas en `imagenes/`: `maskgit_curvas_folds.png`, `maskgit_confusion_folds.png`,
`maskgit_f1_folds.png` y `maskgit_final_resumen_folds.json`.

### Curvas de entrenamiento

| Prior | Épocas | Mejor `val/loss` | Época |
|---|---|---|---|
| máscara aleatoria | 20 | 3,1713 | 14 |
| máscara de ventana | 19 | 3,6897 | 13 |

Las dos pérdidas **no son comparables entre sí**, porque cada prior se evalúa con la forma de
máscara con que fue entrenado y completar un rectángulo contiguo es más difícil que completar
celdas sueltas. La brecha entre validación y entrenamiento se mantiene negativa casi todo el
recorrido y recién cruza cero al final, que es donde corta el early stopping.

### Matriz de confusión y macro F1

Sobre 40 controles y 40 perturbadas, con el umbral en el percentil 95 de los controles, que deja
2 falsos positivos fijos por construcción:

| $\Delta f$ | Verdaderos positivos | Falsos negativos | macro F1 viejo | **macro F1 nuevo** |
|---|---|---|---|---|
| 30 | 33 | 7 | 0,809 | **0,887** |
| 100 | 38 | 2 | 0,874 | **0,950** |
| 300 | 37 | 3 | 0,925 | **0,937** |

El macro F1 sube en todo el rango salvo en $\Delta f = 500$, donde baja de 0,975 a 0,962 porque
el recall pasó de 1,00 a 0,98, o sea un objeto de los 40.

---

## Qué cambió en la tesis

Las cifras de detección del capítulo 4 venían del montaje de septiembre, con los 40 objetos que
tenían fuga y los priors de partición propia. Lo actualizado, todo desde
`final_mad300_folds.json` y `barrido_extendido_folds.json`:

| Dónde | Antes | Ahora |
|---|---|---|
| $\tau$, percentil 95 de controles | 10,65 | 10,51 |
| AUROC en $\Delta f = 10$ | 0,75 | 0,80 |
| AUROC en $\Delta f = 50$ | 0,96 | 0,98 |
| AUROC en el extremo | 1,00 | 0,996 |
| Recall en $\Delta f = 500$ | 1,00 | 0,98 |
| Ganancia de la ventana en $\Delta f = 100$ | 0,13 → 0,80 | 0,38 → 0,95 |
| Localización entre $\Delta f$ 100 y 300 | 75% | 75 a 88% |
| Localización sobre controles | 0 a 5% | 0% |
| Inicio del régimen útil | 200 FLUXCAL, F1 > 0,85 | 140 FLUXCAL, F1 = 0,88 |

Figuras regeneradas y copiadas a `thesis-local/Tesis docs/img/`: `final_mad300.png`,
`roc_deltaf.png`, `dist_max_nll.png`, `f1_recall_vs_flux.png` y `contour_dt_dF.png`. **Hay que
subirlas a Overleaf**, porque el repo local y Overleaf no están sincronizados.

El contorno ahora sale del barrido extendido, así que cubre hasta 320 días en vez de 80, y el eje
vertical indica qué fracción de la banda \textit{r} abarca cada ventana en lugar del número de
puntos.

### Fase B: contenido nuevo

La fase A corrigió cifras que habían quedado desactualizadas. La fase B agrega lo que antes no
existía en el texto. La tesis pasó de 58 a 63 páginas, compila sin referencias ni citas rotas y
sin desbordes sobre 10 pt.

**Capítulo 3**, dos subsecciones nuevas al final de *Detección de anomalías*:

- `sec:protocolo_heldout`, el protocolo de clases excluidas. Qué clases, los conteos, que no
  aparecen ni en train ni en validación y por qué, el remapeo de etiquetas, que los tokens sí se
  generan sobre las 1,39M, el montaje de evaluación, y la justificación de reportar también la
  NLL promedio además del máximo.
- `sec:baselines`, los tres baselines. Incluye el argumento de por qué se evalúan sobre clases
  excluidas y no sobre el barrido sintético, que es el punto metodológico que evita que
  `mad_max` gane por construcción.

**Capítulo 4**, dos subsecciones nuevas:

- `sec:res_piso`, el piso de sensibilidad, con la tabla de $\Delta f$ de 1 a 10 en MAD y en
  FLUXCAL y la figura `barrido_amplitud.png`. El punto es que el AUROC en $\Delta f = 1$ da
  0,493, o sea azar exacto, que es lo que corresponde cuando la perturbación tiene el tamaño de
  la dispersión del propio objeto.
- `sec:res_heldout`, la detección sobre clases excluidas. Tres tablas: AUROC por clase con los
  cuatro métodos, el cruce de máscara contra estadístico, y el diagnóstico por rasgo. La
  observación que estructura la sección es que la elección del puntaje mueve 0,113 de AUROC
  mientras que la del método mueve 0,022.

**Capítulo 5**: dos subsecciones de discusión, *Detección sobre anomalías reales* y *La
normalización temporal descarta la señal*; tres limitaciones reescritas, porque la que decía que
todas las anomalías eran sintéticas ya no es cierta; el objetivo 5 y el párrafo de cierre
actualizados; un cuarto aporte sobre el costo de normalizar por objeto; y tres ítems nuevos de
trabajo futuro, de los que el primero es la ablación que está corriendo.

La simetría entre los dos ejes es lo que da el argumento de la discusión: la normalización por
objeto se aplicó al flujo y al tiempo por las mismas razones y rompió lo mismo en los dos casos,
pero en el flujo bastó cambiar el estadístico porque la información seguía ahí mal escalada,
mientras que en el tiempo está directamente ausente.

### Lo que no se tocó

Dos cifras que **no** se tocaron, por si parecen candidatas. El "recall de 0,38 a 1,00" y la
correlación "de 0,09 a 0,60" de la discusión vienen de la comparación entre codificaciones sobre
39 SNIa, que es otro experimento y no se rehizo. El "75% de sus tokens fuera de ella" del
capítulo 4 es la cobertura de la máscara del contrafactual, otra métrica distinta.

---

## Registro de avance

| Fecha | Qué se hizo |
|---|---|
| 29-09 | Verificado el mapeo de las seis clases a etiquetas y sus conteos contra `labels.json`. |
| 29-09 | Folds de clases excluidas generados: 967.639 vistas (870.876 train, 96.763 val) y 421.508 excluidas. |
| 29-09 | Añadido `FOLDS_SPLIT` a `train_maskgit_enc.py`, para que el prior use la misma partición que el VQ-VAE y no vea las clases excluidas. |
| 29-09 | Lanzada la cadena `run_mad300_heldout.sh`: VQ-VAE 13542119, tokens 13542120, MaskGIT 13542121. |
| 29-09 | Punto 6 terminado: barrido de $U$ de 5 a 300 con figuras y tabla de uso del carril. |
| 29-09 | `scripts/barrido_extendido.py` escrito y probado con 3 objetos: corre en 53 s, o sea ~12 min con 40. Pendiente de lanzar hasta corregir la selección de objetos. |
| 29-09 | **Detectada fuga**: 32 de los 40 objetos de evaluación estaban en el entrenamiento del MaskGIT. Ver la sección anterior. |
| 29-09 | Corregida la selección: `scripts/split_maskgit.py` reproduce la partición del prior y `candidatos()` exige pertenecer a las dos validaciones. Quedan 4.191 SNIa limpios. |
| 29-09 | Reejecutada la evaluación con objetos limpios: las cifras de titular se mantienen, solo la monotonía baja de 0,545 a 0,376. |
| 29-09 | Relanzados los dos priors de la rama principal con la partición compartida: `mgFa` 13542763 y `mgFv` 13542764, ambos con 1.250.233 train y 138.914 val desde `folds_full.json`. |
| 29-09 | Puntos 3 y 4 terminados con `barrido_extendido.py` (6 min). El piso de sensibilidad queda acotado: AUROC 0,531 en $\Delta f = 1$. Figuras en `imagenes/barrido_amplitud.png` y `barrido_duracion.png`. |
| 29-09 | Documentada la referencia TimeVQVAE-AD con las ubicaciones exactas para mandarle al profe. |
| 29-09 | Añadido `PRIOR_TAG` a la evaluación: apunta a los checkpoints nuevos, desactiva el filtro de `split_maskgit` que con partición compartida sobra, y sufija las salidas. Encolado `run_eval_folds.sh` (13543024) tras los dos priors, para rehacer los puntos 3 y 4 en cuanto estén. |
| 29-09 | Punto 5 arrancado. Decidido evaluar los baselines solo sobre clases excluidas, para evitar que la perturbación sintética favorezca por construcción a un detector de desviación en MAD. Los dos baselines sin modelo ya corrieron: `mad_max` 0,458 promedio, bajo el azar, e Isolation Forest 0,626, que solo funciona con las clases exóticas. |
| 29-09 | Elegido el error de reconstrucción del VQ-VAE como baseline principal, por ser un modelo y por ser la ablación limpia del prior. `mad_max` e Isolation Forest quedan como referencia secundaria. |
| 30-09 | `mgFa` (19:23) y `mgFv` (21:12) terminados con folds compartidos. Convergieron a 3,1713 y 3,6897, contra 3,1837 y 3,6200 de los viejos: prácticamente lo mismo. |
| 30-09 | Reejecutados los puntos 3 y 4 con los priors nuevos (13543024, 8 min). **Las cifras mejoran**, y la monotonía se recupera de 0,376 a 0,503. Con partición compartida el conjunto de candidatos vuelve a los 138.914 de la validación del VQ-VAE, sin filtro extra. |
| 30-09 | Cadena del punto 1 en su última etapa: `vqHO` y `tokHO` terminados, `mgHO` en época 14 de 25. |
| 01-10 | Cadena del punto 1 completa. Puntos 2 y 5 resueltos en la misma tabla (13741910 y 13833301). |
| 01-10 | Con NLL promedio y máscara de ventana el prior queda primero, 0,634 contra 0,626 del Isolation Forest y 0,612 de reconstrucción. Con las dos decisiones al revés quedaba en 0,521, así que la elección del puntaje pesaba más que el método. |
| 01-10 | **Diagnóstico por rasgo**: la señal está en la duración, no en el brillo. El span en días solo da 0,895 en SLSN-I_no_host, más que el pipeline completo. El renderizado normaliza el eje temporal por objeto y borra esa variable. El brillo absoluto no discrimina, así que la hipótesis de las superluminosas queda descartada. |
| 01-10 | Confirmado que las tres clases sutiles no las detecta ningún método ni ningún rasgo individual. Es un límite de los datos, no del prior. |
| 01-10 | **Tesis puesta al día.** Las cinco figuras de detección del capítulo 4 eran del 7 al 9 de septiembre, o sea de los 40 objetos con fuga y los priors viejos. Regeneradas con `PRIOR_TAG=_folds`, junto con las dos tablas y las cifras en prosa de los capítulos 4 y 5 y del resumen. Compila limpio, 58 páginas, sin desbordes ni referencias rotas. |
| 21-10 | Rehechas las tres figuras del profe (macro F1, matriz de confusión y curvas de entrenamiento) con los priors de partición compartida. El macro F1 sube en todo el rango, de 0,809 a 0,887 en $\Delta f = 30$ y de 0,874 a 0,950 en 100. |
| 21-10 | Agregadas a `library.bib` las referencias de predicción por escalas, VAR (`tian2024var`) y NCM (`kumbong2026ncm`), con la nota de por qué conectan con el eje $\Delta t$ y por qué no resuelven lo de la normalización temporal. |
| 22-10 | Medida la distribución de duraciones con `scripts/medir_span.py`: p90/p10 = 4,64, máximo 1104 d. Dispersión chica, así que un eje lineal común es viable y no hace falta comprimir en log. |
| 22-10 | Verificado el precio antes de gastar cómputo con `scripts/fig_eje_temporal.py`: con $T = 1110$ la SLSN ocupa 81% del ancho y el decil más corto 15%, o sea 4,9 tokens. La señal aparece y la pérdida de resolución es tolerable. |
| 22-10 | **Lanzada la ablación de eje temporal** (`run_mad300_tabs.sh`, trabajos 17702371 a 17702396, ~30 h). Rama `mad300t`, pareada objeto a objeto con la principal. |
| 22-10 | **Fase B de la tesis terminada.** Dos subsecciones nuevas en el capítulo 3 y dos en el 4, más la discusión, limitaciones, conclusiones, aportes y trabajo futuro del capítulo 5. De 58 a 63 páginas, compila limpio. |
| 22-10 | Corregidas dos cifras obsoletas de este documento al verificarlas contra los JSON: el AUROC en $\Delta f = 1$ es 0,493 y no 0,531, y la cobertura de $\Delta t$ de 160 y 320 días es 58,3% y 61,6%, no 55,4% y 62,1%. Las dos venían de la corrida con priors viejos. |
| 23-10 | **Ablación de eje temporal terminada.** El promedio sube de 0,634 a 0,657 y SLSN-I_no_host de 0,883 a 0,949, superando por fin a la duración sola (0,895). La representación empeora en todo (perplexity 154 → 56) y la detección mejora igual. El baseline de reconstrucción baja de 0,612 a 0,600, lo que descarta que las imágenes sean simplemente más fáciles. |
| 26-10 | Figura `imagenes/heldout_resumen.png`, que faltaba: AUROC por clase y por método, más el efecto del eje temporal. Es la única figura de los puntos 2 y 5. |
| 26-10 | Escrito `docs/avance-pauta-28sep.tex`, informe de 9 páginas para el profe con los seis puntos, la corrección de folds y el hallazgo del eje temporal. Compila a PDF. |
