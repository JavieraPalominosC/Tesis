---
name: magister-tesis
description: >
  Skill para trabajar en la tesis de magíster de Javiera Palominos Clarke (Universidad de Chile, FCFM).
  Tesis: "Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE y Generación de Ejemplos Contrafactuales".
  Úsala cuando Javiera pida: redactar o expandir secciones de la tesis, agregar contenido nuevo, revisar texto
  existente y humanizarlo, mantener consistencia de notación, o saber a qué archivo pertenece un contenido dado.
  Actívala ante cualquier mención de la tesis, Overleaf, los capítulos, VQ-VAE, ELAsTiCC, curvas de luz,
  contrafactuales, o redacción académica en el contexto de este proyecto.
---

# Skill: Tesis de Magíster — Javiera Palominos

## Contexto del proyecto

**Título:** Detección de Outliers en Curvas de Luz de Supernovas con VQ-VAE y Generación de Ejemplos Contrafactuales  
**Institución:** Universidad de Chile, Facultad de Ciencias Físicas y Matemáticas  
**Autora:** Javiera Palominos Clarke  
**Plataforma:** Overleaf (LaTeX, template ppizarror v3.4.0)  
**Copia local en este repo:** `thesis-local/Tesis docs/`

---

## Estructura de archivos

En este workspace los `.tex` están en `thesis-local/Tesis docs/`. En Overleaf la misma estructura vive en la raíz del proyecto.

```
thesis-local/Tesis docs/
├── capitulos/
│   ├── 01_introduccion.tex       → Cap. 1: Introducción
│   ├── 02_marco_teorico.tex      → Cap. 2: Marco Teórico
│   ├── 03_metodologia.tex        → Cap. 3: Dataset + Cap. 4: Metodología (mismo archivo)
│   ├── 04_resultados.tex         → Cap. 4: Resultados
│   └── 05_discusion_conclusiones.tex → Cap. 5: Discusión y Conclusiones
├── main.tex                      → Portada, resumen, estructura general
└── library.bib                   → Referencias bibliográficas
```

### Qué va en cada archivo

| Contenido | Archivo |
|-----------|---------|
| Motivación, justificación, objetivos | `01_introduccion.tex` |
| Fundamentos teóricos: VQ-VAE, MaskGIT, anomaly detection, representación 2grid | `02_marco_teorico.tex` |
| ELAsTiCC-1, análisis exploratorio, preprocesamiento | `03_metodologia.tex` (sección Dataset) |
| Pipeline completo, decisiones de diseño, hiperparámetros | `03_metodologia.tex` (sección Metodología) |
| Tablas y figuras de resultados de experimentos | `04_resultados.tex` |
| Análisis crítico de resultados, limitaciones, conclusiones, trabajo futuro | `05_discusion_conclusiones.tex` |
| Resumen ejecutivo (~200 palabras) | `main.tex` → bloque `\begin{abstractd}` |

---

## Reglas de escritura — estilo humano estricto

Lee `references/estilo-humano.md` para las reglas completas. Resumen de lo más crítico:

- **Nunca** iniciar párrafo con "Es importante destacar", "Cabe señalar", "En este sentido", "En el contexto de", "En conclusión", "En resumen", "Dado que", "A su vez", "Por otro lado", ni ningún conector de IA.
- **Nunca** usar construcciones como "se puede observar que", "resulta evidente que", "es fundamental", "es crucial", "juega un papel fundamental/crucial/clave".
- **Nunca** listar en prosa con "Primero... Segundo... Finalmente..." cuando puede fluir en un párrafo o en un `\begin{enumerate}`.
- **Evitar** la voz activa forzada estilo IA. El español académico usa impersonal con "se": "se entrena", "se evalúa", "se observa". Pero sin abusar.
- **Evitar** el exceso de adjetivos: "robusto", "eficiente", "innovador", "novedoso", "prometedor". Si no aportan información técnica, sobran.
- **Nunca** usar em-dashes (---) como sustituto de paréntesis o comas. Usar comas, paréntesis o punto y coma según corresponda.  
  ❌ "los enfoques tradicionales ---X, Y y Z--- resultan..."  
  ✅ "los enfoques tradicionales (X, Y y Z) resultan..."
- La longitud de los párrafos debe variar. No todos los párrafos tienen la misma estructura de 3 oraciones.
- **Oraciones largas, encadenadas con comas.** Javiera no escribe frases cortas de remate; una frase
  tipo eslogan es la marca más clara de que el texto no es suyo. Ver `references/muestra-javiera.md`,
  la única muestra verificada de su escritura, y la sección "Voz real de Javiera" de
  `references/estilo-humano.md`, que manda sobre el resto de la guía.
- **Nunca primera persona.** Usa "se" impersonal incluso donde sonaría forzado ("se tuvo que viajar").
- "además de" es su conector natural para agregar información.

---

## Reglas de notación y términos (consistencia)

Lee `references/notacion.md` para la lista completa. Lo más usado:

| Término | Forma correcta |
|---------|---------------|
| El modelo | VQ-VAE (nunca "el modelo VAE" ni "el autoencoder") |
| Las imágenes | imágenes 2grid (nunca "representación 2D" ni "imágenes grid") |
| El dataset | ELAsTiCC-1 (con guión, mayúsculas exactas) |
| El prior | prior MaskGIT (nunca "el transformer" ni "el prior generativo") |
| Las curvas | curvas de luz (nunca "series temporales" en el cuerpo principal) |
| Los tokens | tokens (nunca "códigos" ni "índices del codebook") |
| Score | score de anomalía (nunca "puntaje" ni "puntuación") |
| Las bandas | u, g, r, i, z, Y (siempre en itálica si se mencionan individualmente: \textit{u}, \textit{g}, etc.) |

**Notación matemática estándar:**
- Encoder: $E_\phi$, Decoder: $G_\theta$, clasificador: $f_\psi$
- Mapa continuo: $\mathbf{z}_e$, mapa cuantizado: $\mathbf{z}_q$
- Secuencia de tokens: $\mathbf{s} \in \{1,\ldots,K\}^{H \times W}$, con $K=512$, $H=W=32$
- Codebook: $\{\mathbf{e}_k\}_{k=1}^K$
- Score: $\text{score}(\mathbf{s}) = -\log p(\mathbf{s})$
- Contrafactual: $\mathbf{s}'$, imagen contrafactual: $\hat{\mathbf{x}}'$

---

## Regla dura: ningún número sin fuente

El riesgo más alto al escribir esta tesis no es que el texto suene a IA. Es escribir una cifra que
ya no corresponde, o que contradice otra parte del documento. Ha pasado varias veces.

**Antes de escribir cualquier número, hay que abrirlo en su fuente.** Nunca copiarlo de otra sección
del `.tex`, de un resumen de conversación, ni de memoria.

| Qué se quiere citar | Dónde está la verdad |
|---------------------|----------------------|
| Métricas de detección, contrafactuales, codificación | `imagenes/*.json` en la raíz del repo |
| Hiperparámetros de entrenamiento | `configs/*.yaml` (ojo: `cls_weight` es el $\lambda$) |
| Pérdidas y epoch del mejor modelo | nombre del checkpoint en `results/**/checkpoints/` |
| Tamaño de los conjuntos | `data/enc_cmp/mad300/folds_full.json` |
| Cómo se calcula una métrica | el script en `scripts/` que la produjo |

Errores reales que esta regla habría evitado:

- La tabla de metodología decía $\lambda = 1.0$; los YAML y el capítulo 4 dicen 0.1.
- La metodología describía el score con $R = 10$ y como promedio de NLL; el código usa $R = 24$
  y el máximo.
- `dominio.md` de esta misma skill tenía $\lambda = 1.0$ y propagó el error.

**Al terminar de escribir, verificar coherencia cruzada:** si una cifra aparece en dos capítulos,
tiene que ser la misma. Buscarla con `rg` antes de dar por cerrado el trabajo.

---

## Flujo de trabajo cuando se pide agregar/editar contenido

1. **Identificar el archivo destino** usando la tabla de estructura de arriba.
2. **Leer el archivo completo** (o la sección relevante) para entender el contexto inmediato, el tono, y evitar redundancias.
3. **Abrir las fuentes de los números** que se van a citar (ver la regla dura de arriba).
4. **Revisar notacion.md y estilo-humano.md** antes de redactar.
5. **Redactar** en LaTeX directamente: incluir `\section{}`, `\subsection{}`, ecuaciones, tablas, figuras según corresponda.
6. **Compilar para verificar**, con el envoltorio mínimo descrito más abajo. Revisar que no queden
   `Overfull \hbox` (tablas que se salen del margen) ni referencias rotas.
7. Si hay un `% TODO` relevante en el archivo, marcarlo como resuelto quitando el comentario o reemplazándolo.

---

## Figuras y compilación

- Los scripts dejan las figuras en `imagenes/` en la raíz del repo. Las que se referencian en un
  `.tex` hay que copiarlas a `thesis-local/Tesis docs/img/` y **avisarle a Javiera que las suba a
  Overleaf**, porque el repo local y Overleaf no están sincronizados.
- **Antes de escribir un pie de figura, abrir la imagen con la herramienta de lectura.** Un pie que
  describe paneles que no existen es peor que no ponerlo.
- Macro de figura: el template define `\insertimage[\label{...}]{ruta}{opciones}{caption}`. El
  capítulo 4 tiene además figuras viejas con `\includegraphics` dentro de `figure`. **Las dos
  formas compilan**; usar `\insertimage` en material nuevo y no reescribir las viejas solo por
  uniformidad.
- El template **no compila** con la instalación local de LaTeX (falla en `\AtBeginShipout`; es un
  problema preexistente de la instalación, no del texto). Para validar sintaxis, compilar los
  capítulos sueltos con un envoltorio mínimo que defina `\insertimage` como stub y apunte
  `\graphicspath` a `thesis-local/Tesis docs/`.

---

## Flujo de trabajo cuando se pide revisar/humanizar texto

1. Leer `references/estilo-humano.md` para tener presentes todos los patrones a detectar.
2. Identificar frases y estructuras con olor a IA (ver lista en el reference).
3. Reescribir manteniendo el contenido técnico exacto — no parafrasear conceptos, solo cambiar la forma.
4. No agregar contenido nuevo durante una revisión, salvo que Javiera lo pida explícitamente.
5. Entregar el bloque revisado completo en LaTeX, listo para reemplazar.

---

## Secciones pendientes

- `main.tex`: completar resumen, dedicatoria, agradecimientos
- `04_resultados.tex`: figura de reconstrucciones 2grid (pares original/reconstruido)
- `05_discusion_conclusiones.tex`: subsección "Organización semántica del codebook" (UMAP antes/después, si la pureza mejora los contrafactuales)

Ya escritas: codificación del flujo, prior MaskGIT, detección de anomalías y contrafactuales en
`04_resultados.tex`; limitaciones, calidad de los contrafactuales, conclusiones y trabajo futuro en
`05_discusion_conclusiones.tex`.

## Figuras

Las figuras generadas por los scripts viven en `imagenes/` en la raíz del repo. Las que ya están
referenciadas en los `.tex` se copiaron a `thesis-local/Tesis docs/img/` y hay que subirlas a Overleaf.
El template local no compila con la instalación de LaTeX de la máquina (falla en `\AtBeginShipout`,
problema preexistente); para validar sintaxis se puede compilar los capítulos con un envoltorio
mínimo que defina `\insertimage` como stub.

---

## Conocimiento del dominio

Lee `references/dominio.md` si necesitas contexto sobre el dominio astronómico, el modelo, o decisiones de diseño que ya están fijadas en la tesis.

## Archivos de referencia de esta skill

| Archivo | Cuándo leerlo |
|---------|---------------|
| `references/estilo-humano.md` | Siempre antes de redactar o revisar prosa |
| `references/muestra-javiera.md` | Al calibrar voz. Única muestra verificada de su escritura |
| `references/registro-memoria-fcfm.md` | Convenciones de género: cómo se discute un resultado, cómo se cierran las conclusiones. Referencia de otra persona, no de voz |
| `references/notacion.md` | Siempre antes de redactar; términos y símbolos fijados |
| `references/dominio.md` | Contexto astronómico, arquitectura y estado del experimento |
