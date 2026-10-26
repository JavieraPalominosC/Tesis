# Registro de memoria FCFM — referencia de género

Derivado de la memoria de un compañero de Javiera (Departamento de Ingeniería Eléctrica, FCFM,
mismo template ppizarror). Javiera la aportó como modelo de escritura el 9 de septiembre de 2026.

## Para qué sirve y para qué NO

**Sirve** para las convenciones del género que la muestra de Javiera no cubre, porque su informe de
práctica es descriptivo y esta memoria es argumentativa: cómo se introduce una tabla de resultados,
cómo se discute una discrepancia entre lo esperado y lo medido, cómo se declara una limitación,
cómo se estructura el capítulo de conclusiones.

**No sirve** como modelo de voz. La voz de Javiera está en `muestra-javiera.md` y manda sobre esto.
Mezclar dos voces produce un texto que no suena a ninguna de las dos.

⚠️ Copiar convenciones y estructura, **nunca frases ni contenido**. Es la memoria de otra persona.

---

## Estadísticas

| | Mediana | Media | Oraciones < 12 palabras |
|---|---|---|---|
| Memoria del compañero | 27 palabras | 36.3 | 3% |
| Informe de Javiera | 26 | 27.4 | 7% |

Ambos escriben largo. La media de 36 del compañero refleja una cola de oraciones muy largas, de 50
palabras o más, en las secciones de análisis. Confirma que las frases cortas de remate no
pertenecen a este registro.

---

## Convenciones que conviene adoptar

### 1. La referencia a tabla o figura va al inicio de la oración

> "En la Tabla \ref{tab:balance2020} se muestra la demanda de energía de la Región Metropolitana
> durante el año 2020..."

Coincide exactamente con el patrón de Javiera ("En la Figura \ref{fig:ola} se muestra..."). Es la
forma correcta para esta tesis.

### 2. Discutir la discrepancia, no solo reportarla

Su patrón al comparar lo medido con lo esperado: constata la diferencia, propone la causa, y de ahí
deriva una limitación del método.

> "Se espera que la demanda energética sea estrictamente creciente en el tiempo, pero al comparar
> [...] se puede observar que la demanda energética real del año 2020 es menor que la del año 2017,
> lo cual se puede atribuir a la pandemia de COVID-19. Esto explica las diferencias entre los
> valores reales y proyectados, pero también ilustra una de las limitaciones del modelo."

### 3. Declarar la limitación y decir cuál sería el ideal

> "Sin embargo, el año 2017 no necesariamente es representativo [...], en cambio, idealmente el
> modelo se ajustaría utilizando varios años en los que se conozca la demanda de energía [...]"

### 4. Conclusiones ancladas a los objetivos, con "Se logró"

Abre mapeando conclusiones contra los objetivos de la introducción, un ítem por objetivo:

> "Durante el desarrollo de este trabajo de memoria se cumplieron todos los objetivos planteados en
> la Introducción:"
> - "Se logró determinar variables que permiten calcular la demanda de energía [...]"
> - "Se logró implementar un modelo que utiliza la información disponible [...]"

Después declara explícitamente los aportes: *"Al respecto, se lograron los siguientes aportes al
estado del arte:"*

**Esto falta en la tesis de Javiera.** Sus conclusiones actuales no se mapean contra los cinco
objetivos específicos de `01_introduccion.tex`.

### 5. Autorreferencia al documento

"este trabajo" (12 veces), "esta memoria" (11 veces). Nunca primera persona.

### 6. Sus conectores

`por lo que` (33), `luego` (27), `sin embargo` (15), `además` (14), `por otro lado` (11),
`por lo tanto` (10), `debido a` (9), `además de` (5), `lo cual`, `en cambio`, `al respecto`,
`tales como`, `con respecto a`.

Coincide con Javiera en `por lo que`, `además de` y `tales como`.

**No aparecen nunca:** "asimismo", "en este sentido", "cabe señalar", "es importante destacar",
"de esta manera". Las prohibiciones de `estilo-humano.md` se sostienen contra un texto real.

### 7. Cero em-dashes

Confirmado en las 2.686 líneas. Usa paréntesis y comas.

---

## Convenciones propias de la memoria FCFM

- Sección de reflexión personal al final: *"Comentarios generales sobre la experiencia de este
  trabajo de memoria"*, donde se comenta qué se aprendió. Es del género, no del autor.
- Sección *"Pasos a seguir"* en vez de "Trabajo futuro".
- Tablas anchas envueltas en `\resizebox{\textwidth}{!}{...}`.
- `\\` al final de párrafo. Javiera hace lo mismo, así que es hábito local compartido y no un
  descuido individual. Aun así, los capítulos de la tesis usan línea en blanco: mantener esa forma
  por consistencia interna del documento.
