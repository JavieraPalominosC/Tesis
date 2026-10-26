# Guía de estilo humano — Tesis Javiera Palominos

El objetivo es que el texto suene como lo escribiría una investigadora de FCFM con dominio del tema,
no como lo generaría un modelo de lenguaje. La diferencia más importante no está en las palabras
individuales sino en la estructura y el ritmo del texto.

---

## Frases y estructuras prohibidas (huelen a IA)

### Conectores de relleno
Eliminar siempre. No agregan información.

- "Es importante destacar que..."
- "Cabe señalar que..."
- "Cabe mencionar que..."
- "En este sentido..."
- "En el contexto de..."
- "En este contexto..."
- "En el marco de..."
- "A su vez..."
- "Por otro lado..." (aceptable solo si hay un contraste real, no como conector de transición)
- "Asimismo..."
- "Del mismo modo..."
- "De esta manera..."
- "De esta forma..."
- "En definitiva..."
- "En conclusión..." (fuera de la sección de Conclusiones)
- "En resumen..." (fuera de resúmenes explícitos)
- "Dado que..." (al inicio de párrafo)
- "Teniendo en cuenta que..."
- "Con el objetivo de..."
- "Con el fin de..."
- "A fin de..."

### Adjetivos vacíos
Borrar directamente o reemplazar por un hecho técnico concreto.

- "robusto/a" → decir qué hace que sea robusto, o simplemente no decirlo
- "eficiente" → decir en qué sentido (tiempo, memoria, convergencia)
- "novedoso/a" / "innovador/a" → si la novedad importa, explicarla
- "prometedor/a" → no usar en cuerpo de tesis, solo en discusión y con evidencia
- "poderoso/a"
- "sofisticado/a"
- "fundamental" / "crucial" / "clave" → el texto técnico muestra la importancia sin decirla

### Construcciones con olor a IA
- "juega un papel fundamental/crucial/clave"
- "resulta evidente que" / "es evidente que"
- "permite [verbo]" encadenados: "permite capturar... permite modelar... permite evaluar..."
- Triadas retóricas: "X, Y y Z" donde los tres elementos no aportan igual
- Párrafos que terminan con una oración que resume lo que se acaba de decir
- Enumeraciones en prosa: "Primero,... Segundo,... Finalmente,..." → usar `\begin{enumerate}` o reescribir

### Estructura de párrafo típica de IA (evitar)
```
[Afirmación general]. [Afirmación general ligeramente más específica]. 
[Afirmación aún más específica]. [Oración de cierre que repite la idea inicial].
```

### Inicio de párrafo típico de IA (evitar)
- Párrafos que empiezan con el sujeto + "es" + adjetivo: "El VQ-VAE es un modelo..."
- Párrafos que empiezan con "Uno de los principales..."

---

## Dos reglas que se relajaron (contrastadas contra una memoria FCFM real)

Al revisar la memoria de un compañero de Javiera (ver `registro-memoria-fcfm.md`) resultó que dos
prohibiciones de esta guía eran demasiado estrictas para el género.

**"se observa que" / "se puede observar que" está permitido.** Aparece 21 y 32 veces en esa memoria.
Es fraseo estándar del español académico de ingeniería, no una marca de IA. Lo que sí sobra es el
preámbulo cuando no agrega nada:

❌ "Se puede observar que existe una diferencia entre ambos valores."
✅ "Se observa que la demanda real del sector industrial es considerablemente menor que la proyectada."

La segunda dice algo; la primera solo anuncia que va a decir algo.

**Empezar un párrafo con "Para + infinitivo" está permitido** cuando introduce un propósito real:

✅ "Para evaluar el desempeño del modelo, en la Tabla \ref{tab:X} se muestra..."

Lo que sigue prohibido es "Con el objetivo de" y "Con el fin de", que son la versión inflada de lo
mismo y no aparecen en ninguno de los dos textos de referencia.

---

## Qué sí hacer

### Ritmo

⚠️ Corregido con la muestra real de Javiera. Manda la sección "Voz real de Javiera" de más abajo.

- Sus párrafos son de 1 a 3 oraciones, pero **oraciones largas**, encadenadas con comas.
- **No** escribir frases cortas y contundentes tipo eslogan. No es su registro.
- No todos los párrafos tienen que tener la misma estructura.

### Voz impersonal académica en español
El español académico de ingeniería/ciencias usa el "se" impersonal de forma natural. Javiera lo usa
de forma marcada, incluso donde la primera persona sería lo esperable:
- "se entrena el modelo", "se evalúa la calidad", "se observa un aumento"
- En su informe de práctica escribe "se tuvo que viajar a Dinamarca" y "se estuvo ahí", nunca "viajé".
- **Nunca usar primera persona**, ni singular ni plural ("entrenamos el modelo" no es su registro).

### Afirmaciones directas
En lugar de "se puede observar que la perplexity aumenta", escribir "la perplexity sube de 162 a 335".
Números y hechos concretos reemplazan adverbios y adjetivos.

### Referencias cruzadas naturales
- "como se muestra en la Figura~\ref{fig:X}"
- "ver Tabla~\ref{tab:Y}"
- "descrito en la Sección~\ref{sec:Z}"

No es necesario introducir cada referencia con "A continuación se presenta...".

### Modalizadores honestos
En resultados parciales o preliminares, está bien (y es académicamente correcto) usar:
- "sugiere que", "consistente con", "compatible con"
- "los resultados preliminares indican" (cuando son preliminares de verdad)
- NO: "claramente", "obviamente", "sin duda"

---

## Ejemplos reales de esta tesis (antes / después)

Sacados de ediciones concretas al documento. Sirven mejor que la lista de prohibiciones porque
muestran qué reemplaza a qué.

**Adjetivo vacío por el mecanismo concreto**

❌ "reemplaza los percentiles por centro y escala robustos del propio objeto"
✅ "reemplaza los percentiles por la mediana y la desviación absoluta mediana del propio objeto"

**Em-dash por coma o paréntesis**

❌ "los tipos estándar ---cuyas curvas de luz son similares entre sí--- permanecen mezclados"
✅ "los tipos estándar, cuyas curvas de luz son similares entre sí, permanecen mezclados"

**Preámbulo por afirmación directa con número**

❌ "se puede observar que el recall mejora considerablemente al puntuar con ventana"
✅ "puntuar con ventana levanta el recall en $\Delta f = 100$ de 0.13 a 0.80"

**Cierre redundante: borrarlo**

❌ "...y la correlación sube a 0.60. Estos resultados muestran que la nueva codificación es superior."
✅ "...y la correlación sube a 0.60."

**Frase corta con peso, en vez de subordinada larga**

✅ "El outlier arrastra la escala que debería medirlo."
✅ "El máximo, y no el promedio, porque una anomalía localizada afecta unos pocos tokens de los 1024."

---

## Honestidad sobre resultados

Esta tesis tiene varios resultados parciales o negativos, y el texto queda mejor cuando los dice
en vez de esquivarlos. Patrones que ya están en el documento y conviene mantener:

- Nombrar el fallo con su número: "el contrafactual elimina la observación en el 29% de las épocas
  reescritas".
- Decir cuándo una elección se invierte: "la ventaja del rectángulo dilatado se invierte en
  anomalías pequeñas".
- Separar lo medido de lo que se querría concluir: "lo medido es la sensibilidad del método a una
  perturbación controlada", no "el método detecta supernovas anómalas".
- Cuando dos cifras no son comparables, decirlo antes de que el lector las compare: "las dos
  pérdidas no son comparables entre sí".

---

## Voz real de Javiera

Derivado de `muestra-javiera.md`, su informe de práctica, que es la única muestra verificada.
**Esta sección manda sobre el resto de la guía cuando hay conflicto.**

Recordatorio: la prosa que hay en `thesis-local/Tesis docs/` pasó por el agente y **no** sirve
para calibrar estilo.

### 1. Oraciones largas encadenadas con comas

Es el rasgo más marcado y el que la guía tenía al revés. Ella no escribe frases cortas de remate.

Ejemplo suyo, una sola oración:

> "Esta práctica tuvo una duración de 10 semanas con un horario de 8:00 a 16:00 de lunes a jueves
> y de 8:00 a 13:00, los dias viernes, haciendo así una jornada de 37 horas laborales por semana,
> por lo que la practica en total duró aproximadamente 370 horas."

Encadena con comas, gerundios ("haciendo así") y "por lo que". Cierra la idea completa en una
oración en vez de partirla en tres.

❌ No suena a ella: "El outlier arrastra la escala que debería medirlo."
✅ Suena a ella: "El percentil 99 queda fijado por el punto más brillante de la curva, por lo que
un outlier suficientemente grande termina definiendo la escala con la que se lo mide."

### 2. "además de" es su conector

Lo usa tres veces en un texto corto. Es su forma natural de agregar información. Está permitido y
es preferible a "asimismo", "por otro lado" o "a su vez", que ella no usa.

⚠️ Es **aditivo**, no contrastivo. No sirve para oponer dos cosas. Para contraste usar "mientras
que" o "en cambio". Forzarlo donde el sentido es de oposición rompe la lógica de la frase.

Sus otros encadenadores, vistos en la muestra: "por lo que", "y gracias a eso", "tanto ... como",
y gerundios ("haciendo así").

### 3. Enumeraciones largas y planas, no triadas

Enumera seis o más elementos seguidos sin buscar simetría retórica:

> "energía eólica, solar, hidráulica, mareomotriz, geotérmica y biomasa, además de temas
> relacionados como transporte, calefacción distrital, materiales de construcción ecológicos..."

La advertencia de la guía contra las "triadas retóricas" es correcta, pero por la razón opuesta a
la que parecía: ella no agrupa de a tres, lista todo lo que hay.

### 4. Referencias a figuras: la referencia va primero

Su patrón: **"En la Figura \ref{fig:X} se muestra ..."**

No usa "como se muestra en la Figura~\ref{fig:X}". Preferir el orden de ella.
(El `~` de no-break sí conviene mantenerlo aunque ella no lo ponga; es correcto tipográficamente
y no se nota en el texto.)

### 5. Propósito con "para" + infinitivo

"para cubrir las necesidades", "para mantener informados". Nunca "con el objetivo de" ni "con el
fin de", que la guía ya prohibía y ella efectivamente no usa.

### 6. Ítems de lista que empiezan con sustantivo

Sus `\item` son sintagmas nominales, no oraciones con verbo conjugado:

> "Programas y actividades de formación y educación, diseñados para cubrir las necesidades..."
> "Apoyo a emprendedores, pequeñas y medianas empresas verdes en el desarrollo y pruebas..."

### 7. Lo que se confirma de la guía original

Estas prohibiciones estaban bien y su texto las cumple sin excepción: **no** usa em-dashes, **no**
usa "Es importante destacar", "Cabe señalar", "En este sentido", "Por otro lado" ni "De esta
manera", y **no** cierra los párrafos resumiendo lo que acaba de decir.

### 8. Lo que NO se debe copiar de la muestra

Son descuidos, no estilo:

- Tildes faltantes ("energias", "practica", "dias", "politicos", "formacion")
- Meses y gentilicios con mayúscula/minúscula inconsistente ("Enero", "dinamarca")
- "Está practica" por "Esta práctica", "de el poder" por "del poder"

El `\\` al final de cada párrafo no es descuido suyo: el compañero de FCFM hace lo mismo, así que es
hábito local compartido. Aun así los capítulos de la tesis usan línea en blanco, y conviene
mantener esa forma por consistencia interna del documento.

### Lo que la muestra no dice

Es un informe descriptivo, no argumentativo. No hay evidencia de cómo discute un resultado
negativo, cómo contrasta dos métodos ni cómo maneja los modalizadores ("sugiere que", "consistente
con"). Para esas partes seguir las reglas generales de la guía y, si aparece una muestra
argumentativa suya, actualizar esta sección.

---

## Checklist antes de entregar texto

- [ ] ¿Algún párrafo empieza con frase de relleno?
- [ ] ¿Hay adjetivos vacíos (robusto, clave, eficiente) sin respaldo técnico?
- [ ] ¿Algún párrafo termina resumiendo lo que recién dijo?
- [ ] ¿Hay triadas retóricas (X, Y y Z) donde los tres elementos no pesan igual?
- [ ] ¿El ritmo de párrafos es variado?
- [ ] ¿Los números están presentes donde deberían estar?
- [ ] ¿La notación es consistente con `notacion.md`?
