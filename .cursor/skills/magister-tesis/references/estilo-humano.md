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
- "se puede observar que" (si lo que sigue es una afirmación directa, quitar el preámbulo)
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
- Párrafos que empiezan con "Para [verbo en infinitivo]..."
- Párrafos que empiezan con "Uno de los principales..."

---

## Qué sí hacer

### Ritmo
- Mezclar párrafos cortos (2-3 oraciones) con párrafos más largos cuando la explicación lo requiere.
- Una oración sola está bien si tiene peso.
- No todos los párrafos tienen que tener la misma estructura.

### Voz impersonal académica en español
El español académico de ingeniería/ciencias usa el "se" impersonal de forma natural. Es correcto y no suena a IA si no se abusa:
- "se entrena el modelo", "se evalúa la calidad", "se observa un aumento"
- Evitar la voz activa forzada cuando no es natural en español ("entrenamos el modelo" puede sonar bien o mal según el contexto)

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

## Checklist antes de entregar texto

- [ ] ¿Algún párrafo empieza con frase de relleno?
- [ ] ¿Hay adjetivos vacíos (robusto, clave, eficiente) sin respaldo técnico?
- [ ] ¿Algún párrafo termina resumiendo lo que recién dijo?
- [ ] ¿Hay triadas retóricas (X, Y y Z) donde los tres elementos no pesan igual?
- [ ] ¿El ritmo de párrafos es variado?
- [ ] ¿Los números están presentes donde deberían estar?
- [ ] ¿La notación es consistente con `notacion.md`?
