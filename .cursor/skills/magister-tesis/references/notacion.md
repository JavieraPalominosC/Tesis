# Notación y terminología — Tesis Javiera Palominos

## Términos técnicos (forma canónica)

| Concepto | Forma correcta en el texto | Nunca usar |
|----------|---------------------------|------------|
| El modelo principal | VQ-VAE | "el autoencoder", "el VAE", "el modelo VAE" |
| Las imágenes de entrada | imágenes 2grid | "representación 2D", "imágenes grid", "imágenes grilla" |
| El dataset | ELAsTiCC-1 | "ELAsTiCC", "elasticc", "el dataset ELAsTiCC" sin el -1 |
| El modelo generativo | prior MaskGIT | "el transformer", "el prior generativo", "MaskGIT" solo (incluir "prior") |
| Las observaciones temporales | curvas de luz | "series de tiempo" (en el cuerpo principal; aceptable en marco teórico cuando se compara con otras representaciones) |
| Los índices del codebook | tokens | "códigos", "índices", "embeddings" (embeddings son los vectores, no los índices) |
| La métrica de detección | score de anomalía | "puntaje", "puntuación", "score anómalo" |
| Las bandas fotométricas | bandas u, g, r, i, z, Y | no mezclar mayúsculas/minúsculas |
| El mecanismo de actualización | EMA (Exponential Moving Average) | "promedio exponencial" en texto técnico |
| El estimador | straight-through estimator | "STE" (solo si ya se definió el acrónimo) |
| El colapso | colapso del codebook | "codebook collapse" en español (usar traducción) |
| La métrica de uso | perplexity | "perplejidad" (mantener en inglés, es estándar en ML) |
| La supernova tipo Ia | SNIa | "SN Ia", "Type Ia" |
| La observación | Observatorio Vera C. Rubin | nunca abreviar como "Rubin" sin presentar antes |

---

## Términos de la etapa de detección y contrafactuales

Fijados al escribir los capítulos 4 y 5. Si aparece un sinónimo, es un error.

| Concepto | Forma correcta | Nunca usar |
|----------|---------------|------------|
| La codificación del flujo adoptada | `mad300` (en `\texttt{}`) | "asinh/MAD", "la codificación robusta", "U=300" a secas |
| La codificación anterior | codificación lineal (Ec. de percentiles) | "la codificación actual", "p01-p99", "waterfall" |
| La dispersión del objeto | MAD, definido como desviación absoluta mediana | "MAD score", "desviación mediana" |
| Las unidades de flujo | FLUXCAL | "flujo absoluto", "unidades de flujo" |
| La amplitud de la perturbación | $\Delta f$, en unidades del MAD del objeto | "df", "delta flujo", "amplitud" a secas |
| La amplitud en unidades físicas | $\overline{\Delta F}$ en FLUXCAL | mezclar $\Delta f$ y $\Delta F$ sin aclarar unidades |
| Las dos formas de máscara | máscara aleatoria / máscara de ventana | "random", "window", "máscara contigua", "parche" |
| La imagen sin perturbar | control | "baseline", "curva base", "original" |
| La zona donde se inyecta | zona perturbada / ventana perturbada | "región anómala", "zona objetivo" |
| El acierto de $\arg\max$ NLL | localización | "detección espacial", "hit espacial" |
| El límite con máscara perfecta | oráculo | "upper bound", "cota superior", "techo" a secas |
| El contrafactual que elimina la época | borrado | "eliminación", "pérdida de puntos", "supresión" |
| Cuánto de la anomalía tapa la máscara | cobertura | "recall de la máscara", "overlap" |
| El trazo dibujado en la banda equivocada | confusión de banda | "band confusion", "error de color" |
| La reparación en flujo | reducción del residuo | "mejora", "corrección", "fidelidad" |

**Cuidado con dos palabras que se prestan a confusión y significan cosas distintas:**

- **MAD** se usa en dos roles: como estadístico de escala dentro de la codificación (Ec. `mad300`)
  y como unidad de la amplitud inyectada ($\Delta f$ veces el MAD). Al escribir hay que decir cuál.
- **score de anomalía** es el **máximo** del NLL por token, no el promedio. El promedio diluye una
  anomalía localizada entre los 1024 tokens. Si un texto dice "promedio", está desactualizado.

---

## Notación de la etapa de detección

| Símbolo | Significado |
|---------|-------------|
| $\mathrm{NLL}_i$ | NLL de la posición $i$, promediado sobre las rondas en que fue enmascarada |
| $R = 24$ | Número de rondas de enmascaramiento al puntuar (NO 10; ese valor es de una versión vieja) |
| $\rho = 0.5$ | Fracción de posiciones enmascaradas por ronda |
| $\mathcal{M}_r$ | Conjunto de posiciones enmascaradas en la ronda $r$ |
| $\tau$ | Umbral de detección, percentil 95 de los scores de los controles |
| $U = 300$ | Cuántos MAD entran en el carril de la banda en `mad300` |
| $k = 10$ | Top-$k$ del muestreo al generar el contrafactual |
| $n = 10$ | Pasos de revelado progresivo del contrafactual |
| $\mathcal{R}$ | Rectángulo del estadístico de barrido que delimita la región a reescribir |

---

## Notación matemática estándar

### Variables principales

| Símbolo | Significado |
|---------|-------------|
| $\mathbf{x} \in \mathbb{R}^{3 \times 256 \times 256}$ | Imagen de entrada |
| $E_\phi$ | Encoder (parámetros $\phi$) |
| $G_\theta$ | Decoder (parámetros $\theta$) |
| $f_\psi$ | Cabeza de clasificación (parámetros $\psi$) |
| $\mathbf{z}_e \in \mathbb{R}^{D \times H \times W}$ | Mapa continuo pre-cuantización |
| $\mathbf{z}_q \in \mathbb{R}^{D \times H \times W}$ | Mapa cuantizado |
| $\{\mathbf{e}_k\}_{k=1}^K$ | Codebook con $K$ embeddings |
| $K = 512$ | Tamaño del codebook |
| $D = 256$ | Dimensión de los embeddings |
| $H = W = 32$ | Resolución espacial latente |
| $\mathbf{s} \in \{1,\ldots,K\}^{H \times W}$ | Secuencia de tokens (mapa de índices) |
| $\mathbf{s}' $ | Secuencia de tokens contrafactual |
| $\hat{\mathbf{x}}$ | Imagen reconstruida |
| $\hat{\mathbf{x}}'$ | Imagen contrafactual |
| $y \in \{1,\ldots,17\}$ | Etiqueta de tipo de supernova |
| $C = 17$ | Número de clases |
| $\gamma = 0.99$ | Factor de decay EMA |
| $\beta = 0.25$ | Peso del commitment loss |
| $\lambda$ | Peso del classification loss |

### Funciones de pérdida

```latex
\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2
\mathcal{L}_{\text{commitment}} = \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2
\mathcal{L}_{\text{class}} = \text{CrossEntropy}(f_\psi(\mathbf{z}_q), y)
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{commitment}}   % modelo base
\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \lambda \cdot \mathcal{L}_{\text{class}}   % con clasificación
```

### Score y contrafactuales

```latex
\text{score}(\mathbf{s}) = -\log p(\mathbf{s})
\mathbf{s}' \sim p(\mathbf{s}_{\text{mask}} \mid \mathbf{s}_{\text{visible}})
```

---

## Convenciones LaTeX

- Vectores y matrices: `\mathbf{x}`, `\mathbf{z}_e`, `\mathbf{e}_k` (negrita)
- Conjuntos: `\mathbb{R}`, `\mathbb{Z}`
- Calligráfico para pérdidas: `\mathcal{L}`
- Bandas en itálica cuando se nombran individualmente: `\textit{u}`, `\textit{g}`, etc.
- Tablas con resultados: usar `[H]` y `\centering` siempre
- Figuras: usar `\insertimage` del template ppizarror
- Referencias cruzadas: `\ref{fig:X}`, `\ref{tab:X}`, `\ref{sec:X}` con tilde de no-break: `Figura~\ref{fig:X}`
- Citas: `\cite{autor2024keyword}` formato BibTeX

---

## Acrónimos (definir en primera mención, luego solo el acrónimo)

| Acrónimo | Definición completa |
|----------|---------------------|
| VQ-VAE | Vector Quantized Variational Autoencoder |
| EMA | Exponential Moving Average |
| MaskGIT | Masked Generative Image Transformer |
| ELAsTiCC | Extended LSST Astronomical Time Series Classification Challenge |
| LSST | Legacy Survey of Space and Time |
| MJD | Modified Julian Date |
| UMAP | Uniform Manifold Approximation and Projection |
| ROC | Receiver Operating Characteristic |
| AUROC | Area Under the ROC Curve |
| ResBlock | Residual Block |
| MLP | Multi-Layer Perceptron |
| HPC | High-Performance Computing |
| SN | Supernova (plural: SNe) |

---

## Tipos de supernovas en el dataset

Exactamente como aparecen en ELAsTiCC-1 (usar estos nombres, no traducir):
SNIa-SALT2, SNIa-91bg, SNIax, SNII-Templates, SNII-NMF, SNII+HostXT\_V19,
SNIb-Templates, SNIb+HostXT\_V19, SNIc-Templates, SNIc+HostXT\_V19,
SNIcBL+HostXT\_V19, SNIIb+HostXT\_V19, SNIIn-MOSFIT, SNIIn+HostXT\_V19,
SLSN-I+host, SLSN-I\_no\_host, PISN
