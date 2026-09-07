\section{Representación de curvas de luz como imágenes}

La representación de series de tiempo como imágenes ha sido explorada en distintos contextos de aprendizaje automático, con el objetivo de aprovechar la capacidad de las redes convolucionales para detectar patrones espaciales. En este trabajo, cada curva de luz multibanda se convierte en una imagen 2grid de 256×256 píxeles RGB, donde las seis bandas fotométricas (u, g, r, i, z, Y) se disponen verticalmente como subgráficos apilados. El eje horizontal codifica el tiempo en días julianos modificados (MJD), y el eje vertical el flujo fotométrico, ambos con límites fijos para garantizar comparabilidad entre objetos.

Esta representación presenta ventajas concretas frente al uso directo de series de tiempo: elimina la necesidad de manejar muestreo irregular, permite codificar de forma natural la correlación entre bandas como patrones espaciales en la imagen, y habilita el uso de arquitecturas convolucionales estándar sin modificaciones. La conversión se realiza con matplotlib usando \texttt{figsize=[2.56, 2.56]} pulgadas y DPI=100, produciendo imágenes de exactamente 256×256 píxeles.


\section{Autoencoder con cuantización vectorial (VQ-VAE)}

\subsection{Descripción general}

El autoencoder con cuantización vectorial (VQ-VAE, \textit{Vector Quantized Variational Autoencoder}) fue propuesto por van den Oord et al. (2017) como un modelo generativo que aprende representaciones discretas del espacio latente. A diferencia de los VAE estándar que producen representaciones continuas mediante el truco de la reparametrización, el VQ-VAE mapea cada vector latente a su vecino más cercano en un codebook de embeddings aprendidos. Esta discretización permite modelar distribuciones multimodales complejas sin colapso posterior y produce representaciones interpretables como secuencias de tokens, lo que facilita el uso de modelos de lenguaje sobre el espacio latente.

El VQ-VAE consta de tres componentes principales: un encoder, un codebook y un decoder.

\subsection{Encoder}

El encoder $E_\phi$ transforma una imagen de entrada $\mathbf{x} \in \mathbb{R}^{3 \times 256 \times 256}$ en un mapa de features continuo $\mathbf{z}_e \in \mathbb{R}^{D \times H \times W}$, donde $D$ es la dimensión del embedding y $H \times W$ es la resolución espacial del espacio latente.

En este trabajo, el encoder utiliza tres capas convolucionales con stride 2 y tamaño de kernel 4, que reducen la resolución espacial de 256×256 a 32×32, seguidas de dos bloques residuales (ResBlocks) que refinan las representaciones sin alterar la resolución:

\begin{align}
256 \times 256 &\xrightarrow{\text{Conv}(3 \to 64,\, s=2)} 128 \times 128 \\
128 \times 128 &\xrightarrow{\text{Conv}(64 \to 128,\, s=2)} 64 \times 64 \\
64 \times 64 &\xrightarrow{\text{Conv}(128 \to 256,\, s=2)} 32 \times 32 \\
32 \times 32 &\xrightarrow{\text{ResBlock} \times 2} 32 \times 32
\end{align}

El resultado es un tensor $\mathbf{z}_e \in \mathbb{R}^{256 \times 32 \times 32}$, es decir, 1024 vectores de dimensión 256, uno por cada posición espacial 32×32. Cada uno de estos vectores representa un parche de 8×8 píxeles de la imagen original.

\subsection{Cuantización vectorial (codebook)}

El codebook es un conjunto de $K$ vectores aprendidos $\{\mathbf{e}_k\}_{k=1}^{K} \subset \mathbb{R}^D$, donde $K$ es el tamaño del codebook (512 en este trabajo) y $D$ es la dimensión del embedding (256). Para cada vector continuo $\mathbf{z}_e(i,j)$ en la posición espacial $(i,j)$ del mapa latente, se selecciona el embedding más cercano:

\begin{equation}
k^* = \arg\min_k \|\mathbf{z}_e(i,j) - \mathbf{e}_k\|_2^2
\end{equation}

El vector cuantizado $\mathbf{z}_q(i,j) = \mathbf{e}_{k^*}$ reemplaza al vector continuo antes de ser pasado al decoder. El resultado es un mapa de índices enteros $\mathbf{s} \in \{1, \ldots, K\}^{H \times W}$, que representa la imagen como una secuencia discreta de 1024 tokens.

\textbf{Straight-through estimator.} La operación de argmin no es diferenciable, lo que impide el flujo de gradientes hacia el encoder. Para resolver esto, se utiliza el estimador straight-through: en el paso hacia adelante se usa $\mathbf{z}_q$, pero en el paso hacia atrás el gradiente se pasa directamente de $\mathbf{z}_q$ a $\mathbf{z}_e$ sin modificación, como si la cuantización fuera una función identidad.

\subsection{Decoder}

El decoder $G_\theta$ reconstruye la imagen a partir del mapa cuantizado $\mathbf{z}_q \in \mathbb{R}^{256 \times 32 \times 32}$, invirtiendo el proceso del encoder mediante capas de convolución transpuesta con stride 2:

\begin{align}
32 \times 32 &\xrightarrow{\text{ResBlock} \times 2} 32 \times 32 \\
32 \times 32 &\xrightarrow{\text{ConvT}(256 \to 256,\, s=2)} 64 \times 64 \\
64 \times 64 &\xrightarrow{\text{ConvT}(256 \to 128,\, s=2)} 128 \times 128 \\
128 \times 128 &\xrightarrow{\text{ConvT}(128 \to 64,\, s=2)} 256 \times 256 \\
256 \times 256 &\xrightarrow{\text{Conv}(64 \to 3)} 256 \times 256
\end{align}

La activación final es Tanh, que produce valores en el rango $[-1, 1]$, consistente con la normalización aplicada a las imágenes de entrada.

\subsection{Función de pérdida}

El VQ-VAE se entrena minimizando una función de pérdida compuesta por tres términos:

\begin{equation}
\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \beta \cdot \mathcal{L}_{\text{commitment}}
\end{equation}

donde:

\begin{itemize}
    \item $\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$ es la pérdida de reconstrucción, que mide la diferencia en el espacio de píxeles entre la imagen original y la reconstruida.
    
    \item $\mathcal{L}_{\text{codebook}} = \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|_2^2$ es la pérdida del codebook, que acerca los embeddings del codebook hacia los vectores del encoder ($\text{sg}[\cdot]$ denota stop-gradient).
    
    \item $\mathcal{L}_{\text{commitment}} = \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$ es la pérdida de compromiso, que acerca los vectores del encoder hacia los embeddings del codebook. El factor $\beta = 0.25$ controla su peso relativo.
\end{itemize}


\section{Modelo prior sobre el espacio latente discreto}

Una vez entrenado el VQ-VAE, cada imagen queda representada como una secuencia de 1024 tokens enteros $\mathbf{s} \in \{1, \ldots, 512\}^{32 \times 32}$. El objetivo del modelo prior es aprender la distribución conjunta $p(\mathbf{s})$ sobre estas secuencias, de modo que sea posible:

\begin{enumerate}
    \item Asignar una probabilidad $p(\mathbf{s})$ a cualquier secuencia de tokens (base para el score de anomalía).
    \item Muestrear nuevas secuencias plausibles $\mathbf{s} \sim p(\mathbf{s})$ (base para la generación de contrafactuales).
\end{enumerate}

En este trabajo se utiliza un prior tipo \textbf{MaskGIT} (\textit{Masked Generative Image Transformer}), que opera sobre la representación 2D de los tokens (32×32) y utiliza un mecanismo de atención bidireccional basado en el Transformer BERT. A diferencia de los modelos autoregresivos que generan tokens de izquierda a derecha, MaskGIT genera todos los tokens en paralelo mediante un proceso iterativo de enmascaramiento y completado.

\subsection{Score de anomalía}

Para un objeto con representación discreta $\mathbf{s}$, el score de anomalía se define como la log-verosimilitud negativa bajo el prior aprendido:

\begin{equation}
\text{score}(\mathbf{s}) = -\log p(\mathbf{s})
\end{equation}

Un objeto con una secuencia de tokens frecuente (curva de luz típica) tendrá un score bajo. Un outlier, cuya curva de luz produce combinaciones de tokens inusuales, tendrá un score alto. Este score es agnóstico al tipo de supernova y no requiere etiquetas de entrenamiento.

\subsection{Generación de contrafactuales}

Dado un objeto anómalo con secuencia $\mathbf{s}$, se identifican las posiciones de mayor contribución al score de anomalía (los tokens más improbables bajo el prior). Estas posiciones se enmascaran y se remuestrean condicionalmente al resto:

\begin{equation}
\mathbf{s}' \sim p(\mathbf{s}_{\text{mask}} \mid \mathbf{s}_{\text{visible}})
\end{equation}

El resultado $\mathbf{s}'$ es un contrafactual: una versión modificada del objeto que es plausible bajo la distribución aprendida. Al decodificar $\mathbf{s}'$ con el decoder del VQ-VAE se obtiene la curva de luz contrafactual en el espacio de imágenes, que puede ser comparada con el objeto original para identificar qué aspectos de la curva de luz son responsables de la anomalía.
