\chapter{Metodología}

La metodología propuesta se organiza en tres etapas secuenciales: preprocesamiento de los datos como imágenes, entrenamiento del VQ-VAE para aprender una representación discreta, y detección de anomalías con generación de contrafactuales mediante un modelo prior. A continuación se describe cada etapa en detalle.

\section{Preprocesamiento: conversión de curvas de luz a imágenes 2grid}

Las curvas de luz del dataset elasticc\_1 no se utilizan directamente como series de tiempo, sino que se convierten en imágenes RGB de 256×256 píxeles. Esta representación, denominada 2grid, organiza las seis bandas fotométricas (u, g, r, i, z, Y) como subgráficos apilados verticalmente, donde el eje horizontal codifica el tiempo en días julianos modificados (MJD) y el eje vertical el flujo fotométrico. Los límites de ambos ejes se fijan globalmente para garantizar comparabilidad entre objetos.

La imagen se genera con matplotlib usando \texttt{figsize=[2.56, 2.56]} pulgadas y DPI=100, produciendo imágenes de exactamente 256×256 píxeles. Las bandas sin observaciones se representan como rectángulos blancos. El proceso de generación se realizó en paralelo en el cluster Leftraru HPC, lanzando un job SLURM independiente por cada tipo de supernova, lo que permitió procesar aproximadamente 895.000 supernovas del dataset en paralelo.

Las imágenes generadas se almacenan en disco y se cargan durante el entrenamiento mediante un DataLoader estándar de PyTorch, lo que elimina la necesidad de generarlas on-the-fly y reduce significativamente el tiempo de entrenamiento. Antes de ser procesadas por el modelo, las imágenes se normalizan al rango $[-1, 1]$:

\begin{equation}
\hat{x} = \frac{x - 0.5}{0.5}
\end{equation}

Esta normalización es consistente con la función de activación Tanh del decoder, que también produce valores en $[-1, 1]$.

\section{Entrenamiento del VQ-VAE}

El VQ-VAE se entrena de forma no supervisada sobre el conjunto de imágenes 2grid. El objetivo es aprender una representación discreta de las curvas de luz que capture los patrones más frecuentes del dataset.

\subsection{Arquitectura}

La arquitectura sigue el diseño descrito en el capítulo de Marco Teórico:

\begin{itemize}
    \item \textbf{Encoder:} tres capas Conv2d con stride 2 y kernel 4 que reducen la resolución de 256×256 a 32×32, seguidas de dos ResBlocks. La salida es un tensor $\mathbf{z}_e \in \mathbb{R}^{256 \times 32 \times 32}$ con 1024 vectores de dimensión 256.
    
    \item \textbf{Codebook:} 512 embeddings de dimensión 256, inicializados con distribución uniforme. La cuantización se realiza por distancia euclídea mínima y el gradiente se propaga mediante el estimador straight-through.
    
    \item \textbf{Decoder:} espejo del encoder, con tres capas ConvTranspose2d con stride 2 que recuperan la resolución 32×32 → 256×256, seguidas de una convolución final y activación Tanh.
\end{itemize}

El modelo tiene aproximadamente 31 millones de parámetros entrenables.

\subsection{Configuración del entrenamiento}

El entrenamiento se realizó en el cluster Leftraru HPC, en la partición \texttt{mi210} con una GPU AMD Instinct MI210, usando PyTorch ROCm 6.3 y PyTorch Lightning 2.4. La configuración utilizada es la siguiente:

\begin{table}[H]
\centering
\begin{tabular}{|l|c|}
\hline
\textbf{Hiperparámetro} & \textbf{Valor} \\
\hline
Batch size & 32 \\
Learning rate & $2 \times 10^{-4}$ \\
Optimizador & Adam \\
Commitment cost ($\beta$) & 0.25 \\
Tamaño del codebook ($K$) & 512 \\
Dimensión del embedding ($D$) & 256 \\
Resolución latente & $32 \times 32$ \\
Máximo de épocas & 100 \\
Early stopping (patience) & 10 \\
\hline
\end{tabular}
\caption{Hiperparámetros del entrenamiento del VQ-VAE.}
\label{tab:vqvae_config}
\end{table}

El dataset se divide aleatoriamente en 90\% para entrenamiento (809.496 imágenes) y 10\% para validación (89.944 imágenes). El criterio de early stopping monitorea la pérdida de validación con paciencia de 10 épocas.

\section{Detección de anomalías}

Una vez entrenado el VQ-VAE, cada supernova queda representada como un mapa de índices discretos $\mathbf{s} \in \{1, \ldots, 512\}^{32 \times 32}$. El pipeline de detección de anomalías consta de dos pasos: entrenamiento de un modelo prior y cálculo del score de anomalía.

\subsection{Entrenamiento del prior}

Se entrena un modelo prior tipo MaskGIT sobre las secuencias de tokens del conjunto de entrenamiento. El prior aprende la distribución conjunta $p(\mathbf{s})$ sobre los 1024 tokens de cada imagen, capturando qué combinaciones de tokens son frecuentes (curvas de luz típicas) y cuáles son raras (posibles outliers).

\subsection{Score de anomalía}

Para cada supernova del conjunto de evaluación, se calcula su score de anomalía como la log-verosimilitud negativa bajo el prior:

\begin{equation}
\text{score}(\mathbf{s}) = -\log p(\mathbf{s})
\end{equation}

Las supernovas con score alto corresponden a objetos cuyas curvas de luz producen combinaciones de tokens poco probables bajo la distribución aprendida, y son candidatas a ser outliers.

\section{Generación de contrafactuales}

Para cada supernova identificada como outlier, se genera una explicación contrafactual que describe qué aspectos de su curva de luz son responsables de la anomalía.

El proceso es el siguiente:

\begin{enumerate}
    \item Se identifican las posiciones espaciales del mapa de tokens $\mathbf{s}$ con mayor contribución al score de anomalía, es decir, los tokens más improbables bajo el prior.
    
    \item Estas posiciones se enmascaran y se remuestrean condicionalmente al resto de los tokens visibles:
    \begin{equation}
    \mathbf{s}' \sim p(\mathbf{s}_{\text{mask}} \mid \mathbf{s}_{\text{visible}})
    \end{equation}
    
    \item La secuencia contrafactual $\mathbf{s}'$ se decodifica con el decoder del VQ-VAE para obtener la imagen contrafactual $\hat{\mathbf{x}}'$.
    
    \item Se compara $\hat{\mathbf{x}}'$ con la imagen original $\mathbf{x}$ para identificar las diferencias, que corresponden a las regiones temporales y bandas fotométricas que hacen al objeto anómalo.
\end{enumerate}

La calidad de los contrafactuales se evalúa según tres criterios: plausibilidad (el contrafactual debe ser similar a curvas de luz reales del dataset), parsimonia (las modificaciones deben ser mínimas) y consistencia temporal entre bandas (las modificaciones deben ser coherentes entre las seis bandas fotométricas).
