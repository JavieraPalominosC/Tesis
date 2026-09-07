\chapter{Metodología}

La metodología propuesta se organiza en tres etapas secuenciales: preprocesamiento de los datos como imágenes, entrenamiento del VQ-VAE para aprender una representación discreta, y detección de anomalías con generación de contrafactuales mediante un modelo prior. A continuación se describe cada etapa en detalle.

\section{Exploración metodológica previa}

Antes de adoptar el enfoque basado en imágenes descrito en este trabajo, se realizó una exploración inicial utilizando series de tiempo directamente como input del VQ-VAE, siguiendo el esquema del modelo TimeVQVAE \cite{lee2023timevqvae}. Esta sección describe ese proceso y los resultados que motivaron el cambio de representación.

\subsection{Enfoque inicial: series de tiempo con TimeVQVAE}

El punto de partida fue el código de TimeVQVAE, que utiliza la transformada de Fourier para convertir la serie de tiempo al espacio tiempo-frecuencia antes de pasarla al VQ-VAE. Sin embargo, para datos de supernovas esta representación no es adecuada, ya que las curvas de luz tienen muestreo irregular y características morfológicas (el peak de brillo, el tiempo de caída) que no se benefician de una descomposición en frecuencias. Por esta razón, se eliminó la transformada de Fourier y se trabajó directamente con la serie de tiempo original.

\subsection{Resultados del enfoque inicial}

Se realizaron dos corridas principales variando el tamaño del codebook. En la primera corrida (codebook\_dim = 8), se obtuvo una perplexity de 1.00, lo que indica que el codebook estaba siendo subutilizado: de los 256 embeddings disponibles, el modelo utilizaba efectivamente solo 1 token. En la segunda corrida (codebook\_dim = 64), la perplexity aumentó a 5.23, pero seguía siendo baja en comparación con el tamaño del codebook.

\subsection{Motivación para el cambio de representación}

Los resultados anteriores, combinados con las limitaciones inherentes del enfoque de series de tiempo (muestreo irregular, necesidad de interpolación, dificultad para capturar la correlación entre bandas), motivaron el cambio hacia una representación basada en imágenes.

\section{Preprocesamiento: conversión de curvas de luz a imágenes 2grid}

Las curvas de luz del dataset elasticc\_1 se convierten en imágenes RGB de 256×256 píxeles mediante una representación denominada 2grid. Esta representación organiza las seis bandas fotométricas (u, g, r, i, z, Y) como subgráficos apilados verticalmente, donde el eje horizontal codifica el tiempo y el eje vertical el flujo fotométrico.

\subsection{Normalización por objeto}

Un aspecto crítico del preprocesamiento es la normalización de los ejes. Para cada supernova, el tiempo MJD y el flujo fotométrico se normalizan individualmente al rango $[0, 1]$:

\begin{equation}
\hat{t} = \frac{t - t_{\min}}{t_{\max} - t_{\min}}
\end{equation}

\begin{equation}
\hat{f} = \frac{f - Q_{0.01}}{Q_{0.99} - Q_{0.01}}
\end{equation}

donde $Q_{0.01}$ y $Q_{0.99}$ son los percentiles 1\% y 99\% del flujo de esa supernova. Esta normalización garantiza que:

\begin{itemize}
    \item La \textbf{forma} de la curva sea comparable entre objetos de distinta luminosidad
    \item Las curvas queden siempre dentro del área visible de la imagen
    \item El modelo VQ-VAE aprenda patrones morfológicos, no valores absolutos de flujo
\end{itemize}

Sin normalización, las curvas quedan fuera del área visible de la imagen porque los valores de MJD (~60.000) y flujo (~-44 a 95) no corresponden a los límites del gráfico.

\subsection{Generación en paralelo}

Dado el volumen del dataset (~895.000 supernovas), la generación se realizó en paralelo en el cluster Leftraru HPC, lanzando un job SLURM independiente por cada tipo de supernova en la partición \texttt{mi210}. Las imágenes generadas se almacenan en:

\begin{verbatim}
data/images/elasticc_1/2grid/{SNID}.png
\end{verbatim}

Con la normalización correcta, las imágenes tienen aproximadamente un 10\% de píxeles con contenido (las curvas de luz), comparado con el 4\% que se obtenía sin normalización.

\subsection{Normalización para el modelo}

Antes de ser procesadas por el VQ-VAE, las imágenes se normalizan al rango $[-1, 1]$:

\begin{equation}
\hat{x} = \frac{x - 0.5}{0.5}
\end{equation}

Esta normalización es consistente con la función de activación Tanh del decoder.

\section{Entrenamiento del VQ-VAE}

El VQ-VAE se entrena de forma no supervisada sobre el conjunto de imágenes 2grid.

\subsection{Arquitectura}

\begin{itemize}
    \item \textbf{Encoder:} tres capas Conv2d con stride 2 que reducen la resolución de 256×256 a 32×32, seguidas de dos ResBlocks. Salida: $\mathbf{z}_e \in \mathbb{R}^{256 \times 32 \times 32}$.
    
    \item \textbf{Codebook con EMA:} 512 embeddings de dimensión 256, actualizados mediante Exponential Moving Average (EMA) con decay = 0.99. Se utiliza un mecanismo de \textbf{codebook reset}: cada 10 steps, los embeddings con uso muy bajo ($\text{ema\_cluster\_size} < 1.0$) se reinicializan con vectores reales del encoder. Esto evita el colapso del codebook y mantiene una alta diversidad de embeddings activos.
    
    \item \textbf{Decoder:} espejo del encoder, con tres capas ConvTranspose2d que recuperan la resolución 32×32 → 256×256, seguidas de activación Tanh.
\end{itemize}

El modelo tiene aproximadamente 7.7 millones de parámetros entrenables (3.6M en el encoder y 4.1M en el decoder). El codebook usa \texttt{register\_buffer} para los buffers EMA, por lo que no tiene parámetros entrenables por gradientes.

\subsection{Configuración del entrenamiento}

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
EMA decay & 0.99 \\
Codebook reset threshold & 1.0 \\
Codebook reset cada & 10 steps \\
Precisión & float16 mixta \\
Gradient clip & 1.0 \\
Máximo de épocas & 100 \\
Early stopping (patience) & 10 \\
\hline
\end{tabular}
\caption{Hiperparámetros del entrenamiento del VQ-VAE.}
\label{tab:vqvae_config}
\end{table}

Se utiliza precisión mixta float16 (\texttt{precision="16-mixed"}) para reducir el uso de memoria GPU de ~58 GB a ~1 GB, lo que permite entrenar en una sola GPU AMD Instinct MI210 con batch size 32.

\subsection{Cross-validation}

Se utiliza una estrategia de cross-validation con 5 folds estratificados por tipo de supernova, generados con \texttt{StratifiedKFold} con semilla fija (\texttt{random\_state=42}). Esto garantiza que cada fold tenga la misma proporción de cada tipo de SN. Durante el desarrollo se trabaja con el fold 0, y la evaluación final utiliza los 5 folds.

Cada fold contiene:
\begin{itemize}
    \item \textbf{Training:} 718.673 imágenes (80\%)
    \item \textbf{Validación:} 179.669 imágenes (20\%)
\end{itemize}

\subsection{Métricas de monitoreo}

Durante el entrenamiento se registran las siguientes métricas en Weights \& Biases (wandb):

\begin{itemize}
    \item \textbf{train/loss} y \textbf{val/loss}: pérdida total (reconstrucción + commitment)
    \item \textbf{train/recon\_loss}: pérdida de reconstrucción MSE
    \item \textbf{train/vq\_loss}: pérdida de cuantización total
    \item \textbf{train/commitment\_loss}: pérdida de commitment
    \item \textbf{train/codebook\_loss}: pérdida del codebook (0 con EMA)
    \item \textbf{train/perplexity}: número efectivo de embeddings usados. Con codebook reset, se alcanza una perplexity de ~90, indicando que el modelo usa activamente ~90 de los 512 embeddings disponibles.
\end{itemize}

\section{Detección de anomalías}

Una vez entrenado el VQ-VAE, cada supernova queda representada como un mapa de índices discretos $\mathbf{s} \in \{1, \ldots, 512\}^{32 \times 32}$. El pipeline de detección de anomalías consta de dos pasos: entrenamiento de un modelo prior y cálculo del score de anomalía.

\subsection{Entrenamiento del prior}

Se entrena un modelo prior tipo MaskGIT sobre las secuencias de tokens del conjunto de entrenamiento. El prior aprende la distribución conjunta $p(\mathbf{s})$ sobre los 1024 tokens de cada imagen.

\subsection{Score de anomalía}

Para cada supernova del conjunto de evaluación, se calcula su score de anomalía como la log-verosimilitud negativa bajo el prior:

\begin{equation}
\text{score}(\mathbf{s}) = -\log p(\mathbf{s})
\end{equation}

\section{Generación de contrafactuales}

Dado un objeto anómalo con secuencia $\mathbf{s}$, se identifican las posiciones con mayor contribución al score, se enmascaran y se remuestrean condicionalmente:

\begin{equation}
\mathbf{s}' \sim p(\mathbf{s}_{\text{mask}} \mid \mathbf{s}_{\text{visible}})
\end{equation}

La calidad de los contrafactuales se evalúa según plausibilidad, parsimonia y consistencia temporal entre bandas.
