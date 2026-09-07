\section{Exploración metodológica previa}

Antes de adoptar el enfoque basado en imágenes descrito en este trabajo, se realizó una exploración inicial utilizando series de tiempo directamente como input del VQ-VAE, siguiendo el esquema del modelo TimeVQVAE \cite{lee2023timevqvae}. Esta sección describe ese proceso y los resultados que motivaron el cambio de representación.

\subsection{Enfoque inicial: series de tiempo con TimeVQVAE}

El punto de partida fue el código de TimeVQVAE, que utiliza la transformada de Fourier para convertir la serie de tiempo al espacio tiempo-frecuencia antes de pasarla al VQ-VAE. Sin embargo, para datos de supernovas esta representación no es adecuada, ya que las curvas de luz tienen muestreo irregular y características morfológicas (el peak de brillo, el tiempo de caída) que no se benefician de una descomposición en frecuencias. Por esta razón, se eliminó la transformada de Fourier y se trabajó directamente con la serie de tiempo original.

\subsection{Dataset sintético}

Para esta exploración inicial se utilizó un dataset sintético de supernovas generado con la fórmula paramétrica que caracteriza la curva de luz de una supernova:

\begin{equation}
f(t) = A \cdot \frac{e^{-(t - t_0)/t_{\text{fall}}}}{1 + e^{-(t - t_0)/t_{\text{rise}}}} + B
\end{equation}

El dataset consistía en 10.000 supernovas, cada una con 5.000 puntos temporales, divididas en 8.000 para entrenamiento y 2.000 para test. El preprocesamiento de cada curva incluyó la creación de una grilla temporal uniforme, interpolación, normalización y conversión a tensor de PyTorch.

\subsection{Arquitectura del encoder-decoder}

Se probaron dos arquitecturas para el encoder-decoder del VQ-VAE sobre series de tiempo. La primera, basada en capas fully connected, producía reconstrucciones de baja calidad (una línea recta), lo que motivó el cambio a una arquitectura convolucional:

\begin{equation}
\text{Conv1D} \rightarrow \text{AvgPool} \rightarrow \text{Linear} \rightarrow \mathbf{z}_e \rightarrow \text{VQ} \rightarrow \text{decoder}
\end{equation}

Esta segunda arquitectura mejoró notablemente la calidad de las reconstrucciones.

\subsection{Resultados del enfoque inicial}

Se realizaron dos corridas principales variando el tamaño del codebook. Los resultados se resumen en las tablas siguientes.

\textbf{Corrida 1: codebook\_dim = 8}

\begin{table}[H]
\centering
\begin{tabular}{|l|c|}
\hline
\textbf{Métrica} & \textbf{Valor} \\
\hline
Epoch final & 19 \\
Learning rate (AdamW) & 0.00027 \\
\hline
Pérdida total (train) & 0.25576 \\
Pérdida de reconstrucción (train) & 0.25012 \\
Pérdida de cuantización (train) & 0.00563 \\
Perplexity (train) & 1.00 \\
\hline
Pérdida total (val) & 0.19547 \\
Pérdida de reconstrucción (val) & 0.19547 \\
Pérdida de cuantización (val) & 0.00000 \\
Perplexity (val) & 1.00 \\
\hline
\end{tabular}
\caption{Resultados del Stage 1 con codebook\_dim = 8.}
\label{tab:stage1_v1}
\end{table}

Una perplexity de 1.00 indica que el codebook estaba siendo subutilizado: de los 256 embeddings disponibles, el modelo utilizaba efectivamente solo 1 token para representar todas las curvas de luz. Esto evidencia que el codebook era demasiado pequeño en relación con la dimensión del espacio latente (64).

\textbf{Corrida 2: codebook\_dim = 64}

Al aumentar la dimensión del codebook a 64 (igual a la dimensión del espacio latente), la perplexity aumentó y la calidad de las reconstrucciones mejoró:

\begin{table}[H]
\centering
\begin{tabular}{|l|c|}
\hline
\textbf{Métrica} & \textbf{Valor} \\
\hline
Epoch final & 19 \\
Learning rate (AdamW) & 0.00027 \\
\hline
Pérdida total (train) & 0.0412 \\
Pérdida de reconstrucción (train) & 0.02683 \\
Pérdida de cuantización (train) & 0.01437 \\
Perplexity (train) & 5.23 \\
\hline
Pérdida total (val) & 0.02773 \\
Pérdida de reconstrucción (val) & 0.02773 \\
Pérdida de cuantización (val) & 0.00000 \\
Perplexity (val) & 5.33 \\
\hline
\end{tabular}
\caption{Resultados del Stage 1 con codebook\_dim = 64.}
\label{tab:stage1_v2}
\end{table}

Aunque la perplexity mejoró de 1.00 a 5.23, sigue siendo baja en comparación con el tamaño del codebook (256 embeddings), lo que indica que el modelo no estaba aprovechando la capacidad expresiva del espacio latente discreto.

\subsection{Motivación para el cambio de representación}

Los resultados anteriores, combinados con las limitaciones inherentes del enfoque de series de tiempo (muestreo irregular, necesidad de interpolación, dificultad para capturar la correlación entre bandas), motivaron el cambio hacia una representación basada en imágenes. Al convertir las curvas de luz multibanda en imágenes 2grid, se logra:

\begin{itemize}
    \item Representar de forma natural la correlación temporal entre las seis bandas fotométricas como patrones espaciales en la imagen.
    \item Eliminar la necesidad de interpolación y manejo de muestreo irregular.
    \item Usar el dataset real elasticc\_1 en lugar de datos sintéticos.
    \item Aprovechar arquitecturas convolucionales estándar diseñadas para imágenes, que son más estables y están mejor estudiadas que las arquitecturas para series de tiempo.
\end{itemize}

Con este nuevo enfoque, el VQ-VAE opera sobre imágenes de 256×256 píxeles con un codebook de 512 embeddings de dimensión 256, lo que proporciona una capacidad expresiva significativamente mayor que el enfoque anterior.
