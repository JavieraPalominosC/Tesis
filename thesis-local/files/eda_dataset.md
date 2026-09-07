\section{Análisis exploratorio del dataset}

\subsection{Distribución de objetos por tipo}

El dataset utilizado en este trabajo contiene un total de 1.389.147 supernovas distribuidas en 17 tipos. La Tabla~\ref{tab:distribucion_tipos} muestra la distribución de objetos, el número total de observaciones y las estadísticas de observaciones por objeto para cada tipo.

\begin{table}[H]
\centering
\small
\begin{tabular}{|l|r|r|r|r|}
\hline
\textbf{Tipo} & \textbf{Objetos} & \textbf{Obs. total} & \textbf{Obs./obj. (media)} & \textbf{Obs./obj. (mediana)} \\
\hline
SNIa-SALT2         & 211.892 & 23.832.381 & 112.5 & 124.0 \\
SNIa-91bg          &  53.414 &  6.102.090 & 114.2 & 125.0 \\
SNIax              &  53.727 &  6.344.784 & 118.1 & 126.0 \\
SNII-Templates     & 103.536 & 11.298.138 & 109.1 & 122.0 \\
SNII-NMF           & 102.251 & 12.288.097 & 120.2 & 127.0 \\
SNII+HostXT\_V19   & 103.189 & 11.214.651 & 108.7 & 121.0 \\
SNIb-Templates     &  40.065 &  4.530.177 & 113.1 & 125.0 \\
SNIb+HostXT\_V19   & 102.883 & 11.287.051 & 109.7 & 122.0 \\
SNIc-Templates     &  20.617 &  2.383.542 & 115.6 & 125.0 \\
SNIc+HostXT\_V19   & 103.885 & 11.425.289 & 110.0 & 123.0 \\
SNIcBL+HostXT\_V19 &  42.878 &  4.774.667 & 111.4 & 123.0 \\
SNIIb+HostXT\_V19  & 103.683 & 11.294.039 & 108.9 & 122.0 \\
SNIIn-MOSFIT       &  31.162 &  3.905.954 & 125.3 & 130.0 \\
SNIIn+HostXT\_V19  & 105.281 & 12.285.982 & 116.7 & 125.0 \\
SLSN-I+host        & 100.977 & 15.814.118 & 156.6 & 163.0 \\
SLSN-I\_no\_host   &   4.261 &    761.439 & 178.7 & 193.0 \\
PISN               & 105.446 & 14.563.011 & 138.1 & 139.0 \\
\hline
\textbf{Total} & \textbf{1.389.147} & \textbf{163.905.410} & 120.4 & 126.0 \\
\hline
\end{tabular}
\caption{Distribución de objetos por tipo de supernova en el dataset elasticc\_1.}
\label{tab:distribucion_tipos}
\end{table}

Se observa que el dataset presenta un desbalance importante entre tipos: el tipo más abundante (SNIa-SALT2, con 211.892 objetos) tiene 50 veces más objetos que el tipo más escaso (SLSN-I\_no\_host, con 4.261 objetos). Este desbalance es relevante para la detección de anomalías, ya que tipos poco frecuentes como SLSN-I\_no\_host podrían ser detectados como outliers simplemente por su rareza en el dataset.

\subsection{Características temporales}

Todos los tipos de supernovas comparten el mismo rango temporal de 1.104,3 días (aproximadamente 3 años), lo que es consistente con la duración esperada de una campaña de observación del Observatorio Vera C. Rubin. El número de observaciones por objeto varía entre tipos, con una mediana global de 126 observaciones por supernova.

Destacan dos grupos con comportamientos distintos:

\begin{itemize}
    \item \textbf{Supernovas estándar} (SNIa, SNII, SNIb, SNIc): entre 108 y 127 observaciones por objeto en promedio, lo que refleja el patrón típico de observación de eventos transitorios.
    
    \item \textbf{Supernovas superluminosas} (SLSN-I+host, SLSN-I\_no\_host) y \textbf{PISN}: entre 138 y 178 observaciones por objeto, lo que refleja su mayor duración e intrínsicamente mayor brillo, que permite seguirlas durante más tiempo.
\end{itemize}

\subsection{Cobertura en bandas}

Todos los objetos del dataset tienen observaciones en las seis bandas fotométricas (u, g, r, i, z, Y), lo que garantiza que todas las imágenes 2grid generadas tienen información en todos los subgráficos. Sin embargo, la densidad de observaciones por banda varía según el tipo de objeto y la estrategia de observación simulada.

\subsection{Objetos potencialmente anómalos}

Desde el punto de vista de la detección de anomalías sin supervisión, el dataset presenta varios candidatos naturales a outliers:

\begin{itemize}
    \item \textbf{SLSN-I\_no\_host} (4.261 objetos): el tipo más escaso, con curvas de luz significativamente más largas (mediana de 193 observaciones) que el resto. Su rareza y morfología distintiva los hace candidatos naturales a ser detectados como outliers.
    
    \item \textbf{PISN} (105.446 objetos): supernovas de inestabilidad de par, con curvas de luz más largas que las supernovas estándar (mediana de 139 observaciones) y una morfología particular asociada a estrellas muy masivas.
    
    \item \textbf{SLSN-I+host} (100.977 objetos): supernovas superluminosas con galaxia huésped, con curvas de luz muy extendidas (mediana de 163 observaciones).
\end{itemize}

\subsection{Imágenes 2grid generadas}

A partir del dataset descrito, se generaron 895.354 imágenes 2grid de 256×256 píxeles, correspondientes al subconjunto de supernovas procesadas exitosamente. La diferencia respecto al total de 1.389.147 objetos se debe a que la generación de imágenes se realizó con un límite de tiempo de 12 horas por job SLURM, y algunos archivos no alcanzaron a procesarse completamente. Para el entrenamiento del VQ-VAE se utilizaron 899.440 imágenes (incluyendo algunas generadas en corridas previas), divididas en 809.496 para entrenamiento y 89.944 para validación.
