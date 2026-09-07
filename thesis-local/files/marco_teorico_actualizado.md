\section{Representación de curvas de luz como imágenes}

La representación de series de tiempo como imágenes permite aprovechar la capacidad de las redes convolucionales para detectar patrones espaciales. En este trabajo, cada curva de luz multibanda se convierte en una imagen 2grid de 256×256 píxeles RGB, donde las seis bandas fotométricas (u, g, r, i, z, Y) se disponen verticalmente como subgráficos apilados.

Para que las curvas sean visibles y comparables entre objetos de distinta luminosidad, el tiempo MJD y el flujo fotométrico de cada supernova se normalizan individualmente al rango $[0, 1]$ antes de graficar. Sin esta normalización, las curvas quedan fuera del área visible de la imagen porque los valores reales (MJD ~60.000, flujo ~ $-$44 a 95) no corresponden a los límites del gráfico.

\section{Autoencoder con cuantización vectorial (VQ-VAE)}

\subsection{Descripción general}

El autoencoder con cuantización vectorial (VQ-VAE, \textit{Vector Quantized Variational Autoencoder}) fue propuesto por van den Oord et al. (2017) como un modelo generativo que aprende representaciones discretas del espacio latente. El VQ-VAE es un tipo de autoencoder que aprende un espacio latente discreto. El proceso funciona en tres etapas: primero, el encoder comprime la imagen de entrada a un conjunto de vectores continuos. Luego, cada uno de estos vectores se cuantiza buscando en el codebook (un conjunto de vectores aprendidos durante el entrenamiento) cuál es el más parecido, y se reemplaza por ese. Finalmente, el decoder recibe estos vectores cuantizados y reconstruye la imagen original.

\subsection{Encoder}

El encoder $E_\phi$ transforma una imagen $\mathbf{x} \in \mathbb{R}^{3 \times 256 \times 256}$ en un mapa de features continuo $\mathbf{z}_e \in \mathbb{R}^{256 \times 32 \times 32}$ mediante tres capas convolucionales con stride 2 y dos bloques residuales:

\begin{align}
256 \times 256 &\xrightarrow{\text{Conv}(3 \to 64,\, s=2)} 128 \times 128 \\
128 \times 128 &\xrightarrow{\text{Conv}(64 \to 128,\, s=2)} 64 \times 64 \\
64 \times 64 &\xrightarrow{\text{Conv}(128 \to 256,\, s=2)} 32 \times 32 \\
32 \times 32 &\xrightarrow{\text{ResBlock} \times 2} 32 \times 32
\end{align}

El resultado son 1024 vectores de dimensión 256, cada uno representando un parche de 8×8 píxeles de la imagen original.

\subsection{Cuantización vectorial con EMA}

El codebook es un conjunto de $K = 512$ vectores aprendidos $\{\mathbf{e}_k\}_{k=1}^{K} \subset \mathbb{R}^{256}$. Para cada vector continuo $\mathbf{z}_e(i,j)$, se selecciona el embedding más cercano:

\begin{equation}
k^* = \arg\min_k \|\mathbf{z}_e(i,j) - \mathbf{e}_k\|_2^2
\end{equation}

En vez de actualizar el codebook por gradientes, se utiliza \textbf{Exponential Moving Average (EMA)}, que actualiza los embeddings directamente a partir de los vectores del encoder que caen en cada cluster:

\begin{align}
N_k &\leftarrow \gamma N_k + (1-\gamma) n_k \\
\mathbf{m}_k &\leftarrow \gamma \mathbf{m}_k + (1-\gamma) \sum_{i: k^*_i = k} \mathbf{z}_{e,i} \\
\mathbf{e}_k &\leftarrow \frac{\mathbf{m}_k}{N_k}
\end{align}

donde $\gamma = 0.99$ es el factor de decay y $n_k$ es el número de vectores asignados al embedding $k$ en el batch actual. Las actualizaciones EMA se realizan sin gradientes (\texttt{torch.no\_grad()}) y usando operaciones in-place para evitar acumulación de memoria.

\textbf{Straight-through estimator.} Como el argmin no es diferenciable, en el backward pass el gradiente se pasa directamente de $\mathbf{z}_q$ a $\mathbf{z}_e$ sin modificación.

\subsection{Codebook reset}

Un problema común en VQ-VAE es el \textbf{colapso del codebook}: el modelo converge a usar solo unos pocos embeddings de los $K$ disponibles, lo que reduce la capacidad expresiva del espacio latente. Para evitarlo, se implementa un mecanismo de \textbf{codebook reset}: cada 10 steps de entrenamiento, los embeddings con uso muy bajo ($N_k < 1.0$) se reinicializan con vectores aleatorios del batch actual:

\begin{equation}
\mathbf{e}_k \leftarrow \mathbf{z}_{e,\text{random}} \quad \text{si } N_k < \theta_{\text{reset}}
\end{equation}

Este mecanismo permite que la perplexity del codebook alcance valores de ~90 (de 512 posibles), comparado con valores de 3-4 sin codebook reset.

\subsection{Decoder}

El decoder $G_\theta$ reconstruye la imagen a partir del mapa cuantizado $\mathbf{z}_q$ mediante capas de convolución transpuesta con stride 2 y activación Tanh final.

\subsection{Función de pérdida}

\begin{equation}
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{commitment}}
\end{equation}

Con EMA, el término de codebook loss desaparece (los embeddings se actualizan directamente), quedando solo la pérdida de reconstrucción y la pérdida de commitment:

\begin{itemize}
    \item $\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$: diferencia en el espacio de píxeles
    \item $\mathcal{L}_{\text{commitment}} = \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$: acerca los vectores del encoder hacia los embeddings, con $\beta = 0.25$
\end{itemize}

\subsection{Perplexity del codebook}

La perplexity mide cuántos embeddings se están usando efectivamente:

\begin{equation}
\text{perplexity} = \exp\left(-\sum_k \bar{p}_k \log \bar{p}_k\right)
\end{equation}

donde $\bar{p}_k$ es la fracción de vectores asignados al embedding $k$. Si perplexity = 1, solo un embedding se usa (colapso). Si perplexity = $K$, todos los embeddings se usan uniformemente. Con codebook reset, se alcanza una perplexity de ~90 durante el entrenamiento.

\subsection{Precisión mixta}

Para reducir el uso de memoria GPU, se utiliza precisión mixta float16 (\texttt{precision="16-mixed"}). Las operaciones forward y backward se realizan en float16, reduciendo el uso de VRAM de ~58 GB a ~1 GB. Las actualizaciones EMA del codebook se mantienen en float32 para mayor estabilidad numérica.

\section{Modelo prior sobre el espacio latente discreto}

Una vez entrenado el VQ-VAE, cada imagen queda representada como una secuencia de 1024 tokens enteros $\mathbf{s} \in \{1, \ldots, 512\}^{32 \times 32}$. Se entrena un prior tipo MaskGIT que aprende la distribución $p(\mathbf{s})$, lo que permite asignar probabilidades a secuencias y muestrear nuevas secuencias plausibles.

\subsection{Score de anomalía}

\begin{equation}
\text{score}(\mathbf{s}) = -\log p(\mathbf{s})
\end{equation}

Una curva de luz típica produce tokens frecuentes → score bajo. Un outlier produce tokens inusuales → score alto.

\subsection{Generación de contrafactuales}

Los tokens más improbables se enmascaran y remuestrean condicionalmente:

\begin{equation}
\mathbf{s}' \sim p(\mathbf{s}_{\text{mask}} \mid \mathbf{s}_{\text{visible}})
\end{equation}

Al decodificar $\mathbf{s}'$ se obtiene la curva de luz contrafactual, que al compararse con el original revela qué aspectos hacen al objeto anómalo.
