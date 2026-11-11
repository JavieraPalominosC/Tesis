# Superficies GP como input del VQ-VAE

Además de las imágenes 2grid, ahora se está probando como input del VQ-VAE la superficie que entrega un Gaussian Process (GP) ajustado a la curva de luz, con el objetivo de comparar ambas representaciones bajo el mismo pipeline (VQ-VAE, tokens y prior MaskGIT) y el mismo criterio de detección de anomalías.

## Datos

Se usa el sample simulado de LSST preparado por F. Förster para el paper en preparación *ALeRCE joint class-distance estimator using fast Gaussian Process estimation and causal networks* (no publicado), que viene en dos archivos: `lsst_paper_sample_v2` (tabla de objetos y fotometría) y `lsst_paper_sample_v2_gp` (los ajustes GP). El texto de los README de ambos está en [`lsst_paper_sample_v2.md`](lsst_paper_sample_v2.md).

El sample tiene 26 clases balanceadas (15 transientes y 11 no transientes), observadas con cadencias DDF y WFD, con 1.000 objetos por clase. Se divide en un split `cv`, con folds 0 a 4 y 800 objetos por clase, y un split `test` de 200 objetos por clase que no se usa en ningún experimento, para dejarlo intacto como lo pide el paper.

## La superficie GP

Para cada objeto se ajusta un GP 2D sobre (tiempo, ln λ), con un kernel Matérn-3/2 en el tiempo multiplicado por un kernel exponencial cuadrático en ln λ, donde cada banda se ubica en su longitud de onda efectiva, y los hiperparámetros se ajustan por objeto. La media se evalúa en una grilla de 200 valores de ln λ (λ entre 3200 y 10200 Å) por 300 valores de tiempo (de −100 a +300 días desde el trigger), y se guarda dividida por un factor de escala por objeto (`scale_mean`).

Como input se toma esa media tal cual viene (as-is), recortada a [−1, 1] para que calce con la salida Tanh del decoder, sin la codificación mad300 que se usa en las imágenes 2grid. La superficie se redimensiona bilinealmente de 200×300 a 256×256 y queda como una imagen de un canal.

La diferencia conceptual con 2grid es que el GP modela conjuntamente todas las bandas y el tiempo, de modo que la superficie interpola entre bandas y entre épocas, mientras que en 2grid se grafican los puntos observados de cada banda por separado.

## Modelo

La arquitectura del VQ-VAE y del prior MaskGIT no cambia. El encoder tiene 3 convoluciones con stride 2, que llevan la imagen de 256×256 a una grilla de 32×32 tokens, con codebook de K = 512 vectores de dimensión D = 256, y un clasificador sobre $\mathbf{z}_q$ con `cls_weight` = 0,1. El MaskGIT tiene 4 capas, 4 cabezas de atención, dimensión oculta 256 y secuencias de largo 1024. Lo único que cambia es el número de canales de entrada (1) y el número de clases de cada experimento.

## Experimento 1: piloto de transientes

Se arma un subset balanceado de las 15 clases transientes con 16 objetos por clase, survey y fold, lo que da 2.400 objetos (1.920 de entrenamiento y 480 de validación, con el fold 0 como validación).

El VQ-VAE, evaluado sobre la validación completa (480 objetos), alcanza una exactitud de 41,3% frente a un 6,7% por azar, con 46,7% en DDF y 35,8% en WFD. El codebook tiene perplejidad 284 y usa 487 de los 512 tokens, y el MSE de reconstrucción es 0,0037.

| Mejores clases | Exactitud | Peores clases | Exactitud |
|---|---|---|---|
| TDE-MOSFIT | 94% | ILOT | 12% |
| KN-BULLA19 | 81% | SNIax | 19% |
| SNIa-91bg | 72% | KN-K17 | 22% |
| PISN-MOSFIT | 62% | PISN-STELLA-HYDROGENIC | 22% |
| SNIa-SALT2 | 53% | | |

Las métricas que muestra la barra de progreso de Lightning corresponden al último batch (se registran con `on_step` y la validación no se baraja), no al split entero, por lo que se agregó `scripts/eval_vqvae_gp.py`, que recorre el split completo y agrega las métricas. El MaskGIT del piloto está en curso.

## Experimento 2: transientes completos con clases excluidas

Se sigue el mismo criterio que el experimento heldout de ELAsTiCC-1 (`scripts/make_folds_heldout.py`), excluyendo del entrenamiento y de la validación SNIa-91bg-elastic, SNIax-elastic, SLSN-I, PISN-MOSFIT, PISN-STELLA-HECORE y PISN-STELLA-HYDROGENIC, que equivalen a SNIa-91bg, SNIax, SLSN-I+host/no_host y PISN de ese experimento. SNIIb+HostXT_V19 no tiene par en este sample.

Las 9 clases vistas son CaRT, ILOT-MOSFIT, KN-BULLA19, KN-K17, SNII-NMF, SNIIn-MOSFIT, SNIa-SALT2, SNIbc-MOSFIT y TDE-MOSFIT; CaRT, ILOT, las dos KN y TDE no existían en ELAsTiCC-1. Quedan 7.200 objetos vistos (5.760 de entrenamiento y 1.440 de validación, 160 por clase en validación) y 4.800 excluidos, que se tokenizan con el VQ-VAE entrenado para calcularles el score de anomalía con el MaskGIT. El balance del paper es por clase (800 en `cv`), así que los conteos por survey y por fold varían en unos pocos objetos. Este experimento está en ejecución.

## Pendiente

Comparar contra 2grid sobre los mismos objetos y con el mismo criterio de clases excluidas.

## Archivos

| Archivo | Qué hace |
|---|---|
| `src/data/gp_dataset.py` | Dataset y DataModule de superficies GP; codifica la media (as-is o mad300) y la redimensiona a 256×256. |
| `src/models/vqvae/model_gp.py` | `VQVAEGP`: el mismo VQ-VAE, con el log de reconstrucciones adaptado a 1–2 canales. |
| `scripts/prepare_gp_pilot.py` | Lee los dos tar, cachea cada superficie como `.npy` (C, 256, 256) y escribe folds, etiquetas y clases del split `cv`. |
| `scripts/make_gp_subset.py` | Arma un subset balanceado por (clase, survey, fold) sobre el cache, opcionalmente solo transientes o no transientes. |
| `scripts/make_gp_heldout.py` | Folds de transientes completos sin las seis clases excluidas, más la lista de excluidas para evaluarlas como anomalías. |
| `scripts/train_vqvae_gp.py` | Entrena el VQ-VAE GP a partir de un YAML. |
| `scripts/generate_tokens_gp.py` | Tokeniza las superficies con el mejor checkpoint del VQ-VAE. |
| `scripts/eval_vqvae_gp.py` | Evalúa el VQ-VAE sobre el split completo: exactitud por clase y survey, perplejidad, uso del codebook, MSE y matriz de confusión. |
| `scripts/train_maskgit_gp.py` | Entrena el prior MaskGIT sobre los tokens GP (máscara aleatoria o por ventana). |
| `configs/vqvae_gp_pilot.yaml` | Config del piloto con las 26 clases. |
| `configs/vqvae_gp_pilot_s80.yaml` | Config del subset de 26 clases con 16 objetos por clase, survey y fold. |
| `configs/vqvae_gp_pilot_tr.yaml` | Config del piloto de transientes (experimento 1). |
| `configs/vqvae_gp_tr_ho.yaml` | Config de transientes con clases excluidas (experimento 2). |
| `run_gp_pilot.sh` | Cadena completa del piloto de 26 clases: preparación, VQ-VAE, tokens y MaskGIT. |
| `run_gp_pilot_s80.sh` | Cadena del subset de 26 clases. |
| `run_gp_pilot_tr.sh` | Cadena del piloto de transientes. |
| `run_gp_tr_ho.sh` | Cadena del experimento con clases excluidas, incluida la evaluación del VQ-VAE. |
