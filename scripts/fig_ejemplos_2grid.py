"""Panel de ejemplo de la representacion 2grid con codificacion mad300.

La tesis describe las imagenes 2grid en tres capitulos y nunca muestra una.
Toma una imagen ya renderizada por el pipeline (no re-renderiza nada) por cada
tipo elegido y las pone lado a lado con los rotulos de banda.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

RAIZ = 'data/enc_cmp/mad300'
SALIDA = 'results/figs_profe/ejemplo_2grid.png'

# El label_id es el indice del arreglo de parquets de run_mad300_full.sh
TIPOS = {0: 'SNIa-SALT2', 4: 'SNII-NMF', 14: 'SLSN-I+host', 16: 'PISN'}
BANDAS = ['Y', 'z', 'i', 'r', 'g', 'u']   # j=5 arriba, j=0 abajo

labels = json.load(open(f'{RAIZ}/labels.json'))

elegidos = {}
for snid, lab in labels.items():
    if lab in TIPOS and lab not in elegidos:
        elegidos[lab] = snid
    if len(elegidos) == len(TIPOS):
        break

fig, axes = plt.subplots(1, len(TIPOS), figsize=(11, 3.1))
for ax, (lab, nombre) in zip(axes, TIPOS.items()):
    snid = elegidos[lab]
    ax.imshow(imread(f'{RAIZ}/images/{snid}.png'))
    ax.set_title(f'{nombre}\nSNID {snid}', fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

# Rotulos de banda solo en el panel de la izquierda.
alto = imread(f'{RAIZ}/images/{elegidos[0]}.png').shape[0]
paso = alto / len(BANDAS)
axes[0].set_yticks([(k + 0.5) * paso for k in range(len(BANDAS))])
axes[0].set_yticklabels(BANDAS, fontsize=8)
axes[0].set_ylabel('banda', fontsize=9)

axes[0].set_xlabel('tiempo normalizado', fontsize=8.5)
fig.suptitle('Representacion 2grid con codificacion mad300', fontsize=10.5)
fig.tight_layout()
fig.savefig(SALIDA, dpi=160, bbox_inches='tight')
print('escrita', SALIDA)
print('objetos:', {TIPOS[k]: v for k, v in elegidos.items()})
