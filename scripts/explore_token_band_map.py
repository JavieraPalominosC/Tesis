import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

BAND_COLORS = {0:'#00FF7F',1:'#7FFF00',2:'#FF007F',3:'#FF7F00',4:'#007FFF',5:'#7F00FF'}
BAND_NAMES = {0:'u',1:'g',2:'r',3:'i',4:'z',5:'Y'}
GRID = 32
PATCH = 8

def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i+2],16) for i in (0,2,4)], dtype=np.float32)/255.0

BAND_RGB = {k: hex_to_rgb(v) for k,v in BAND_COLORS.items()}

def dominant_band_per_patch(img):
    band_map = np.full((GRID,GRID), -1, dtype=int)
    band_presence = np.zeros((GRID,GRID,6), dtype=np.float32)
    for gi in range(GRID):
        for gj in range(GRID):
            patch = img[gi*PATCH:(gi+1)*PATCH, gj*PATCH:(gj+1)*PATCH, :]
            pixels = patch.reshape(-1,3)
            non_white = pixels[pixels.mean(axis=1) < 0.9]
            if len(non_white) == 0:
                continue
            for band_id, rgb in BAND_RGB.items():
                dists = np.linalg.norm(non_white - rgb, axis=1)
                band_presence[gi,gj,band_id] = (dists < 0.3).sum()
            if band_presence[gi,gj].sum() > 0:
                band_map[gi,gj] = band_presence[gi,gj].argmax()
    return band_map, band_presence

def main():
    with open('data/images_exp/labels_subset.json') as f:
        labels = json.load(f)
    snid = None
    for s,l in labels.items():
        if l == 15:
            if os.path.exists(f'data/images_exp/overlay/{s}.png'):
                snid = s; break
    if snid is None:
        print("No SLSN found"); return
    img = np.asarray(Image.open(f'data/images_exp/overlay/{snid}.png').convert('RGB').resize((256,256)), dtype=np.float32)/255.0
    print(f"SN: {snid} (SLSN)", flush=True)

    band_map, bp = dominant_band_per_patch(img)
    total = (band_map>=0).sum()
    print(f"Tokens con contenido: {total}/{GRID*GRID}", flush=True)
    for b in range(6):
        print(f"  Banda {BAND_NAMES[b]}: {(band_map==b).sum()} tokens", flush=True)

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    axes[0].imshow(img); axes[0].set_title(f'Overlay original {snid}'); axes[0].axis('off')

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['white']+[BAND_COLORS[i] for i in range(6)])
    norm = BoundaryNorm(range(-1,7), cmap.N)
    axes[1].imshow(band_map, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('Banda dominante por token'); axes[1].axis('off')

    content_col = (band_map>=0).sum(axis=0)
    axes[2].bar(range(GRID), content_col, color='steelblue')
    axes[2].set_xlabel('Columna (tiempo)'); axes[2].set_ylabel('Tokens con contenido')
    axes[2].set_title('Perfil de contenido por tiempo')

    plt.tight_layout()
    plt.savefig('results/explore_token_band_map.png', dpi=120, bbox_inches='tight')
    print("Guardado en results/explore_token_band_map.png", flush=True)

if __name__ == '__main__':
    main()
