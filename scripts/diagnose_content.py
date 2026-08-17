import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image

GRID = 32
PATCH = 8

paths_file = 'data/tokens/paths_waterfall_full.txt'
with open(paths_file) as f:
    paths = [l.strip() for l in f][:20]

fracs = []
for p in paths:
    img = np.asarray(Image.open(p).convert('RGB').resize((256,256)), dtype=np.float32)/255.0
    cm = np.zeros((GRID, GRID), dtype=bool)
    for gi in range(GRID):
        for gj in range(GRID):
            patch = img[gi*PATCH:(gi+1)*PATCH, gj*PATCH:(gj+1)*PATCH, :]
            cm[gi, gj] = (patch.reshape(-1,3).mean(axis=1) < 0.9).sum() > 0
    fracs.append(cm.sum() / (GRID*GRID))

fracs = np.array(fracs)
print(f"Fraccion de tokens CON contenido: media={fracs.mean():.3f}  min={fracs.min():.3f}  max={fracs.max():.3f}")
print(f"  -> en promedio {fracs.mean()*1024:.0f} de 1024 tokens tienen curva")
print(f"  -> {(1-fracs.mean())*100:.1f}% de los tokens son fondo blanco")
