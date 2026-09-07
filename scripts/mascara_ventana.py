"""Enmascarado de una ventana 2D contigua para MaskGIT.

El MaskGIT original enmascara tokens sueltos al azar, asi que alrededor de cada
hueco casi siempre quedan vecinos visibles y el modelo puede interpolar. Con una
ventana contigua tiene que reconstruir una region entera sin contexto local, que
se parece mas a lo que pasa cuando una anomalia ocupa una zona compacta de la
curva.

Ojo al comparar: predecir un bloque contiguo es una tarea intrinsecamente mas
dificil, asi que la val/loss va a ser mas alta que con mascara aleatoria. Eso no
significa peor modelo; las dos estrategias solo se pueden comparar en la tarea de
deteccion, no en la loss.

Se usa desde train_maskgit_window.py y train_maskgit_enc.py, para que las dos
compartan exactamente la misma logica.
"""
import math

import torch


def enmascarar_ventana(tokens, mask_token):
    """Enmascara un rectangulo contiguo por muestra.

    Conserva el mismo cronograma coseno de cantidad de tokens que usa el MaskGIT
    original: lo unico que cambia es que los tokens elegidos son contiguos en la
    grilla 2D en vez de estar dispersos. Si la cantidad objetivo no forma un
    rectangulo exacto se recorta o se completa al azar para respetarla.

    Devuelve (tokens_enmascarados, mascara) o None si la secuencia no es cuadrada.
    """
    B, T = tokens.shape
    H = W = int(round(math.sqrt(T)))
    if H * W != T:
        return None

    dev = tokens.device
    r = torch.rand(B, device=dev)
    r = (1 - torch.cos(r * math.pi)) / 2
    n_mask = (r * T).long().clamp(min=1)

    masked = tokens.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    for i in range(B):
        n = int(n_mask[i].item())
        w = int(torch.randint(1, W + 1, (1,), device=dev).item())
        h = min(H, max(1, (n + w - 1) // w))
        w = min(W, max(1, (n + h - 1) // h))
        r0 = 0 if h == H else int(torch.randint(0, H - h + 1, (1,), device=dev).item())
        c0 = 0 if w == W else int(torch.randint(0, W - w + 1, (1,), device=dev).item())

        grid = torch.zeros(H, W, dtype=torch.bool, device=dev)
        grid[r0:r0 + h, c0:c0 + w] = True
        flat = grid.reshape(-1)
        idx = flat.nonzero(as_tuple=False).view(-1)
        if idx.numel() > n:
            keep = idx[torch.randperm(idx.numel(), device=dev)[:n]]
            flat[:] = False
            flat[keep] = True
        elif idx.numel() < n:
            falta = n - int(idx.numel())
            hueco = (~flat).nonzero(as_tuple=False).view(-1)
            flat[hueco[torch.randperm(hueco.numel(), device=dev)[:falta]]] = True
        mask[i] = flat
        masked[i, mask[i]] = mask_token
    return masked, mask
