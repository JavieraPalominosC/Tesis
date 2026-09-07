"""Codificaciones de flujo a altura de pixel.

Unica fuente de verdad: la usan tanto el renderizador del entrenamiento como el
de la evaluacion sintetica, para que no se puedan desincronizar.

  actual   u = clip( (f - p01) / (p99 - p01), 0, 1 )
           El p99 lo fija el punto mas brillante, asi que un outlier define su
           propia escala: al meter el pico, la curva entera se re-escala y el
           pico deja de subir. Medido, el p99 se multiplica por 60 entre df=10
           y df=1000.

  mad<U>   u = ( asinh( (f - mediana) / MAD ) / asinh(U) + 1 ) / 2
           Centro y escala robustos del propio objeto. El MAD necesita mas de
           50% de contaminacion para moverse, asi que el pico no arrastra la
           escala (medido: se queda en 1.4x para cualquier df). Sin recorte, asi
           que mas flujo siempre es mas altura. U fija cuantos MAD entran en el
           carril y es el compromiso entre resolucion para la forma normal y
           techo para el outlier.

En los dos casos los estadisticos se calculan sobre la curva que se esta
dibujando, incluida la perturbada, que es lo que haria el pipeline en produccion
al no tener una version limpia de referencia.
"""
import numpy as np


def escala_de(enc):
    """None para la codificacion actual, o el U de la variante mad."""
    return float(enc[3:]) if enc.startswith('mad') else None


def codificar(flux_vals, obj_flux, enc):
    """Mapea flujo a [0, 1] dentro del carril de la banda."""
    flux_vals = np.asarray(flux_vals, dtype='float64')
    obj_flux = np.asarray(obj_flux, dtype='float64')

    if enc == 'actual':
        p01, p99 = np.quantile(obj_flux, 0.01), np.quantile(obj_flux, 0.99)
        rng = (p99 - p01) if p99 > p01 else 1.0
        return np.clip((flux_vals - p01) / rng, 0.0, 1.0)

    u_max = escala_de(enc)
    if u_max is None:
        raise ValueError(f'codificacion desconocida: {enc}')

    med = np.median(obj_flux)
    s = np.median(np.abs(obj_flux - med))
    s = s if (np.isfinite(s) and s > 0) else 1.0
    u = np.arcsinh((flux_vals - med) / s) / np.arcsinh(u_max)
    return (np.clip(u, -1.0, 1.0) + 1.0) / 2.0
