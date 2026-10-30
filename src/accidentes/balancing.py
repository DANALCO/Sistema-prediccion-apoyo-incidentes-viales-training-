"""
BALANCEO DE CLASES (SMOTENC)
=============================================================================
Responsabilidad unica: balancear el ENTRENAMIENTO. Jamas se aplica a
validacion ni a test: tienen que seguir pareciendose a la realidad.

SMOTENC (SMOTE para datos Nominales y Continuos) genera casos criticos
sinteticos: interpola las variables continuas entre vecinos y a las
categoricas les asigna el valor mas frecuente entre esos vecinos. Asi no
aparecen valores absurdos como 0.4 en una columna de si/no, que si produce
SMOTE sobre datos ya convertidos a one-hot.

La generacion se hace POR LOTES: una sola llamada con ~1.4 millones de
sinteticas necesitaria ~9 GB de memoria; con lotes de 300 mil el pico es de
~3 GB. Estadisticamente es lo mismo (cada lote elige sus vecinos al azar).

LIMITACION A DECLARAR EN EL INFORME: al interpolar Start_Lat y Start_Lng
entre dos accidentes lejanos, el sintetico puede quedar en un lugar donde no
hubo ningun accidente.
=============================================================================
"""

import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC


def recortar_normales(X, y, max_filas, semilla):
    """Tope opcional de filas de la clase mayoritaria (None = no recorta)."""
    y = np.asarray(y)
    if max_filas is None:
        return X, y
    clases, cuentas = np.unique(y, return_counts=True)
    mayoritaria = clases[cuentas.argmax()]
    idx_may = np.flatnonzero(y == mayoritaria)
    if len(idx_may) <= max_filas:
        return X, y
    rng = np.random.default_rng(semilla)
    conservar = np.sort(np.concatenate([
        np.flatnonzero(y != mayoritaria),
        rng.choice(idx_may, size=max_filas, replace=False),
    ]))
    return X.iloc[conservar], y[conservar]


def aplicar_smotenc(X_mixto, y, posiciones_categoricas, ratio=1.0, k_vecinos=5,
                    lote=300_000, semilla=42):
    """Devuelve (X_balanceado, y_balanceado). X_mixto: DataFrame 'mixto' del Preprocesador.

    ratio: minoritaria / mayoritaria despues de SMOTENC (1.0 = 50/50).
    """
    y = np.asarray(y)
    clases, cuentas = np.unique(y, return_counts=True)
    minoritaria, mayoritaria = clases[cuentas.argmin()], clases[cuentas.argmax()]
    n_min, n_may = int(cuentas.min()), int(cuentas.max())
    faltan = int(round(n_may * ratio)) - n_min
    if faltan <= 0:
        return X_mixto, y
    if n_min <= k_vecinos:
        raise ValueError(f"Hay solo {n_min} casos criticos en train; SMOTENC necesita > {k_vecinos}")

    X = X_mixto.to_numpy("float64")
    rng = np.random.default_rng(semilla)
    idx_min = np.flatnonzero(y == minoritaria)
    idx_relleno = rng.choice(np.flatnonzero(y == mayoritaria), size=min(200, n_may), replace=False)
    base = np.concatenate([idx_min, idx_relleno])
    X_base, y_base = X[base], y[base]

    nuevas, generadas, numero = [], 0, 0
    total_lotes = int(np.ceil(faltan / lote))
    while generadas < faltan:
        t0 = time.time()
        n_lote = min(lote, faltan - generadas)
        sm = SMOTENC(categorical_features=posiciones_categoricas, k_neighbors=k_vecinos,
                     sampling_strategy={minoritaria: n_min + n_lote},
                     random_state=semilla + numero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_res, _ = sm.fit_resample(X_base, y_base)
        nuevas.append(X_res[len(X_base):])        # lo sintetico va al final
        generadas += n_lote
        numero += 1
        print(f"    lote {numero}/{total_lotes}: +{n_lote:,} criticos sinteticos "
              f"({time.time() - t0:.0f} s)")

    X_final = np.vstack([X] + nuevas)
    y_final = np.concatenate([y, np.full(faltan, minoritaria, dtype=y.dtype)])

    orden = np.random.default_rng(semilla).permutation(len(y_final))   # mezclar
    X_final, y_final = X_final[orden], y_final[orden]

    salida = pd.DataFrame(X_final, columns=X_mixto.columns)
    for pos in posiciones_categoricas:
        col = X_mixto.columns[pos]
        salida[col] = np.rint(salida[col].to_numpy())     # valores enteros validos
    return salida, y_final


def composicion(etapa, y):
    """Fila para la tabla de composicion de clases (antes/despues del balanceo)."""
    y = np.asarray(y)
    criticos = int(y.sum())
    return {
        "Etapa": etapa,
        "Filas": len(y),
        "Normales": len(y) - criticos,
        "Criticos": criticos,
        "Pct_critico": round(criticos / len(y) * 100, 3),
    }
