"""
PARTICION TEMPORAL
=============================================================================
Responsabilidad unica: dividir los datos por FECHA (nunca al azar). Se
entrena con lo antiguo y se evalua con lo reciente, como en la realidad:
un modelo desplegado siempre predice el futuro.

    train       lo mas antiguo; aqui se aprenden TODOS los parametros
    val_ajuste  primera mitad de la validacion: calibracion y combinador
    val_umbral  segunda mitad de la validacion: umbral de decision
    test        lo mas reciente; se usa UNA sola vez, al final

Las particiones devuelven POSICIONES de fila (enteros), no copias. Asi los
resultados del modelo se pueden alinear despues con la tabla limpia (por
ejemplo, para analizar falsos positivos por estado u hora).
=============================================================================
"""

import numpy as np
import pandas as pd

NOMBRES = ("train", "val_ajuste", "val_umbral", "test")


def dividir_temporal(fechas, corte_train, corte_val_ajuste, corte_test):
    fechas = pd.to_datetime(fechas).reset_index(drop=True)
    masc = {
        "train": fechas < corte_train,
        "val_ajuste": (fechas >= corte_train) & (fechas < corte_val_ajuste),
        "val_umbral": (fechas >= corte_val_ajuste) & (fechas < corte_test),
        "test": fechas >= corte_test,
    }
    posiciones = {nombre: np.flatnonzero(m.to_numpy()) for nombre, m in masc.items()}

    # Verificaciones: ninguna particion vacia, sin solapes, cubren todo.
    for nombre, pos in posiciones.items():
        if len(pos) == 0:
            raise ValueError(f"La particion '{nombre}' quedo vacia; revise los cortes en config.py")
    total = sum(len(p) for p in posiciones.values())
    if total != len(fechas):
        raise ValueError("Las particiones no cubren exactamente todas las filas")
    return posiciones


def resumen_particion(df, posiciones, col_fecha, col_objetivo):
    """Tabla con filas, rango de fechas y tasa de criticos por particion."""
    filas = []
    for nombre in NOMBRES:
        sub = df.iloc[posiciones[nombre]]
        tasa = float(sub[col_objetivo].mean())
        filas.append({
            "Particion": nombre,
            "Filas": len(sub),
            "Desde": sub[col_fecha].min().date(),
            "Hasta": sub[col_fecha].max().date(),
            "Criticos": int(sub[col_objetivo].sum()),
            "Pct_critico": round(tasa * 100, 3),
            "Baseline_exactitud": round(1 - tasa, 4),   # exactitud de "todo normal"
        })
    return pd.DataFrame(filas)
