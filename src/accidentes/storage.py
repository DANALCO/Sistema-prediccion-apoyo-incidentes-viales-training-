"""
ALMACENAMIENTO DE TABLAS INTERMEDIAS
=============================================================================
Responsabilidad unica: guardar y leer las tablas que pasan de un script al
siguiente. Usa Parquet (rapido, conserva tipos) y, si no esta instalado
pyarrow, cae a CSV con encoding explicito.

    pip install pyarrow        (recomendado)
=============================================================================
"""

import os
import pandas as pd

from accidentes.config import ENCODING

COLUMNAS_FECHA = ("Start_Time", "End_Time")


def guardar_tabla(df, ruta_base):
    """Guarda df en <ruta_base>.parquet (o .csv si no hay motor parquet).
    Devuelve la ruta realmente escrita."""
    os.makedirs(os.path.dirname(ruta_base) or ".", exist_ok=True)
    try:
        ruta = ruta_base + ".parquet"
        df.to_parquet(ruta, index=False)
    except ImportError:
        ruta = ruta_base + ".csv"
        print("  Aviso: pyarrow no esta instalado; se guarda en CSV (mas lento).")
        df.to_csv(ruta, index=False, encoding=ENCODING)
    return ruta


def leer_tabla(ruta_base):
    """Lee <ruta_base>.parquet o, si no existe, <ruta_base>.csv."""
    ruta_parquet = ruta_base + ".parquet"
    if os.path.exists(ruta_parquet):
        return pd.read_parquet(ruta_parquet)
    ruta_csv = ruta_base + ".csv"
    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv, encoding=ENCODING,
                           parse_dates=[c for c in COLUMNAS_FECHA])
    raise FileNotFoundError(
        f"No existe {ruta_parquet} ni {ruta_csv}. Ejecute antes 01_limpiar_dataset.py."
    )
