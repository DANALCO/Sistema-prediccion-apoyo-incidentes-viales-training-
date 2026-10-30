"""
LIMPIEZA DE DATOS
=============================================================================
Responsabilidad unica: pasos SEGUROS de limpieza, previos a cualquier
division temporal. Son seguros porque NO dependen de la distribucion de los
datos: que la humedad no pueda superar 100% es una regla del mundo fisico,
no un valor calculado con el dataset.

Lo que este modulo NO hace (y por que): no imputa, no recorta con IQR, no
codifica, no escala y no balancea. Todo eso "aprende" parametros de los
datos y debe calcularse SOLO con train (ver preprocessing.py). Hacerlo aqui
produciria DATA LEAKAGE.
=============================================================================
"""

import numpy as np
import pandas as pd


def normalizar_booleanas(df, columnas):
    """Los nulos de una columna si/no se leen como 'no' (no se inventa nada:
    en estos datos las columnas de infraestructura no traen nulos)."""
    for col in columnas:
        df[col] = df[col].fillna(False).astype(bool)
    return df


def _parsear_fechas(serie):
    """Parseo rapido por formato fijo; lo que falle se reintenta con 'mixed'."""
    texto = serie.astype(str).str[:19]
    fechas = pd.to_datetime(texto, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    fallo = fechas.isna() & serie.notna()
    if fallo.any():
        fechas[fallo] = pd.to_datetime(serie[fallo], errors="coerce", format="mixed")
    return fechas


def convertir_fechas(df, columnas=("Start_Time", "End_Time")):
    for col in columnas:
        if col in df.columns:
            df[col] = _parsear_fechas(df[col])
    return df


def eliminar_fechas_invalidas(df, bitacora):
    n = len(df)
    df = df[df["Start_Time"].notna()].reset_index(drop=True)
    bitacora.registrar("Fecha invalida", n, len(df),
                       "Start_Time no convertible a fecha; es la base de las variables "
                       "temporales y de la particion temporal, y no puede imputarse")
    return df


def validar_objetivo(df, bitacora, col_severidad="Severity"):
    """Sin etiqueta no hay aprendizaje supervisado; el objetivo nunca se imputa."""
    n = len(df)
    df = df[df[col_severidad].notna() & df[col_severidad].between(1, 4)].reset_index(drop=True)
    df[col_severidad] = df[col_severidad].astype("int8")
    bitacora.registrar("Target invalido", n, len(df),
                       "Severity nula o fuera del rango 1-4; el target nunca se imputa")
    return df


def eliminar_duplicados(df, bitacora, ignorar=()):
    """Quita filas repetidas (se conserva la primera). `ignorar`: columnas que NO cuentan para
    decidir si dos filas son iguales (por ejemplo Tipo_via, que se derivo de un texto que ya
    se descarto); asi el conjunto de filas que sobrevive no depende de ellas."""
    n = len(df)
    columnas = [c for c in df.columns if c not in set(ignorar)]
    df = df.drop_duplicates(subset=columnas).reset_index(drop=True)
    nota = "Filas identicas en todas las columnas; aportan peso artificial"
    if ignorar:
        nota = f"Filas identicas en todas las columnas salvo {', '.join(ignorar)}; aportan peso artificial"
    bitacora.registrar("Duplicados", n, len(df), nota)
    return df


def marcar_valores_imposibles(df, limites):
    """Valores fisicamente imposibles -> NaN. NO se borra la fila: si un
    sensor reporto 150% de humedad, no tiene sentido perder tambien la hora,
    el estado y la infraestructura de ese accidente. El preprocesador lo
    imputa despues con la mediana de train.

    Devuelve (df, resumen) con cuantos valores se marcaron por columna."""
    resumen = {}
    for col, (minimo, maximo) in limites.items():
        if col not in df.columns:
            continue
        fuera = ~df[col].between(minimo, maximo) & df[col].notna()
        cuantos = int(fuera.sum())
        if cuantos:
            df.loc[fuera, col] = np.nan
            resumen[col] = cuantos
            print(f"  {col:<20} {cuantos:>10,} valores fuera de [{minimo}, {maximo}] -> NaN")
    if not resumen:
        print("  (no se encontraron valores fuera de rango fisico)")
    return df, resumen


def agregar_duracion(df):
    """Duracion del incidente: SOLO para trazabilidad y para documentar el
    hallazgo en el informe. NO es predictor (ver config.py)."""
    df["Duracion_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    df.loc[df["Duracion_min"] < 0, "Duracion_min"] = np.nan
    df.loc[df["Duracion_min"] > 60 * 24 * 30, "Duracion_min"] = np.nan   # > 30 dias
    return df


def optimizar_memoria(df):
    """Downcast de numericos. Devuelve (df, memoria_antes_mb, memoria_despues_mb)."""
    antes = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    despues = df.memory_usage(deep=True).sum() / 1024 ** 2
    return df, antes, despues
