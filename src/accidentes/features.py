"""
CONSTRUCCION DE VARIABLES (deterministas)
=============================================================================
Responsabilidad unica: crear variables a partir de reglas FIJAS (la fecha de
cada fila, palabras clave del clima). Ningun parametro se aprende de los
datos, asi que se pueden aplicar antes de la particion temporal sin causar
fuga de informacion.

Estas funciones tambien las reutiliza la aplicacion: el formulario recibe
una fecha, un clima y una ubicacion, y debe construir exactamente las mismas
variables que vio el modelo al entrenar.
=============================================================================
"""

import re

import numpy as np
import pandas as pd

from accidentes.config import (COL_OBJETIVO, COL_SEVERIDAD, ESTACIONES, HORAS_PICO,
                    REGLAS_CLIMA, SEVERIDAD_MINIMA_CRITICA)


def agregar_variables_temporales(df, col_fecha="Start_Time"):
    """Hora, Dia_semana (0=lunes), Mes, Anio, Es_fin_de_semana, Es_hora_pico, Estacion.

    Anio se conserva solo para trazabilidad: NO es predictor (ver config.py)."""
    fecha = df[col_fecha]
    df["Hora"] = fecha.dt.hour.astype("int8")
    df["Dia_semana"] = fecha.dt.dayofweek.astype("int8")
    df["Mes"] = fecha.dt.month.astype("int8")
    df["Anio"] = fecha.dt.year.astype("int16")
    df["Es_fin_de_semana"] = df["Dia_semana"] >= 5

    en_pico = np.zeros(len(df), dtype=bool)
    for desde, hasta in HORAS_PICO:
        en_pico |= df["Hora"].between(desde, hasta).to_numpy()
    df["Es_hora_pico"] = en_pico & (df["Dia_semana"] < 5).to_numpy()   # solo dia habil

    df["Estacion"] = df["Mes"].map(ESTACIONES)
    return df


def agrupar_condicion_climatica(valor):
    """Mapea un texto crudo ('Light Rain with Thunder') a una categoria amplia
    ('Tormenta') buscando palabras clave en orden de prioridad. Los nulos
    quedan nulos; los 'N/A ...' son un dato faltante disfrazado de texto."""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).lower()
    if "n/a" in texto:
        return np.nan
    for etiqueta, palabras in REGLAS_CLIMA:
        if any(palabra in texto for palabra in palabras):
            return etiqueta
    return "Otro"


def consolidar_weather_condition(df):
    """Aplica la agrupacion y devuelve (df, mapa) donde `mapa` es una tabla
    Original -> Grupo, para guardarla como parte de la trazabilidad."""
    originales = df["Weather_Condition"].dropna().unique()
    mapa = {v: agrupar_condicion_climatica(v) for v in originales}
    df["Weather_Condition"] = df["Weather_Condition"].map(mapa)
    tabla = pd.DataFrame({"Original": list(mapa.keys()), "Grupo": list(mapa.values())})
    return df, tabla.sort_values(["Grupo", "Original"], na_position="last")


def agregar_objetivo_binario(df):
    """Critico = Severity >= SEVERIDAD_MINIMA_CRITICA (1) o normal (0)."""
    df[COL_OBJETIVO] = (df[COL_SEVERIDAD] >= SEVERIDAD_MINIMA_CRITICA).astype("int8")
    return df


# ----------------------------------------------------------------------------
# TIPO DE VIA (derivado del nombre de la calle; regla fija, nada se aprende)
# ----------------------------------------------------------------------------
# Evidencia (eda/diagnostico_tipo_via.py): incluirla subio el PR-AUC de LightGBM de 0.153 a
# 0.190 en val_umbral (y de 0.136 a 0.168 en val_ajuste). Ojo con el SENTIDO: las carreteras
# US y las "Hwy" tienen MAS criticos que la media; interestatales y autopistas, MENOS.
# Un nombre vacio queda como nulo y el preprocesador lo manda a 'Desconocido', igual que
# las demas categoricas.
TIPOS_VIA = ("Interestatal", "Carretera_US", "Ruta_estatal", "Autopista", "Carretera", "Local", "Otro")

_RE_INTERESTATAL = re.compile(r"(^|\s)(i[- ]?\d+[a-z]?|interstate[- ]?\d+[a-z]?)\b")   # I-95, I-35E, Interstate 80
_RE_US = re.compile(r"(^|\s)(us|u\.s\.)[- ]?(hwy|highway|route|rte)?[- ]?\d+")
_RE_CONDADO = re.compile(r"county (road|rd|highway|hwy|route)|^cr[- ]?\d+")
_RE_ESTATAL = re.compile(r"^[a-z]{2}-\d+|\bstate (route|road|rd|hwy|highway)\b|^(sr|rte|route)[- ]?\d+")
_RE_AUTOPISTA = re.compile(r"\b(fwy|freeway|expy|expwy|expressway|pkwy|parkway|tpke|turnpike|beltway|bltwy|"
                           r"thruway|trwy|tollway|tlwy|toll|bypass|byp|northway|greeneway)\b")
_RE_CARRETERA = re.compile(r"\b(hwy|highway)\b")
_RE_LOCAL = re.compile(r"\b(st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|ct|court|way|pl|place|"
                       r"cir|circle|ter|terrace|trl|trail|loop|sq|square|aly|alley|path|pike|row|run|walk|plz|plaza)\b")


def clasificar_via(calle):
    """'I-95 N' -> 'Interestatal', 'W Main St' -> 'Local', ... en orden de prioridad.
    Nombre vacio -> nulo.

    Es una regla de texto fija (como la del clima): se puede aplicar antes de la particion
    temporal sin fuga, y la aplicacion ofrece estas mismas categorias en un desplegable."""
    if pd.isna(calle) or not str(calle).strip():
        return np.nan
    t = str(calle).lower().strip()
    if _RE_INTERESTATAL.search(t):      # incluye 'Ramp To I-95 N'
        return "Interestatal"
    if _RE_US.search(t):
        return "Carretera_US"
    if _RE_CONDADO.search(t):
        return "Local"
    if _RE_ESTATAL.search(t):
        return "Ruta_estatal"
    if _RE_AUTOPISTA.search(t):
        return "Autopista"
    if _RE_CARRETERA.search(t):
        return "Carretera"
    if _RE_LOCAL.search(t):
        return "Local"
    return "Otro"


def tipos_de_via(calles):
    """Serie de tipos de via; clasifica solo los nombres UNICOS (hay muchos menos que filas)."""
    unicos = pd.Series(calles.dropna().unique())
    mapa = {c: clasificar_via(c) for c in unicos}
    return calles.map(mapa)


def agregar_tipo_via(df, col_calle="Street"):
    """Crea 'Tipo_via' y DESCARTA el texto de la calle (alta cardinalidad, no es predictor)."""
    df["Tipo_via"] = tipos_de_via(df[col_calle])
    return df.drop(columns=[col_calle])
