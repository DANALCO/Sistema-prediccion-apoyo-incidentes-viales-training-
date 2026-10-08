"""
Limpieza del dataset US Accidents (2016-2023)
Proyecto: Predicción de severidad en accidentes de tránsito

Este script:
1. Carga SOLO las columnas relevantes para predecir la severidad (descarta el resto).
2. Elimina cualquier fila que tenga un valor nulo en esas columnas.
3. Elimina filas con valores incoherentes (fuera de un rango lógico).
4. Elimina duplicados.
5. Guarda el resultado limpio en CSV, y también en Excel SI el tamaño final lo permite.
"""

import os
from pathlib import Path

import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Las rutas son relativas a la ubicación de este script (carpeta notebooks/),
# así funcionan igual en el computador de cualquier integrante del equipo.
#
# Estructura esperada del repositorio:
#   proyecto/
#   ├── data/US_Accidents_March23.csv   <-- dataset original (Muy pesado para GitHub)
#   └── notebooks/
#       ├── 01_LimpiarDataSet.py        <-- este script
#       └── output/                     <-- aquí se guardan los resultados
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data" / "US_Accidents_March23.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en: {INPUT_PATH}\n"
        "Descárgalo y colócalo en la carpeta data/ con el nombre "
        "US_Accidents_March23.csv"
    )

# Columnas que SÍ sirven para predecir la severidad
# (variable objetivo + clima + tiempo + infraestructura vial)
COLUMNAS_UTILES = [
    "Severity",              # variable objetivo (target)
    "Start_Time",            # para extraer hora, día de la semana, mes
    "State",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

# Columnas que se DESCARTAN y por qué (documentado para el informe del proyecto):
#   ID                             -> solo identificador, sin valor predictivo
#   Source                         -> metadata del proveedor de datos, no describe el accidente
#   End_Time                       -> fuga de datos: solo se conoce DESPUÉS de terminado el accidente
#   End_Lat, End_Lng               -> fuga de datos: idem
#   Distance(mi)                   -> fuga de datos: es consecuencia de la severidad, no una causa
#   Description                    -> texto libre; requeriría NLP, fuera del alcance del proyecto
#   Street, City, County, Zipcode  -> demasiado granulares / alta cardinalidad para el modelo
#   Country                        -> valor constante ("US"), no aporta información
#   Timezone                       -> redundante con State
#   Airport_Code                   -> metadata de la estación del clima, no del accidente
#   Weather_Timestamp              -> redundante con Start_Time
#   Start_Lat, Start_Lng           -> se usa State como variable geográfica en su lugar

# Tipos de dato optimizados para reducir uso de memoria en un dataset de millones de filas
BOOL_COLS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
             "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
             "Traffic_Signal", "Turning_Loop"]

CATEGORY_COLS = ["State", "Wind_Direction", "Weather_Condition", "Sunrise_Sunset",
                  "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]

dtype_dict = {col: "bool" for col in BOOL_COLS}
dtype_dict.update({col: "category" for col in CATEGORY_COLS})

# ============================================================
# 2. CARGA DE DATOS (solo columnas relevantes -> menos memoria)
# ============================================================

print(f"Cargando datos desde: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, usecols=COLUMNAS_UTILES, dtype=dtype_dict)
print(f"Filas cargadas: {len(df):,} | Columnas: {len(df.columns)}")

print("\nValores nulos por columna (ANTES de limpiar):")
print(df.isnull().sum().sort_values(ascending=False))

# ============================================================
# 3. LIMPIEZA
# ============================================================

filas_inicial = len(df)

# 3.1 Convertir Start_Time a fecha real; lo que no se pueda convertir queda como nulo
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

# 3.2 Eliminar cualquier fila con un valor nulo en las columnas relevantes
df = df.dropna(subset=COLUMNAS_UTILES)
print(f"\nDespués de eliminar nulos: {len(df):,} filas")

# 3.3 Eliminar duplicados exactos
df = df.drop_duplicates()
print(f"Después de eliminar duplicados: {len(df):,} filas")

# 3.4 Eliminar filas con valores incoherentes (fuera de un rango lógico posible)
df = df[df["Severity"].between(1, 4)]
df = df[df["Humidity(%)"].between(0, 100)]
df = df[df["Visibility(mi)"] >= 0]
df = df[df["Wind_Speed(mph)"] >= 0]
df = df[df["Precipitation(in)"] >= 0]
df = df[df["Pressure(in)"] > 0]
df = df[df["Temperature(F)"].between(-50, 150)]
df = df[df["Wind_Chill(F)"].between(-80, 150)]
print(f"Después de eliminar valores incoherentes: {len(df):,} filas")

filas_final = len(df)
porcentaje = (filas_final / filas_inicial) * 100
print(f"\nResumen: se conservó el {porcentaje:.1f}% de las filas originales "
      f"({filas_final:,} de {filas_inicial:,}).")

# ============================================================
# 4. GUARDAR RESULTADO
# ============================================================

csv_path = os.path.join(OUTPUT_DIR, "us_accidents_limpio.csv")
df.to_csv(csv_path, index=False)
print(f"\nArchivo CSV limpio guardado en: {csv_path}")

LIMITE_EXCEL = 1_048_576  # límite de filas por hoja en Excel
if filas_final <= LIMITE_EXCEL - 1:  # -1 por la fila de encabezado
    xlsx_path = os.path.join(OUTPUT_DIR, "us_accidents_limpio.xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Archivo Excel guardado en: {xlsx_path}")
else:
    print(
        f"\nAVISO: el resultado tiene {filas_final:,} filas, que supera el límite "
        f"de Excel ({LIMITE_EXCEL:,} filas por hoja). Por eso NO se generó un .xlsx "
    )