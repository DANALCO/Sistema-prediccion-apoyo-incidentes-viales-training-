"""
=============================================================================
SCRIPT 1 DE 2  -  LIMPIEZA DEL DATASET
=============================================================================
Proyecto : Sistema de prediccion en tiempo real y apoyo a la decision
           para la gestion de incidentes viales
Dataset  : US Accidents (2016-2023), Moosavi [23]
Entregable 2  ->  criterio "Construccion del Dataset"

-----------------------------------------------------------------------------
QUE HACE ESTE SCRIPT (solo pasos SEGUROS, previos a la division train/test)
-----------------------------------------------------------------------------
  1. Carga solo las columnas necesarias, con tipos optimizados en memoria.
  2. Inspecciona el dataset: filas, columnas, nulos, memoria, clases.
  3. Convierte fechas y genera variables temporales derivadas.
  4. Elimina duplicados exactos.
  5. Marca como NaN los valores fisicamente imposibles (NO borra la fila).
  6. Elimina unicamente las filas sin variable objetivo valida.
  7. Reduce los tipos numericos (downcast).
  8. Guarda el dataset limpio y, opcionalmente, una muestra estratificada.

-----------------------------------------------------------------------------
QUE **NO** HACE ESTE SCRIPT, Y POR QUE
-----------------------------------------------------------------------------
  - NO imputa nulos         -> la mediana debe calcularse SOLO con train
  - NO recorta outliers IQR -> Q1 y Q3 deben calcularse SOLO con train
  - NO codifica ni escala   -> aprenden parametros a partir de los datos
  - NO aplica SMOTE         -> va solo sobre train, jamas sobre test

  Todo lo anterior va en un Script 2, DESPUES de train_test_split.
  Hacerlo aqui produciria DATA LEAKAGE: el modelo "espiaria" el examen y
  las metricas finales saldrian infladas y no serian defendibles.

  Los pasos que SI estan aqui son seguros porque no dependen de la
  distribucion de los datos. Que la humedad no pueda superar el 100% es una
  regla del mundo fisico, no un valor calculado a partir del dataset.
=============================================================================
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. CONFIGURACION
# =============================================================================

# Las rutas son relativas a la ubicacion de este script (carpeta notebooks/),
# asi funcionan igual en el computador de cualquier integrante del equipo.
#
# Estructura esperada del repositorio:
#   proyecto/
#   |-- data/US_Accidents_March23.csv   <-- dataset original (NO se sube a GitHub por el peso)
#   `-- notebooks/
#       |-- 01_limpiar_dataset.py       <-- este script
#       |-- 02_preparar_datos.py
#       `-- output/                     <-- aqui se guardan los resultados
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data" / "US_Accidents_March23.csv"
OUTPUT_DIR = BASE_DIR / "output"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset en: {INPUT_PATH}\n"
        "Descargalo y colocalo en la carpeta data/ con el nombre "
        "US_Accidents_March23.csv"
    )

# El dataset completo tiene ~7.7 millones de filas. Entrenar Random Forest y
# XGBoost sobre ese volumen (mas one-hot encoding y SMOTE) puede tardar horas
# y agotar la RAM de un portatil. La rubrica exige mas de 35.000 registros.
# Una muestra estratificada mantiene la proporcion original de cada clase de
# Severity y queda MUY por encima de ese umbral.

GUARDAR_MUESTRA = True
TAMANO_MUESTRA = 300_000
SEMILLA = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. DEFINICION DE COLUMNAS
# =============================================================================
# La separacion en dos grupos es el nucleo metodologico del proyecto y
# responde directamente a la pregunta de investigacion 4.

TARGET = "Severity"

# ---- GRUPO A: informacion disponible AL MOMENTO DEL REPORTE -----------------
# Es lo unico que tendria un operador de un centro de gestion de trafico en el
# instante en que entra la llamada. Alimenta el MODELO DE TIEMPO REAL.

COLS_TEMPORALES = ["Start_Time"]

COLS_GEOGRAFICAS = ["State", "Start_Lat", "Start_Lng"]

COLS_CLIMA = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Direction", "Wind_Speed(mph)",
    "Precipitation(in)", "Weather_Condition",
]

COLS_INFRAESTRUCTURA = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]

COLS_LUZ = [
    "Sunrise_Sunset", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight",
]

COLS_TIEMPO_REAL = (
    COLS_TEMPORALES + COLS_GEOGRAFICAS + COLS_CLIMA
    + COLS_INFRAESTRUCTURA + COLS_LUZ
)

# ---- GRUPO B: informacion conocida SOLO DESPUES del incidente --------------
# NO puede usarla el modelo de tiempo real. Existe unicamente para construir
# el MODELO DE REFERENCIA con informacion completa contra el cual se mide la
# perdida de desempeno.
#
# Nota:
# se sospechaba fuga de datos (Severity se define por la duracion del
# retraso), pero la correlacion resulto casi nula (-0.003 a -0.004). El
# motivo: el proveedor de datos inyecta valores "por defecto" cuando no
# captura el cierre real del incidente. Evidencia encontrada:
#   - 360.0 min (6 horas) exactas en 4.81% de las filas (364,859 filas)
#   - 30.0 min exactas en 1.47% de las filas (111,610 filas)
#   Un pico de casi el 5% de 7.58 millones de filas en un unico valor
#   decimal es estadisticamente imposible en una variable continua real;
#   confirma que es un artefacto del proveedor, no una medicion genuina.
#
# DECISION DE MODELADO (Opcion A, la mas simple): Duracion_min y End_Time
# NO se usan como predictores en ningun escenario. El modelo de referencia
# con "informacion completa" (pregunta de investigacion 4) representa esa
# informacion posterior UNICAMENTE con Distance(mi), cuya correlacion
# (0.035) tambien es baja en terminos lineales, pero Pearson no detecta
# relaciones no lineales; su aporte real se evaluara despues del
# entrenamiento con SHAP, no se descarta a priori por este numero.
#
# Duracion_min y End_Time se conservan en el CSV limpio solo para
# trazabilidad y para documentar el hallazgo en el informe (seccion de
# limitaciones); el Script 2 debe EXCLUIRLAS al construir las matrices de
# entrenamiento, usando las listas COLS_MODELADO_* definidas abajo.
COLS_INFO_COMPLETA = ["Distance(mi)", "End_Time"]

COLUMNAS_A_CARGAR = [TARGET] + COLS_TIEMPO_REAL + COLS_INFO_COMPLETA

# ---- Listas que usara el Script 2 (OJO AL QUE ESTE HACIENDO ESA PARTE) para armar cada escenario de modelado ----
# Fuente unica de verdad: si se cambia de opinion sobre que columnas
# entran en cada escenario, se edita SOLO aqui.
COLS_MODELADO_TIEMPO_REAL = list(COLS_TIEMPO_REAL)
COLS_MODELADO_INFO_COMPLETA = list(COLS_TIEMPO_REAL) + ["Distance(mi)"]

# ---- Columnas descartadas y su justificacion (para el informe) -------------
#   ID                       -> identificador, sin valor predictivo
#   Source                   -> metadata del proveedor, no describe el accidente
#   End_Lat, End_Lng         -> redundantes con Distance(mi) para el escenario completo
#   Description              -> texto libre; exigiria NLP, fuera del alcance
#   Street, City, County, Zipcode -> cardinalidad demasiado alta para one-hot
#   Country                  -> constante ("US"), no aporta informacion
#   Timezone                 -> redundante con State
#   Airport_Code             -> metadata de la estacion meteorologica
#   Weather_Timestamp        -> redundante con Start_Time

# ---- Tipos optimizados para la lectura -------------------------------------
BOOL_COLS = COLS_INFRAESTRUCTURA
CATEGORY_COLS = ["State", "Wind_Direction", "Weather_Condition"] + COLS_LUZ

dtype_dict = {c: "bool" for c in BOOL_COLS}
dtype_dict.update({c: "category" for c in CATEGORY_COLS})


# =============================================================================
# 3. UTILIDADES DE TRAZABILIDAD
# =============================================================================
# El criterio "Simulacion de Resultados" de la rubrica evalua la trazabilidad.
# Este registro documenta cuantas filas se pierden en cada paso y por que.

bitacora = []


def registrar(paso, filas_antes, filas_despues, motivo):
    eliminadas = filas_antes - filas_despues
    pct = (eliminadas / filas_antes * 100) if filas_antes else 0
    bitacora.append({
        "Paso": paso,
        "Filas antes": filas_antes,
        "Filas despues": filas_despues,
        "Eliminadas": eliminadas,
        "% eliminado": round(pct, 3),
        "Motivo": motivo,
    })
    print(f"  [{paso}] {filas_antes:,} -> {filas_despues:,} "
          f"({eliminadas:,} eliminadas, {pct:.3f}%)")


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# =============================================================================
# 4. CARGA
# =============================================================================

titulo("PASO 1 - CARGA DE DATOS")
print(f"Archivo: {INPUT_PATH}")
print("Cargando solo las columnas necesarias (ahorra memoria)...")

df = pd.read_csv(INPUT_PATH, usecols=COLUMNAS_A_CARGAR, dtype=dtype_dict)

filas_originales = len(df)
print(f"\nFilas cargadas   : {filas_originales:,}")
print(f"Columnas cargadas: {len(df.columns)}")
print(f"Memoria en uso   : {df.memory_usage(deep=True).sum() / 1024**2:,.1f} MB")


# =============================================================================
# 5. INSPECCION INICIAL
# =============================================================================

titulo("PASO 2 - INSPECCION INICIAL")

print("Valores nulos por columna (solo las que tienen nulos):")
nulos = df.isnull().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)
if len(nulos) == 0:
    print("  (ninguna columna tiene nulos)")
else:
    for col, n in nulos.items():
        print(f"  {col:<25} {n:>12,}  ({n / filas_originales * 100:6.2f}%)")

print("\nDistribucion de la variable objetivo (Severity):")
dist = df[TARGET].value_counts().sort_index()
for clase, n in dist.items():
    print(f"  Severity {clase}: {n:>12,}  ({n / filas_originales * 100:6.2f}%)")

ratio = dist.max() / dist.min()
print(f"\nRatio de desbalance (clase mayoritaria / minoritaria): {ratio:.1f} a 1")
print("Este desbalance justifica el uso de SMOTE y de metricas macro")
print("(F1-macro, AUC) en lugar de la exactitud global. -> Script 2")


# =============================================================================
# 6. LIMPIEZA
# =============================================================================

titulo("PASO 3 - LIMPIEZA")

# --- 6.1 Conversion de fechas ----------------------------------------------
print("\n3.1 Conversion de fechas a datetime")
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce", format="mixed")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce", format="mixed")

n = len(df)
df = df[df["Start_Time"].notna()]
registrar("Fecha invalida", n, len(df),
          "Start_Time no convertible a fecha; es la base de todas las "
          "variables temporales y no puede imputarse")

# --- 6.2 Variable objetivo valida ------------------------------------------
# Unica eliminacion de filas obligatoria: sin etiqueta no hay aprendizaje
# supervisado posible, y el target jamas debe imputarse.
print("\n3.2 Validacion de la variable objetivo")
n = len(df)
df = df[df[TARGET].notna() & df[TARGET].between(1, 4)]
df[TARGET] = df[TARGET].astype("int8")
registrar("Target invalido", n, len(df),
          "Severity nula o fuera del rango 1-4; el target nunca se imputa")

# --- 6.3 Ingenieria de caracteristicas temporales --------------------------
# Estas transformaciones son DETERMINISTAS: se derivan de la fecha de cada
# fila, no de la distribucion del dataset. Por eso pueden ir antes de la
# division train/test sin causar leakage.
print("\n3.3 Ingenieria de caracteristicas temporales")

df["Hora"] = df["Start_Time"].dt.hour.astype("int8")
df["Dia_semana"] = df["Start_Time"].dt.dayofweek.astype("int8")   # 0=lunes
df["Mes"] = df["Start_Time"].dt.month.astype("int8")
df["Anio"] = df["Start_Time"].dt.year.astype("int16")

df["Es_fin_de_semana"] = (df["Dia_semana"] >= 5)

# Hora pico: solo cuenta en dia habil (pregunta de investigacion 2)
df["Es_hora_pico"] = (
    df["Hora"].between(7, 9) | df["Hora"].between(16, 19)
) & (df["Dia_semana"] < 5)

estaciones = {12: "Invierno", 1: "Invierno", 2: "Invierno",
              3: "Primavera", 4: "Primavera", 5: "Primavera",
              6: "Verano", 7: "Verano", 8: "Verano",
              9: "Otonio", 10: "Otonio", 11: "Otonio"}
df["Estacion"] = df["Mes"].map(estaciones).astype("category")

print("  Creadas: Hora, Dia_semana, Mes, Anio, Es_fin_de_semana,")
print("           Es_hora_pico, Estacion")

# --- 6.4 Duracion (solo escenario de informacion completa) -----------------
print("\n3.4 Duracion del incidente (escenario de informacion completa)")
df["Duracion_min"] = (
    (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
)
# Duraciones negativas o absurdas -> NaN, no se borra la fila
df.loc[df["Duracion_min"] < 0, "Duracion_min"] = np.nan
df.loc[df["Duracion_min"] > 60 * 24 * 30, "Duracion_min"] = np.nan  # > 30 dias

corr = df[["Duracion_min", TARGET]].corr().iloc[0, 1]
print(f"  Correlacion Duracion_min vs Severity: {corr:.3f}")
print("  ADVERTENCIA: Severity esta DEFINIDA por la duracion del retraso.")
print("  Si esta correlacion es alta, incluir Duracion_min en el modelo de")
print("  referencia haria la comparacion circular. Recomendacion: excluirla")
print("  del modelado y usar solo Distance(mi) como informacion posterior.")

# --- 6.5 Duplicados exactos -------------------------------------------------
print("\n3.5 Duplicados exactos")
n = len(df)
df = df.drop_duplicates()
registrar("Duplicados", n, len(df),
          "Filas identicas en todas las columnas; aportan peso artificial")

# --- 6.6 Valores fisicamente imposibles -> NaN ------------------------------
# DIFERENCIA CLAVE frente a la version anterior del script: estos valores ya
# no eliminan la fila completa. Si un sensor meteorologico reporto 150% de
# humedad, no tiene sentido perder tambien la hora, el estado y la
# infraestructura vial de ese accidente. El valor se marca como faltante y el
# Script 2 lo imputa (con la mediana calculada SOLO sobre train).
print("\n3.6 Valores fisicamente imposibles -> NaN")

limites_fisicos = {
    "Humidity(%)":       (0, 100),     # definicion de porcentaje
    "Visibility(mi)":    (0, 150),     # visibilidad negativa no existe
    "Wind_Speed(mph)":   (0, 250),     # por encima de cualquier huracan
    "Precipitation(in)": (0, 50),      # por hora, extremo generoso
    "Pressure(in)":      (20, 35),     # presion barometrica plausible
    "Temperature(F)":    (-60, 135),   # records mundiales con margen
    "Wind_Chill(F)":     (-100, 135),
    "Distance(mi)":      (0, 500),
}

total_marcados = 0
for col, (minimo, maximo) in limites_fisicos.items():
    if col not in df.columns:
        continue
    fuera = ~df[col].between(minimo, maximo) & df[col].notna()
    cuantos = int(fuera.sum())
    if cuantos > 0:
        df.loc[fuera, col] = np.nan
        total_marcados += cuantos
        print(f"  {col:<20} {cuantos:>10,} valores fuera de "
              f"[{minimo}, {maximo}] -> NaN")

if total_marcados == 0:
    print("  (no se encontraron valores fuera de rango fisico)")
else:
    print(f"\n  Total marcado como NaN: {total_marcados:,} valores")
    print("  Ninguna fila fue eliminada por este paso.")


# =============================================================================
# 7. OPTIMIZACION DE MEMORIA
# =============================================================================

titulo("PASO 4 - OPTIMIZACION DE MEMORIA (downcast)")

mem_antes = df.memory_usage(deep=True).sum() / 1024**2

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")

mem_despues = df.memory_usage(deep=True).sum() / 1024**2
ahorro = (1 - mem_despues / mem_antes) * 100 if mem_antes else 0

print(f"Memoria antes  : {mem_antes:,.1f} MB")
print(f"Memoria despues: {mem_despues:,.1f} MB")
print(f"Ahorro         : {ahorro:.1f}%")


# =============================================================================
# 8. RESUMEN
# =============================================================================

titulo("PASO 5 - RESUMEN DE LA LIMPIEZA")

resumen = pd.DataFrame(bitacora)
if not resumen.empty:
    print(resumen.to_string(index=False))

filas_finales = len(df)
conservado = filas_finales / filas_originales * 100

print(f"\nFilas originales: {filas_originales:,}")
print(f"Filas finales   : {filas_finales:,}")
print(f"Conservado      : {conservado:.2f}%")

print("\nDistribucion final de Severity:")
dist_final = df[TARGET].value_counts().sort_index()
for clase, n in dist_final.items():
    print(f"  Severity {clase}: {n:>12,}  ({n / filas_finales * 100:6.2f}%)")

print("\nNulos restantes (los imputa el Script 2, despues de dividir):")
nulos_fin = df.isnull().sum()
nulos_fin = nulos_fin[nulos_fin > 0].sort_values(ascending=False)
if len(nulos_fin) == 0:
    print("  (ninguno)")
else:
    for col, n in nulos_fin.items():
        print(f"  {col:<25} {n:>12,}  ({n / filas_finales * 100:6.2f}%)")

print(f"\nRequisito de la rubrica (mas de 35.000 registros): "
      f"{'CUMPLE' if filas_finales > 35_000 else 'NO CUMPLE'}")

print("\nDecision de modelado (Opcion A, confirmada por diagnostico):")
print("  - Duracion_min EXCLUIDA de ambos escenarios de modelado.")
print("    Motivo: ~4.81% de las filas tienen exactamente 360.0 min (6h) y")
print("    ~1.47% exactamente 30.0 min; son valores 'por defecto' del")
print("    proveedor de datos, no mediciones reales. Se conserva en el CSV")
print("    unicamente para trazabilidad y para documentar el hallazgo.")
print("  - End_Time EXCLUIDA de ambos escenarios (solo sirvio para calcular")
print("    Duracion_min).")
print("  - El escenario de 'informacion completa' usara UNICAMENTE")
print("    Distance(mi) como variable posterior al reporte.")
print(f"  - COLS_MODELADO_TIEMPO_REAL   : {len(COLS_MODELADO_TIEMPO_REAL)} columnas")
print(f"  - COLS_MODELADO_INFO_COMPLETA : {len(COLS_MODELADO_INFO_COMPLETA)} columnas")


# =============================================================================
# 9. GUARDADO
# =============================================================================

titulo("PASO 6 - GUARDADO")

ruta_completo = os.path.join(OUTPUT_DIR, "us_accidents_limpio.csv")
df.to_csv(ruta_completo, index=False)
tam = os.path.getsize(ruta_completo) / 1024**2
print(f"Dataset completo -> {ruta_completo}  ({tam:,.1f} MB, {filas_finales:,} filas)")

if GUARDAR_MUESTRA and filas_finales > TAMANO_MUESTRA:
    # Muestra ESTRATIFICADA: conserva la proporcion exacta de cada clase de
    # Severity. Un sample aleatorio simple podria dejar casi sin
    # representacion a las clases minoritarias (1 y 4).
    df_muestra, _ = train_test_split(
        df,
        train_size=TAMANO_MUESTRA,
        stratify=df[TARGET],
        random_state=SEMILLA,
    )
    ruta_muestra = os.path.join(OUTPUT_DIR, "us_accidents_muestra.csv")
    df_muestra.to_csv(ruta_muestra, index=False)
    tam_m = os.path.getsize(ruta_muestra) / 1024**2

    print(f"Muestra estratificada -> {ruta_muestra}  "
          f"({tam_m:,.1f} MB, {len(df_muestra):,} filas)")
    print("\n  Proporciones conservadas:")
    for clase in sorted(df[TARGET].unique()):
        p_orig = (df[TARGET] == clase).mean() * 100
        p_mues = (df_muestra[TARGET] == clase).mean() * 100
        print(f"    Severity {clase}: completo {p_orig:6.2f}%  |  "
              f"muestra {p_mues:6.2f}%")

# Bitacora exportada para anexarla al informe
ruta_bitacora = os.path.join(OUTPUT_DIR, "bitacora_limpieza.csv")
resumen.to_csv(ruta_bitacora, index=False)
print(f"\nBitacora de trazabilidad -> {ruta_bitacora}")

print("\n" + "=" * 70)
print("LIMPIEZA COMPLETADA")
print("Siguiente paso: Script 2 -> division train/test, imputacion,")
print("IQR, encoding, scaling, SMOTE y entrenamiento de los 4 modelos.")
print("=" * 70)
