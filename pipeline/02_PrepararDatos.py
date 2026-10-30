"""
=============================================================================
SCRIPT 2 DE 3  -  PREPARACION DE DATOS PARA MODELADO
=============================================================================
Proyecto : Sistema de prediccion en tiempo real y apoyo a la decision
           para la gestion de incidentes viales
Entrada  : el CSV generado por 01_limpiar_dataset.py
Salida   : arreglos numericos listos para entrenar, guardados en disco

-----------------------------------------------------------------------------
ALCANCE DE ESTE SCRIPT (a proposito, no incluye entrenamiento)
-----------------------------------------------------------------------------
Este script llega HASTA justo antes de entrenar. La rubrica exige 4 modelos
de ML, el entrenamiento se hara en un Script 3 aparte.

-----------------------------------------------------------------------------
ORDEN DE OPERACIONES (ojo a esto compañeros)
-----------------------------------------------------------------------------
  1. Cargar el CSV limpio.
  2. Dividir en train/test -> UNA SOLA VEZ, estratificado por Severity.
     Esta MISMA division (las mismas filas) se usa para los dos escenarios
     (tiempo real / informacion completa). Si se dividiera distinto para
     cada uno, la comparacion de la pregunta de investigacion 4 quedaria
     invalidada: no se sabria si la diferencia de desempeno se debe a la
     restriccion de informacion o a que a un modelo le tocaron filas mas
     faciles por pura casualidad.
  3. Para cada escenario, EN ESTE ORDEN:
       a. Imputar numericas -> mediana por grupo de 'State', calculada
          SOLO con train.
       b. Imputar categoricas -> nueva categoria 'Desconocido'.
       c. Recortar outliers -> limites IQR calculados SOLO con train.
       d. Encoding (OneHotEncoder) + Scaling (RobustScaler) -> parametros
          aprendidos SOLO con train, aplicados igual a test.
       e. SMOTE -> UNICAMENTE sobre el train ya transformado. El test
          jamas se balancea: tiene que seguir pareciendose a la realidad.

  Todo lo que "aprende" un parametro de los datos (medianas, limites IQR,
  categorias del encoder, media/mediana del scaler) se calcula EXCLUSIVAMENTE
  sobre train y despues se aplica a test. Hacerlo al reves es data leakage (OJO):
  el modelo "espia" el examen y las metricas finales quedan infladas.

-----------------------------------------------------------------------------
DECISIONES INCORPORADAS DESPUES DEL EDA (analisis_exploratorio_datos.py)
-----------------------------------------------------------------------------
  1. Wind_Chill(F) ELIMINADA: Pearson = 0.994, Spearman = 0.999 con
     Temperature(F); son practicamente la misma variable, y Wind_Chill(F)
     tenia ademas mucho mas nulos (26% vs 2%).
  2. Civil_Twilight, Nautical_Twilight, Astronomical_Twilight ELIMINADAS:
     Cramer's V de hasta 0.889 con Sunrise_Sunset; las 4 miden casi el
     mismo concepto (que tan oscuro esta) a distinta precision solar. Se
     conserva solo Sunrise_Sunset.
  3. Weather_Condition CONSOLIDADA de 96 a ~12 categorias amplias (Lluvia,
     Nieve, Niebla, etc.) mediante palabras clave, antes de dividir
     train/test. Es una regla de texto fija, no calculada de los datos,
     por lo que no genera fuga de informacion.

-----------------------------------------------------------------------------
REQUISITOS
-----------------------------------------------------------------------------
  pip install scikit-learn imbalanced-learn joblib (instalenlo)
=============================================================================
"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib


# =============================================================================
# 1. CONFIGURACION
# =============================================================================

# Las rutas son relativas a la ubicacion de este script (carpeta notebooks/),
# no a la carpeta desde donde se ejecute. Lee lo que genero el Script 1 en
# notebooks/output/ y guarda sus resultados en notebooks/output/preparado/.
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "us_accidents_muestra.csv"
OUTPUT_DIR = BASE_DIR / "output" / "preparado"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo de entrada: {INPUT_PATH}\n"
        "Ejecuta primero 01_limpiar_dataset.py para generarlo."
    )

TARGET = "Severity"
TEST_SIZE = 0.20
SEMILLA = 42
COLUMNA_GRUPO_IMPUTACION = "State"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. DEFINICION DE COLUMNAS
# =============================================================================
# Deben coincidir exactamente con las que genero 01_limpiar_dataset.py.
# Decision de modelado (Opcion A, ver 01_limpiar_dataset.py): Duracion_min
# y End_Time NO se usan como predictores. El escenario "informacion
# completa" se representa unicamente con Distance(mi).

# ---- Numericas que SI tuvieron nulos en la limpieza (necesitan imputacion) --
# DECISION (post-EDA): Wind_Chill(F) se elimina como predictor. El EDA
# (analisis_exploratorio_datos.py, Paso 6) encontro que esta practicamente
# duplicada con Temperature(F): Pearson = 0.994, Spearman = 0.999. Ademas
# tiene mucho mas nulos que Temperature (26% vs 2%), asi que quitarla
# tambien reduce la carga de imputacion sin perder informacion real.
NUM_A_IMPUTAR = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
]

# ---- Numericas que NO tuvieron nulos (no requieren imputacion) -------------
NUM_SIN_NULOS = ["Start_Lat", "Start_Lng", "Hora", "Dia_semana", "Mes", "Anio"]
# Nota: Hora (0-23) y Dia_semana (0-6) son variables ciclicas (23 y 0 son
# consecutivos en el reloj); aqui se tratan como numericas simples porque
# asi decian las diapositivas. Una mejora futura opcional seria codificarlas
# con seno/coseno para que el modelo capture esa continuidad pero tenemos que hablarlo.

# ---- Categoricas (algunas con nulos, otras no; todas se codifican igual) ---
# DECISION (post-EDA): de las 4 columnas de crepusculo, solo se conserva
# Sunrise_Sunset. El EDA (Paso 7, Cramer's V) encontro asociacion muy alta
# entre las 4 (hasta 0.889), porque miden casi el mismo concepto (que tan
# oscuro esta) a distintos niveles de precision solar. Mantener las 4
# infla el numero de columnas tras el encoding sin aportar informacion
# nueva, y diluye el "credito" que SHAP le asignaria a este concepto entre
# 4 variables casi identicas en vez de una sola, clara, para el operador.
CATEGORICAS = [
    "State", "Wind_Direction", "Weather_Condition", "Estacion",
    "Sunrise_Sunset",
]

# ---- Booleanas de infraestructura y las derivadas en el Script 1 ----------
BOOLEANAS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop", "Es_fin_de_semana", "Es_hora_pico",
]

# Columna adicional exclusiva del escenario de informacion completa
COL_INFO_COMPLETA = "Distance(mi)"

# Union de columnas a cargar del CSV (evita leer Start_Time/End_Time/
# Duracion_min, que no se usan como predictores segun la Opcion A)
COLUMNAS_A_CARGAR = (
    [TARGET] + NUM_A_IMPUTAR + NUM_SIN_NULOS + CATEGORICAS + BOOLEANAS
    + [COL_INFO_COMPLETA]
)

CATEGORY_COLS_LECTURA = CATEGORICAS
BOOL_COLS_LECTURA = BOOLEANAS
dtype_dict = {c: "category" for c in CATEGORY_COLS_LECTURA}
dtype_dict.update({c: "bool" for c in BOOL_COLS_LECTURA})


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# =============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO
# =============================================================================
# Cada funcion recibe explicitamente cuales datos son "train" para dejar en
# claro, en cada llamada, de donde se aprenden los parametros.

def obtener_medianas_grupo(df_train, columnas_numericas, columna_grupo):
    """Mediana por grupo (ej. por State) y mediana global de respaldo,
    calculadas UNICAMENTE con las filas de entrenamiento."""
    medianas_grupo = {}
    medianas_globales = {}
    for col in columnas_numericas:
        medianas_grupo[col] = df_train.groupby(columna_grupo, observed=True)[col].median()
        medianas_globales[col] = df_train[col].median()
    return medianas_grupo, medianas_globales


def imputar_numericas(df, columnas_numericas, medianas_grupo, medianas_globales, columna_grupo):
    """Rellena nulos con la mediana del grupo correspondiente; si el grupo
    no tiene dato (ej. un State raro que no aparecio en train), usa la
    mediana global de train como respaldo."""
    df = df.copy()
    for col in columnas_numericas:
        if df[col].isna().sum() == 0:
            continue
        mapa = df[columna_grupo].map(medianas_grupo[col])
        df[col] = df[col].fillna(mapa)
        df[col] = df[col].fillna(medianas_globales[col])
    return df


def imputar_categoricas(df, columnas_categoricas, valor_relleno="Desconocido"):
    """Rellena nulos categoricos con una categoria explicita 'Desconocido'
    en lugar de asumir la moda, para no inventar informacion."""
    df = df.copy()
    for col in columnas_categoricas:
        if df[col].isna().sum() == 0:
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype) and valor_relleno not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories([valor_relleno])
        df[col] = df[col].fillna(valor_relleno)
    return df


def calcular_limites_iqr(df_train, columnas_numericas, factor=1.5):
    """Q1, Q3 e IQR calculados UNICAMENTE con train (Modulo B de la guia).

    Si una columna tiene IQR = 0 (mas de la mitad de los valores de train
    son identicos -- tipico de variables con fuerte concentracion en un
    valor fijo, como Precipitation en 0 o Visibility en su techo de 10 mi),
    NO se recorta. Aplicar la formula tal cual convertiria esa columna en
    una constante: cualquier valor distinto del punto de concentracion
    (ej. un accidente con lluvia real, o con niebla real) quedaria
    forzado a ese unico valor, borrando justo la informacion mas
    interesante de la variable. Los valores fisicamente imposibles ya
    fueron filtrados en el Script 1; esta columna sencillamente no
    necesita (ni puede, con esta formula) un segundo filtro estadistico.
    """
    limites = {}
    for col in columnas_numericas:
        q1 = df_train[col].quantile(0.25)
        q3 = df_train[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            print(f"    AVISO: {col} tiene IQR = 0 (Q1 = Q3 = {q1:.2f}); se omite "
                  f"el recorte para no convertir la variable en una constante.")
            continue
        limites[col] = (q1 - factor * iqr, q3 + factor * iqr)
    return limites


def recortar_outliers(df, limites):
    """clip(): recorta al limite, NO elimina la fila (Modulo B.3 de la guia)."""
    df = df.copy()
    for col, (lower, upper) in limites.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---- DECISION (post-EDA): consolidar Weather_Condition ---------------------
# El EDA (Paso 3) encontro 96 categorias distintas en Weather_Condition. Sin
# agrupar, el OneHotEncoder crearia 96 columnas nuevas solo para esta
# variable, mucho mas que State (49) o Wind_Direction (24). Esto perjudica
# especialmente a modelos sensibles a la dimensionalidad (ej. KNN, uno de
# los candidatos aun sin decidir para completar los 4 modelos de la
# rubrica) y complica la lectura del tablero SHAP para el operador.
#
# Esta es una regla de texto FIJA (basada en palabras clave), no algo que
# se calcule a partir de los datos, asi que es segura de aplicar sobre todo
# el dataset ANTES de dividir train/test: no hay fuga de informacion porque
# ningun parametro se "aprende" de las filas, solo se reescribe texto.
REGLAS_CLIMA = [
    ("Tornado",    ["tornado", "funnel"]),
    ("Tormenta",   ["thunder", "t-storm", "tstorm"]),
    ("Granizo",    ["hail"]),
    ("Hielo",      ["ice", "sleet", "wintry", "freezing"]),
    ("Nieve",      ["snow"]),
    ("Lluvia",     ["rain", "drizzle", "shower"]),
    ("Niebla",     ["fog", "mist", "haze"]),
    ("Humo_Polvo", ["smoke", "ash", "dust", "sand"]),
    ("Viento",     ["squall", "wind"]),
    ("Nublado",    ["cloud", "overcast"]),
    ("Despejado",  ["clear", "fair"]),
]


def agrupar_condicion_climatica(valor):
    """Mapea un texto crudo de Weather_Condition (ej. 'Light Rain with
    Thunder') a una categoria amplia (ej. 'Tormenta'), buscando palabras
    clave en orden de prioridad. Los nulos se dejan igual: la imputacion
    categorica (Desconocido) se encarga de ellos despues. Valores tipo
    'N/A ...' son en realidad un marcador de dato faltante disfrazado de
    texto, asi que se tratan tambien como nulos en vez de caer en 'Otro'."""
    if pd.isna(valor):
        return valor
    texto = valor.lower()
    if "n/a" in texto:
        return np.nan
    for etiqueta, palabras in REGLAS_CLIMA:
        if any(palabra in texto for palabra in palabras):
            return etiqueta
    return "Otro"


def consolidar_weather_condition(df):
    """Aplica agrupar_condicion_climatica e imprime, por transparencia, en
    que categoria amplia cayo cada categoria original (para poder auditar
    a mano si alguna quedo mal clasificada)."""
    originales = df["Weather_Condition"].dropna().unique()
    mapa = {v: agrupar_condicion_climatica(v) for v in originales}

    print(f"Weather_Condition: {len(originales)} categorias originales -> "
          f"{len(set(v for v in mapa.values() if pd.notna(v)))} categorias agrupadas")

    tabla = pd.DataFrame({"Original": list(mapa.keys()), "Grupo": list(mapa.values())})
    resumen = tabla.groupby("Grupo")["Original"].apply(list).sort_index()
    for grupo, originales_del_grupo in resumen.items():
        print(f"  {grupo:<12} ({len(originales_del_grupo)} categorias originales)")

    otros = resumen.get("Otro", [])
    if len(otros) > 0:
        print(f"\n  Categorias sin palabra clave reconocida (revisar si alguna "
              f"deberia reclasificarse a mano):")
        for o in otros:
            print(f"    - {o}")

    df = df.copy()
    df["Weather_Condition"] = df["Weather_Condition"].map(mapa).astype("category")
    return df


def guardar_resultado(nombre, resultado):
    ruta_npz = os.path.join(OUTPUT_DIR, f"datos_{nombre}.npz")
    np.savez_compressed(
        ruta_npz,
        X_train=resultado["X_train"], y_train=resultado["y_train"],
        X_test=resultado["X_test"], y_test=resultado["y_test"],
    )

    ruta_pkl = os.path.join(OUTPUT_DIR, f"transformador_{nombre}.pkl")
    joblib.dump({
        "preprocesador": resultado["preprocesador"],
        "medianas_grupo": resultado["medianas_grupo"],
        "medianas_globales": resultado["medianas_globales"],
        "limites_iqr": resultado["limites_iqr"],
        "feature_names": resultado["feature_names"],
    }, ruta_pkl)

    ruta_txt = os.path.join(OUTPUT_DIR, f"features_{nombre}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(resultado["feature_names"]))

    print(f"\n  {ruta_npz}")
    print(f"  {ruta_pkl}  (preprocesador + medianas + limites IQR, para produccion)")
    print(f"  {ruta_txt}  (nombres de columnas, para SHAP mas adelante)")


# =============================================================================
# 4. PIPELINE POR ESCENARIO
# =============================================================================

def preparar_escenario(nombre_escenario, df_train_raw, df_test_raw,
                        columnas_numericas_imputar, columnas_numericas_sin_nulos,
                        columnas_categoricas, columnas_booleanas, target):

    titulo(f"ESCENARIO: {nombre_escenario}")
    t0 = time.time()

    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()
    columnas_numericas = columnas_numericas_imputar + columnas_numericas_sin_nulos

    # --- Imputacion numerica --------------------------------------------
    print("\n[1] Imputacion numerica (mediana por State, calculada solo en train)")
    medianas_grupo, medianas_globales = obtener_medianas_grupo(
        df_train, columnas_numericas_imputar, COLUMNA_GRUPO_IMPUTACION
    )
    df_train = imputar_numericas(df_train, columnas_numericas_imputar,
                                  medianas_grupo, medianas_globales, COLUMNA_GRUPO_IMPUTACION)
    df_test = imputar_numericas(df_test, columnas_numericas_imputar,
                                 medianas_grupo, medianas_globales, COLUMNA_GRUPO_IMPUTACION)
    nulos_num_train = int(df_train[columnas_numericas_imputar].isna().sum().sum())
    nulos_num_test = int(df_test[columnas_numericas_imputar].isna().sum().sum())
    print(f"    Nulos numericos restantes -> train: {nulos_num_train}, test: {nulos_num_test}")

    # --- Imputacion categorica -------------------------------------------
    print("\n[2] Imputacion categorica (nueva categoria 'Desconocido')")
    df_train = imputar_categoricas(df_train, columnas_categoricas)
    df_test = imputar_categoricas(df_test, columnas_categoricas)

    # --- Outliers via IQR --------------------------------------------------
    print("\n[3] Deteccion y recorte de outliers (limites IQR calculados solo en train)")
    limites = calcular_limites_iqr(df_train, columnas_numericas_imputar)
    for col, (lo, hi) in limites.items():
        print(f"    {col:<20} limite = [{lo:,.2f}, {hi:,.2f}]")
    df_train = recortar_outliers(df_train, limites)
    df_test = recortar_outliers(df_test, limites)

    # --- Booleanas a entero (0/1) ------------------------------------------
    for col in columnas_booleanas:
        df_train[col] = df_train[col].astype("int8")
        df_test[col] = df_test[col].astype("int8")

    # --- Encoding + Scaling --------------------------------------------
    print("\n[4] Encoding (OneHotEncoder) + Scaling (RobustScaler)")
    columnas_finales_entrada = columnas_numericas + columnas_categoricas + columnas_booleanas

    preprocesador = ColumnTransformer(transformers=[
        ("num", RobustScaler(), columnas_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columnas_categoricas),
        ("bool", "passthrough", columnas_booleanas),
    ])

    X_train = preprocesador.fit_transform(df_train[columnas_finales_entrada])
    X_test = preprocesador.transform(df_test[columnas_finales_entrada])
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()

    nombres_cat = list(
        preprocesador.named_transformers_["cat"].get_feature_names_out(columnas_categoricas)
    )
    nombres_finales = columnas_numericas + nombres_cat + columnas_booleanas

    print(f"    Columnas antes del encoding: {len(columnas_finales_entrada)}")
    print(f"    Columnas despues del encoding (una columna nueva por categoria): {len(nombres_finales)}")
    print(f"    Forma de X_train: {X_train.shape}")
    print(f"    Forma de X_test : {X_test.shape}")

    # --- SMOTE, solo sobre train --------------------------------------------
    print("\n[5] Balanceo con SMOTE (unicamente sobre el conjunto de entrenamiento)")
    print("    Distribucion ANTES de SMOTE:")
    dist_antes = pd.Series(y_train).value_counts().sort_index()
    for clase, n in dist_antes.items():
        print(f"      Severity {clase}: {n:>10,}  ({n / len(y_train) * 100:5.2f}%)")

    smote = SMOTE(random_state=SEMILLA)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("    Distribucion DESPUES de SMOTE:")
    dist_despues = pd.Series(y_train_res).value_counts().sort_index()
    for clase, n in dist_despues.items():
        print(f"      Severity {clase}: {n:>10,}  ({n / len(y_train_res) * 100:5.2f}%)")
    print(f"    Filas de entrenamiento: {len(y_train):,} -> {len(y_train_res):,} tras SMOTE")
    print("    IMPORTANTE: y_test / X_test NO se tocaron. El test sigue")
    print("    reflejando el desbalance real del mundo, como debe ser.")

    print(f"\n  Tiempo del escenario: {time.time() - t0:.1f} s")

    return {
        "X_train": X_train_res.astype("float32"),
        "y_train": y_train_res.astype("int8"),
        "X_test": X_test.astype("float32"),
        "y_test": y_test.astype("int8"),
        "feature_names": nombres_finales,
        "preprocesador": preprocesador,
        "medianas_grupo": medianas_grupo,
        "medianas_globales": medianas_globales,
        "limites_iqr": limites,
    }


# =============================================================================
# 5. FLUJO PRINCIPAL
# =============================================================================

titulo("PASO 1 - CARGA DEL DATASET LIMPIO")
print(f"Archivo: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, usecols=COLUMNAS_A_CARGAR, dtype=dtype_dict)
print(f"Filas cargadas: {len(df):,} | Columnas: {len(df.columns)}")

print()
df = consolidar_weather_condition(df)


titulo("PASO 2 - DIVISION TRAIN / TEST (UNA SOLA VEZ PARA AMBOS ESCENARIOS)")
df_train, df_test = train_test_split(
    df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEMILLA
)
print(f"Train: {len(df_train):,} filas ({(1 - TEST_SIZE) * 100:.0f}%)")
print(f"Test : {len(df_test):,} filas ({TEST_SIZE * 100:.0f}%)")

print("\nVerificacion de que la estratificacion funciono (deben verse iguales):")
print("  Proporcion de Severity en train:")
for clase, p in (df_train[TARGET].value_counts(normalize=True).sort_index() * 100).items():
    print(f"    Severity {clase}: {p:5.2f}%")
print("  Proporcion de Severity en test:")
for clase, p in (df_test[TARGET].value_counts(normalize=True).sort_index() * 100).items():
    print(f"    Severity {clase}: {p:5.2f}%")


# ---- Escenario 1: tiempo real ----------------------------------------------
resultado_tiempo_real = preparar_escenario(
    "TIEMPO REAL (informacion disponible al momento del reporte)",
    df_train, df_test,
    NUM_A_IMPUTAR, NUM_SIN_NULOS, CATEGORICAS, BOOLEANAS, TARGET,
)

# ---- Escenario 2: informacion completa -------------------------------------
resultado_info_completa = preparar_escenario(
    "INFORMACION COMPLETA (modelo de referencia, incluye Distance(mi))",
    df_train, df_test,
    NUM_A_IMPUTAR, NUM_SIN_NULOS + [COL_INFO_COMPLETA], CATEGORICAS, BOOLEANAS, TARGET,
)


titulo("PASO 3 - GUARDADO DE RESULTADOS")
guardar_resultado("tiempo_real", resultado_tiempo_real)
guardar_resultado("informacion_completa", resultado_info_completa)

print("\n" + "=" * 70)
print("PREPARACION COMPLETADA")
print("Ambos escenarios comparten exactamente las mismas filas de train/test")
print("(misma division, semilla = 42); solo cambian las columnas de entrada.")
print("Siguiente paso: Script 3 -> cargar estos .npz y entrenar los 4 modelos.")
print("=" * 70)
