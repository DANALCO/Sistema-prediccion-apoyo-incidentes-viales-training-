"""
CONFIGURACION CENTRAL DEL PROYECTO
=============================================================================
Proyecto : Sistema de prediccion en tiempo real y apoyo a la decision para
           la gestion de incidentes viales (US Accidents, Moosavi [23]).

Fuente unica de verdad: rutas, alcance de los datos, listas de columnas,
escenarios, cortes temporales y parametros del balanceo. Si se cambia de
opinion sobre algo, se edita SOLO aqui.

Este modulo no ejecuta nada: solo define constantes.
=============================================================================
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. RUTAS Y GENERALES
# -----------------------------------------------------------------------------
# Todas las rutas parten de la RAIZ del repositorio, sin importar desde que
# carpeta se ejecute un script (src/accidentes/config.py -> raiz = 2 niveles arriba).
RAIZ = Path(__file__).resolve().parents[2]

# El CSV original NO se sube a GitHub (ver data/README.md). Se busca en data/,
# o en la ruta que indique la variable de entorno ACCIDENTES_CSV.
RUTA_CSV_ORIGINAL = Path(os.environ.get("ACCIDENTES_CSV",
                                        RAIZ / "data" / "US_Accidents_March23.csv"))

# Salidas pesadas y regenerables (matrices, tablas): carpeta ignorada por git.
DIR_SALIDA = RAIZ / "outputs"
DIR_LIMPIO = DIR_SALIDA / "limpio"
DIR_PREPARADO = DIR_SALIDA / "preparado"
DIR_MODELOS = DIR_SALIDA / "modelos"                  # modelos completos (pesados)

# Salidas LIVIANAS que SI se versionan (van al informe o las consume la app).
DIR_EDA = RAIZ / "eda" / "figures"
DIR_SHAP_FIGURAS = RAIZ / "shap" / "figures"
DIR_SHAP_TABLAS = RAIZ / "shap" / "tablas"
DIR_MODELOS_APP = RAIZ / "models"                     # preprocesador, umbral, pesos

SEMILLA = 42
ENCODING = "utf-8"                 # explicito, como pide la guia de ETL

# -----------------------------------------------------------------------------
# 2. ALCANCE DE LOS DATOS (justificado con la tabla de evidencia que genera 01)
# -----------------------------------------------------------------------------
# En el dataset completo "Severity" no significa lo mismo para todos los
# proveedores: Source1 marca ~91% de sus incidentes como nivel 2, y
# Source2/3 ~65% como nivel 2 y ~33% como nivel 3. Desde 2021, Source1 solo
# usa los niveles 2 y 4. Se trabaja con esa fuente y ese periodo, donde la
# regla de etiquetado es homogenea. El dataset termina en marzo de 2023.
FUENTE = "Source1"
FECHA_INICIO = "2021-01-01"

MIN_REGISTROS_RUBRICA = 35_000     # criterio "Construccion del Dataset"

# -----------------------------------------------------------------------------
# 3. VARIABLE OBJETIVO (binaria: critico vs normal)
# -----------------------------------------------------------------------------
# Critico = Severity >= 3 (impacto alto). En este alcance equivale a
# Severity 4, porque los niveles 1 y 3 casi no existen. Normal = 1 o 2.
# La regla se fija ANTES de evaluar, como pidio la profesora.
COL_SEVERIDAD = "Severity"
COL_OBJETIVO = "Critico"
SEVERIDAD_MINIMA_CRITICA = 3

# -----------------------------------------------------------------------------
# 4. COLUMNAS DEL CSV ORIGINAL
# -----------------------------------------------------------------------------
# Descartadas tras el EDA (no se cargan):
#   Wind_Chill(F)                      -> Pearson 0.994 con Temperature(F)
#   Civil/Nautical/Astronomical_Twilight -> Cramer's V hasta 0.889 con
#                                          Sunrise_Sunset
# Descartadas por decision del equipo:
#   Description -> texto escrito DESPUES del incidente (riesgo de fuga)
#   ID, Source (metadata), End_Lat/End_Lng, City/County/Zipcode
#   (cardinalidad), Country, Timezone, Airport_Code, Weather_Timestamp
# Street NO se usa como texto (cardinalidad), pero de ella se deriva Tipo_via (interestatal,
# carretera US, ... ver features.clasificar_via): mejoro la deteccion de forma relevante en
# eda/diagnostico_tipo_via.py (PR-AUC en val_umbral 0.153 -> 0.190).
CLIMA_NUMERICO = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
]
COORDENADAS = ["Start_Lat", "Start_Lng"]
INFRAESTRUCTURA = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]
COLS_CATEGORICAS_CRUDAS = ["State", "Wind_Direction", "Weather_Condition", "Sunrise_Sunset", "Tipo_via"]

# Informacion conocida SOLO DESPUES del reporte (modelo de referencia con
# informacion completa; pregunta de investigacion 4).
INFO_POSTERIOR = ["Distance(mi)"]
# Se leen solo para trazabilidad (NO son predictores): ver 01.
COLS_TRAZABILIDAD = ["End_Time"]

COLUMNAS_A_LEER = (
    ["Source", COL_SEVERIDAD, "Start_Time"] + COLS_TRAZABILIDAD
    + ["State"] + COORDENADAS + CLIMA_NUMERICO
    + ["Wind_Direction", "Weather_Condition"]
    + INFRAESTRUCTURA + ["Sunrise_Sunset"] + INFO_POSTERIOR
    + ["Street"]          # solo para derivar Tipo_via (features.py); el texto se descarta en 01
)

# Valores fisicamente imposibles -> NaN (no se borra la fila). Son reglas del
# mundo fisico, no valores calculados con los datos: no causan fuga.
LIMITES_FISICOS = {
    "Humidity(%)":       (0, 100),
    "Visibility(mi)":    (0, 150),
    "Wind_Speed(mph)":   (0, 250),
    "Precipitation(in)": (0, 50),
    "Pressure(in)":      (20, 35),
    "Temperature(F)":    (-60, 135),
    "Distance(mi)":      (0, 500),
}

# -----------------------------------------------------------------------------
# 5. VARIABLES DERIVADAS DE LA FECHA (deterministas: no causan fuga)
# -----------------------------------------------------------------------------
ESTACIONES = {12: "Invierno", 1: "Invierno", 2: "Invierno",
              3: "Primavera", 4: "Primavera", 5: "Primavera",
              6: "Verano", 7: "Verano", 8: "Verano",
              9: "Otonio", 10: "Otonio", 11: "Otonio"}
HORAS_PICO = [(7, 9), (16, 19)]          # solo cuentan en dia habil

# Weather_Condition: de ~96 categorias a ~12 por palabras clave (regla de
# texto fija, en orden de prioridad).
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

# -----------------------------------------------------------------------------
# 6. COLUMNAS DE MODELADO Y ESCENARIOS
# -----------------------------------------------------------------------------
# 'Anio' NO es predictor: refleja cambios en como se etiquetaba la
# severidad, no un patron de los accidentes (ver evidencia de 01). Tampoco
# lo son End_Time ni Duracion_min: solo se conocen cuando el incidente TERMINA
# (informacion posterior al reporte). En este alcance Duracion_min casi no se
# relaciona con Critico (correlacion ~0.04), asi que excluirla no cuesta nada.
DISCRETAS = ["Hora", "Dia_semana", "Mes"]
CATEGORICAS = ["State", "Wind_Direction", "Weather_Condition", "Estacion", "Sunrise_Sunset", "Tipo_via"]
BOOLEANAS = INFRAESTRUCTURA + ["Es_fin_de_semana", "Es_hora_pico"]

COLUMNA_GRUPO_IMPUTACION = "State"
VALOR_DESCONOCIDO = "Desconocido"
FACTOR_IQR = 1.5

# Cada escenario es una lista de columnas sobre las MISMAS filas.
#   continuas  : numericas que se escalan (y que SMOTENC interpola)
#   a_imputar  : numericas con nulos (mediana por estado, solo train)
#   a_recortar : numericas a las que se aplica el recorte IQR
ESCENARIOS = {
    "tiempo_real": {
        "continuas": COORDENADAS + CLIMA_NUMERICO,
        "a_imputar": CLIMA_NUMERICO,
        "a_recortar": CLIMA_NUMERICO,
    },
    "informacion_completa": {
        "continuas": COORDENADAS + CLIMA_NUMERICO + INFO_POSTERIOR,
        "a_imputar": CLIMA_NUMERICO + INFO_POSTERIOR,
        "a_recortar": CLIMA_NUMERICO,       # Distance(mi) no se recorta
    },
}

# -----------------------------------------------------------------------------
# 7. PARTICION TEMPORAL (nunca se mezclan fechas)
# -----------------------------------------------------------------------------
#   train      : desde FECHA_INICIO hasta CORTE_TRAIN (exclusivo)
#   val_ajuste : [CORTE_TRAIN, CORTE_VAL_AJUSTE)  -> calibracion y combinador
#   val_umbral : [CORTE_VAL_AJUSTE, CORTE_TEST)   -> umbral de decision
#   test       : desde CORTE_TEST (se usa UNA sola vez, al final)
CORTE_TRAIN = "2022-04-01"
CORTE_VAL_AJUSTE = "2022-05-16"
CORTE_TEST = "2022-07-01"

# -----------------------------------------------------------------------------
# 8. BALANCEO (SMOTENC, solo sobre train)
# -----------------------------------------------------------------------------
# SMOTENC = SMOTE para datos mixtos (continuos + categoricos). Interpola las
# variables continuas y asigna a las categoricas el valor mas frecuente entre
# los vecinos, asi no aparecen valores como 0.4 en una columna de si/no.
RATIO_SMOTE = 1.0                  # minoritaria / mayoritaria tras SMOTENC (1.0 = 50/50)
K_VECINOS_SMOTE = 5
LOTE_SMOTE = 300_000               # sinteticas por llamada (controla la memoria)
MAX_FILAS_NORMALES_ENTRENAMIENTO = None   # tope opcional de filas normales; None = todas
GUARDAR_TRAIN_REAL = True          # guarda tambien el train sin balancear (para comparar)

# -----------------------------------------------------------------------------
# 9. ENTRENAMIENTO E INTEGRACION (script 03)
# -----------------------------------------------------------------------------
# Hiperparametros FIJOS (no se ajustan mirando el test). Son valores iniciales
# razonables, no optimizados. Random Forest va con profundidad y muestra
# limitadas: con 2.8 M de filas un bosque sin limites no cabe en memoria, y
# ademas queda mas facil de explicar con SHAP.
NOMBRES_MODELOS = ("Random_Forest", "XGBoost", "LightGBM", "Regresion_Logistica")
HIPERPARAMETROS = {
    "Random_Forest": dict(n_estimators=150, max_depth=14, min_samples_leaf=100,
                          max_samples=0.2, max_features="sqrt", n_jobs=-1),
    "XGBoost": dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=10, tree_method="hist", n_jobs=-1),
    "LightGBM": dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                     min_child_samples=100, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, n_jobs=-1, verbose=-1),
    "Regresion_Logistica": dict(C=1.0, max_iter=300),
}
XGBOOST_USAR_GPU = False           # True si CUDA esta configurado (GTX 1650)

# Metodos que se comparan y modelo FINAL (elegido de antemano, no mirando el test)
METODOS = NOMBRES_MODELOS + ("Promedio", "Promedio_ponderado", "Stacking")
METODO_FINAL = "Stacking"

# Regla del umbral: en val_umbral, el MAYOR recall cuya exactitud siga siendo
# >= este valor. El 0.94 deja ~2 puntos de margen sobre el nivel de 92% de la
# rubrica, porque la exactitud suele bajar un poco de validacion a prueba.
EXACTITUD_MINIMA_VALIDACION = 0.94
NIVELES_RUBRICA = ((0.95, 50), (0.92, 30), (0.89, 20))   # (exactitud minima, puntos)

# Franja de VIGILANCIA a ambos lados del umbral de decision (tau):
#   borde inferior = puntaje que marcaria 1.5 veces los incidentes que marca tau
#   borde superior = puntaje que marcaria 2/3 de los incidentes que marca tau
# La decision que se evalua es siempre "puntaje >= tau"; la vigilancia es solo
# una marca de caso limite para la interfaz.
FACTOR_VIGILANCIA_INFERIOR = 1.5
FACTOR_VIGILANCIA_SUPERIOR = 2 / 3

# Manejo del desbalance al ENTRENAR los modelos (se cambia sin repetir el script 02):
#   SMOTENC          train balanceado 50/50 con SMOTENC (lo genera el script 02)
#   Sin_balancear    train real (2.5% de criticos), sin ningun ajuste
#   Pesos_moderados  train real + los criticos pesan raiz(N_normales / N_criticos) (~6)
#   Pesos_completos  train real + los criticos pesan N_normales / N_criticos (~38)
# Los pesos de clase los sugirio la profesora ("dar mayor importancia a los casos
# criticos durante el entrenamiento"). Validacion y test NUNCA se tocan.
ESTRATEGIAS_BALANCEO = ("SMOTENC", "Sin_balancear", "Pesos_moderados", "Pesos_completos")
# Elegida comparando las 4 estrategias con LightGBM en val_umbral (tiempo_real):
#   SMOTENC PR-AUC 0.124 | Sin_balancear 0.154 | Pesos_moderados 0.151 | Pesos_completos 0.152
# SMOTENC empeora el ranking; las otras tres son equivalentes. Se usa Pesos_completos
# (= class_weight="balanced"): trata el desbalance de forma explicita, como sugirio la
# profesora, sin datos sinteticos.
ESTRATEGIA_FINAL = "Pesos_completos"
COMPARAR_BALANCEO = True           # LightGBM con cada estrategia, medido en val_umbral

# -----------------------------------------------------------------------------
# 10. EXPLICABILIDAD (script 04)
# -----------------------------------------------------------------------------
# Grupos tematicos de variables: sirven para responder las preguntas de
# investigacion 1 a 3 (clima, patrones temporales, infraestructura) y para el
# grafico de importancia por grupo.
GRUPOS_TEMATICOS = {
    "Clima": CLIMA_NUMERICO + ["Wind_Direction", "Weather_Condition"],
    "Tiempo": DISCRETAS + ["Es_fin_de_semana", "Es_hora_pico", "Estacion", "Sunrise_Sunset"],
    "Infraestructura": INFRAESTRUCTURA + ["Tipo_via"],
    "Ubicacion": COORDENADAS + ["State"],
    "Informacion posterior": INFO_POSTERIOR,     # solo existe en el escenario completo
}
SHAP_N_REAL = 2000        # incidentes al azar del test (distribucion real) para la importancia global
SHAP_N_CRITICOS = 1000    # incidentes criticos del test, para ver que empuja hacia "critico"

# -----------------------------------------------------------------------------
# 11. PREGUNTAS DE INVESTIGACION 1 A 3 (eda/patrones_de_riesgo.py)
# -----------------------------------------------------------------------------
PREGUNTAS_INVESTIGACION = {
    1: ("Clima y visibilidad", ["Weather_Condition", "Visibility(mi)", "Precipitation(in)",
                                "Wind_Speed(mph)", "Temperature(F)", "Humidity(%)"]),
    2: ("Patrones temporales", ["Hora", "Dia_semana", "Mes", "Estacion", "Es_hora_pico",
                                "Es_fin_de_semana", "Sunrise_Sunset"]),
    3: ("Infraestructura", INFRAESTRUCTURA + ["Tipo_via"]),
}
