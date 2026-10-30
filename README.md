# Prediccion del impacto critico de incidentes viales

Pipeline de **modelado** de un sistema de apoyo a la decision para la gestion de incidentes
viales, con el dataset *US Accidents (2016-2023)*. Dado un incidente recien reportado, estima
si tendra un **impacto critico** sobre el trafico usando solo la informacion disponible en ese
momento (lugar, hora, clima, infraestructura cercana).

> Este repositorio contiene **solo** la creacion del modelo (limpieza, preparacion, entrenamiento,
> evaluacion y explicabilidad). **La aplicacion web esta en otro repositorio** y consume los
> artefactos que este deja en `models/` (ver "Relacion con la aplicacion").

## Alcance y decisiones de diseno

| Decision | Detalle |
|---|---|
| Datos | Fuente `Source1`, desde 2021-01-01 (~2.9 M de incidentes). En todo el dataset "Severity" no significa lo mismo para cada proveedor; en este tramo la regla de etiquetado es homogenea. La evidencia la genera el script 01 (`evidencia_fuente_anio.csv`). |
| Objetivo | Binario: **critico** (`Severity >= 3`, en este alcance equivale a Severity 4) frente a normal. Definido antes de evaluar. |
| Particion | **Temporal**: entrenamiento hasta 2022-03-31, validacion (2 mitades) hasta 2022-06-30 y prueba desde 2022-07-01. Nunca se mezclan fechas. |
| Escenarios | `tiempo_real` (opera en la app) e `informacion_completa` (referencia; anade `Distance(mi)`), sobre las mismas filas. |
| Balanceo | **Pesos de clase** (`class_weight="balanced"`: los criticos pesan N_normales/N_criticos). Se probo SMOTENC, pero con LightGBM en validacion empeora el ranking (PR-AUC 0.124 contra ~0.152 de las otras estrategias), asi que se descarto. La estrategia se cambia en `config.py` (`ESTRATEGIA_FINAL`) y la comparacion se reproduce con `--solo-balanceo`. Validacion y prueba conservan siempre la distribucion real. |
| Sin fuga | Todo parametro aprendido (medianas, limites IQR, escaladores, categorias) se calcula **solo con train**. `Anio` no es predictor. |
| Modelos | Random Forest, XGBoost, LightGBM y Regresion Logistica, integrados por **stacking** (comparado con promedio). |
| Descartado | Texto de `Description` (se escribe despues del incidente), cascada de modelos, retroalimentacion y SMOTENC como estrategia final (ver Balanceo). |

## Estructura

```
proyecto-accidentes/
├── README.md
├── requirements.txt
├── pyproject.toml           # hace instalable el paquete `accidentes` (lo usa la app)
├── .gitignore
├── src/accidentes/          # MODULOS reutilizables (cada uno con una sola responsabilidad)
│   ├── config.py            #   rutas, alcance, columnas, escenarios, cortes, balanceo
│   ├── ingestion.py         #   lee el CSV y recorta al alcance
│   ├── cleaning.py          #   limpieza segura (fechas, duplicados, valores imposibles)
│   ├── features.py          #   variables deterministas (tambien las usa la app)
│   ├── splitting.py         #   particion temporal
│   ├── preprocessing.py     #   imputacion, IQR, escalado, codificacion (tambien la app)
│   ├── balancing.py         #   SMOTENC
│   ├── scenarios.py         #   arma y guarda cada escenario
│   ├── models.py            #   los 4 modelos base
│   ├── ensemble.py          #   calibracion, promedio, stacking, umbral, vigilancia
│   ├── evaluation.py        #   metricas
│   ├── training.py          #   flujo de entrenamiento de un escenario
│   ├── tuning.py            #   ajuste de hiperparametros de XGBoost y LightGBM (sin tocar el test)
│   ├── explain.py           #   SHAP aditivo del modelo integrado
│   ├── analysis.py          #   patrones de riesgo y analisis de errores
│   ├── inference.py         #   PredictorIncidentes: punto de entrada de la app (predecir, validar, explicar)
│   ├── storage.py           #   guardar/leer tablas
│   └── utils.py             #   titulos y bitacora de trazabilidad
├── pipeline/                # SCRIPTS que se ejecutan, en orden
│   ├── 01_limpiar_dataset.py
│   ├── 02_preparar_datos.py
│   ├── 03_entrenar_modelos.py
│   ├── 04_explicabilidad_shap.py
│   ├── 05_analisis_errores.py
│   ├── 06_exportar_para_app.py
│   ├── ajustar_hiperparametros.py   #   opcional, entre el 02 y el 03: regulariza XGBoost/LightGBM
│   └── verificar_preparacion.py
├── eda/                     # analisis exploratorio
│   ├── patrones_de_riesgo.py #   preguntas de investigacion 1 a 3 (sin modelo)
│   ├── diagnostico_tipo_via.py #  registro historico: el tipo de via (de Street) mejora la deteccion? (A/B, sin test) -> SI, ya incorporado
│   ├── diagnostico_estados.py #   por que casi no hay alertas en CA/TX/FL? calibracion por estado y reparto de alertas (sin test)
│   ├── ejemplo_modelo_matematico.py #  un incidente real calculado paso a paso (preparacion, Platt, stacking, zona, SHAP) para el documento del modelo matematico
│   └── figures/             #   salidas del EDA y del analisis de errores (se versionan)
├── shap/                    # explicabilidad
│   ├── figures/             #   graficos SHAP (se versionan)
│   └── tablas/              #   tablas SHAP (se versionan)
├── models/                  # artefactos LIVIANOS para la app (preprocesador, umbral, pesos)
├── docs/                    # entregables en .md
├── data/                    # CSV original: NO se sube (ver data/README.md)
└── outputs/                 # matrices y tablas pesadas, regenerables: NO se suben
    ├── limpio/
    ├── preparado/
    └── modelos/
```

## Como ejecutarlo

Desde la **raiz del repositorio**:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Descargar el CSV a data/  (ver data/README.md)
python pipeline/01_limpiar_dataset.py       # limpieza  -> outputs/limpio/
python pipeline/02_preparar_datos.py        # preparacion -> outputs/preparado/
python pipeline/verificar_preparacion.py    # debe terminar en "TODO OK"
python pipeline/03_entrenar_modelos.py --escenario tiempo_real   # 4 modelos + stacking
# (sin --escenario entrena ambos; el completo sirve para medir cuanto se pierde en tiempo real)

python pipeline/04_explicabilidad_shap.py   # SHAP del modelo integrado -> shap/figures y shap/tablas
python pipeline/05_analisis_errores.py      # donde falla el modelo -> eda/figures
python pipeline/06_exportar_para_app.py     # verifica y exporta los artefactos para la app -> models/
python eda/patrones_de_riesgo.py            # preguntas de investigacion 1 a 3 -> eda/figures
python eda/diagnostico_estados.py            # calibracion y reparto de alertas por estado (solo train y validaciones; no reentrena)
python eda/ejemplo_modelo_matematico.py     # un incidente de val_umbral calculado paso a paso + parametros de Platt (no reentrena, no lee test)
# (eda/diagnostico_tipo_via.py ya se corrio: es el registro de por que se incorporo Tipo_via; con los datos nuevos se niega a correr)

# Comparar como manejar el desbalance (LightGBM, en validacion; unos minutos):
python pipeline/03_entrenar_modelos.py --escenario tiempo_real --solo-balanceo
# Entrenar con otra estrategia (SMOTENC | Sin_balancear | Pesos_moderados | Pesos_completos):
python pipeline/03_entrenar_modelos.py --escenario tiempo_real --estrategia Pesos_moderados

# Ajustar hiperparametros de XGBoost y LightGBM (usa train y las 2 validaciones; NUNCA lee el test).
# Reanudable (Ctrl+C muestra el resumen y al repetir el comando continua). No modifica config.py:
python pipeline/ajustar_hiperparametros.py
```

Cada corrida del 03 sobrescribe `outputs/modelos/<escenario>/`; la columna `Estrategia` de
`tabla_comparativa.csv` y el archivo `decision.json` dicen con cual se hizo. Copie la carpeta
si quiere conservar una corrida.

El script 02 tarda ~10 minutos por escenario (SMOTENC sobre ~1.4 M de filas normales) y el 03
varias decenas de minutos por escenario (estimado); ambos necesitan varios GB de RAM libres. Todo es reproducible: la semilla esta en `config.py`.

## Tipo de via (variable derivada de `Street`)

`Street` no sirve como texto (cientos de miles de nombres distintos), pero de ella se deriva `Tipo_via`
con una regla fija (`features.clasificar_via`): `Interestatal`, `Carretera_US`, `Ruta_estatal`, `Autopista`,
`Carretera`, `Local`, `Otro`; un nombre vacio queda como `Desconocido`, igual que el clima. El script 01 la
crea **antes** de quitar duplicados y descarta el texto; al decidir que filas son duplicadas se ignora
`Tipo_via`, asi sobrevive exactamente el mismo conjunto de filas que sin ella.

Por que se incorporo: en `eda/diagnostico_tipo_via.py` (solo train y validaciones; el test no se leyo) el
mismo LightGBM con el tipo de via subio el PR-AUC de 0.153 a 0.190 en `val_umbral` y de 0.136 a 0.168 en
`val_ajuste`, y el recall al 5 % de alertas de 0.351 a 0.415. **Ojo con el sentido del efecto**: las carreteras
US y las "Hwy" concentran MAS criticos que la media; interestatales y autopistas, MENOS. Es una asociacion
observada, no una explicacion causal.

Como la decision se tomo mirando validacion, el test se mide **una sola vez**, con el pipeline final; no se usa
para elegir entre la version con y sin tipo de via. Para el informe conviene conservar la corrida anterior
(`outputs/modelos/tiempo_real_sin_via`).

## Estado

| Paso | Estado |
|---|---|
| 01 Limpieza | listo |
| 02 Preparacion (particion temporal, SMOTENC, escenarios) | listo |
| Verificacion de la preparacion | listo |
| EDA con el nuevo alcance | pendiente |
| 03 Modelos, calibracion, promedio y stacking, umbral, vigilancia | listo; corrido con los datos reales (tiempo_real) |
| Eleccion del balanceo | hecha: pesos de clase |
| SHAP del modelo integrado | listo (probado con datos sinteticos; falta correrlo con los reales) |
| Analisis de errores del modelo y patrones de riesgo (preguntas 1 a 3) | listo (probado con datos sinteticos; falta correrlo con los reales) |
| Exportar artefactos para la app (`06` + `PredictorIncidentes`) | listo (probado con datos sinteticos; falta correrlo con los reales) |
| Tipo de via (`Tipo_via`) incorporado al pipeline | listo en el codigo y probado con datos sinteticos; falta reejecutar 01 -> 06 con los reales |

## Que se versiona y que no

| Si se sube | No se sube |
|---|---|
| Codigo, `docs/`, `eda/figures/`, `shap/` | `data/*` (CSV de ~3 GB) |
| `models/` con archivos **livianos** | `outputs/*` (matrices `.npz`, tablas `.parquet`) |
|  | Modelos pesados (> 100 MB): usar *GitHub Releases* |

## Relacion con la aplicacion web

La app vive en otro repositorio y **no debe reimplementar** las transformaciones: debe usar
las mismas que vio el modelo.

- **Punto de entrada unico:** `accidentes.inference.PredictorIncidentes` (predecir, validar los campos del
  formulario y explicar). La app puede instalar el paquete con `pip install -e ../proyecto-accidentes` (local) o
  `pip install git+https://github.com/<usuario>/proyecto-accidentes.git`.
- **Artefactos que consume:** los que deja `pipeline/06_exportar_para_app.py` en `models/tiempo_real/`
  (preprocesador, 4 modelos, stacking, umbral, explicador) y las tablas de `models/reportes/`. Ver `models/README.md`.
- **Versiones:** los archivos `.joblib` solo cargan bien con las mismas versiones de `pandas`,
  `scikit-learn`, `xgboost` y `lightgbm`; el manifiesto (`models/manifest_tiempo_real.json`) las registra.
- Solo el escenario **`tiempo_real`** opera en la app; `informacion_completa` es la
  referencia con la que se mide cuanto rendimiento se pierde al operar en tiempo real.

## Datos

*US Accidents (2016-2023)*, Kaggle (<https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>).
Ver `data/README.md`.
