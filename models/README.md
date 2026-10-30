# models/

Artefactos **del modelo final** que consume la aplicacion web (que vive en otro repositorio).
Los genera `pipeline/06_exportar_para_app.py`; no se editan a mano.

```
models/
├── tiempo_real/                  # el escenario que opera en la app
│   ├── preprocesador.joblib      #   imputacion, IQR, escalado, codificacion (aprendidos solo con train)
│   ├── modelo_<nombre>.joblib    #   Random Forest, XGBoost, LightGBM, Regresion Logistica
│   ├── apilador.joblib           #   stacking: pesos de cada modelo
│   ├── decision.json             #   umbral de decision, franja de vigilancia, coeficientes
│   ├── explicador_estado.json    #   lo que necesita la explicacion SHAP
│   └── features.txt              #   nombres de las columnas finales
├── reportes/                     # tablas pequenas para los paneles de la app
│   └── curva_umbral_<escenario>.csv   #   alertas, exactitud, recall... en cada umbral (control de umbral)
├── manifest_tiempo_real.json     # versiones de las librerias, tamanos, SHA-256, resultado de la verificacion
└── README.md
```

## Como usarlos desde la app

```python
from accidentes.inference import PredictorIncidentes   # pip install -e ../proyecto-accidentes

predictor = PredictorIncidentes.cargar("models", "tiempo_real")
predictor.opciones_formulario()                  # valores validos de State, Weather_Condition, Tipo_via, ... (desplegables)
resultado = predictor.predecir(incidentes)       # riesgo, nivel de alerta, aporte de cada modelo
explicacion, base = predictor.explicar(incidentes)   # aporte de cada variable (SHAP)
resultado = predictor.predecir(incidentes, umbral=0.09)   # control de umbral (ver abajo)
```

**No reimplemente** ninguna transformacion: `PredictorIncidentes` usa el mismo codigo, el mismo
stacking y el mismo umbral con los que se evaluo el modelo.

### Campos del formulario
- **Obligatorios:** `Start_Time`, `State`, `Start_Lat`, `Start_Lng`. Si faltan, lanza `ValueError`.
- **Opcionales:** clima (`Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`,
  `Wind_Speed(mph)`, `Precipitation(in)`, `Wind_Direction`, `Weather_Condition`), `Sunrise_Sunset`, las 13
  columnas de infraestructura y `Tipo_via`. Si faltan se estiman con la mediana de entrenamiento (las categoricas
  pasan a `Desconocido`) y la columna `Campos_estimados` lo avisa (prediccion menos fiable).
- **Tipo de via:** dos formas de enviarlo. (a) `Tipo_via` con uno de los valores de
  `opciones_formulario()["Tipo_via"]` (`Interestatal`, `Carretera_US`, `Ruta_estatal`, `Autopista`, `Carretera`,
  `Local`, `Otro`, `Desconocido`); un valor que no este en la lista lanza `ValueError`. (b) `Street`, el nombre de
  la calle en texto libre (`"I-95 N"`, `"W Main St"`): el predictor lo clasifica con la **misma regla** del
  entrenamiento, asi la app no reimplementa nada. Si llegan las dos, manda `Tipo_via`; la calle solo lo completa cuando esta vacio.
- Las casillas de infraestructura deben enviarse **explicitas** (`True`/`False`); si se omiten, cuentan
  como campos estimados.
- Valores fisicamente imposibles (por ejemplo humedad > 100) lanzan `ValueError`.

### Niveles
`Es_critico` es la decision que se evalua (`Probabilidad >= umbral`). `Nivel_simple` es solo para
mostrar: **Normal**, **Vigilancia** (casos en el limite, a ambos lados del umbral) o **Critico**.

### Control de umbral
`predecir(incidentes, umbral=X)` (`0 < X < 1`) cambia **solo la decision y el nivel**: la probabilidad y los
aportes de cada modelo son los mismos. La franja de vigilancia se reescala en la misma proporcion para que
siga rodeando al umbral. Sin `umbral` se usa el oficial de `decision.json` (el que se evaluo en test).

Para que el control muestre **que cuesta cada umbral** sin recalcular nada, use
`reportes/curva_umbral_tiempo_real.csv`: una fila por umbral y por particion (`val_umbral` y `test`).

| Columna | Que es |
|---|---|
| `Umbral`, `Alertas_%` | el umbral y el % de incidentes que marca |
| `Exactitud`, `Baseline` | exactitud de la rubrica y la de "todo normal" (siempre se muestran juntas) |
| `Recall`, `Precision`, `F1`, `Balanced_acc` | calidad de la deteccion |
| `Criticos_detectados_x1000`, `Falsas_alarmas_x1000`, `Criticos_perdidos_x1000` | por cada 1,000 incidentes: la forma mas facil de explicar el costo de mover el umbral |
| `Es_umbral_oficial` | la fila del umbral por defecto (una por particion) |
| `Puntos_solo_exactitud` | puntos de la rubrica por la exactitud sola (las funcionalidades se exigen aparte) |

Use la particion `test` para mostrar al usuario: es la que simula datos futuros. Subir el umbral marca menos
incidentes y sube la exactitud, pero detecta menos criticos; bajarlo hace lo contrario. Las cifras son de
una evaluacion historica: orientativas, no una garantia para incidentes futuros (la tasa de criticos cambia
con el tiempo). `06_exportar_para_app.py` comprueba que el punto oficial de la curva coincide con la
tabla del script 03 antes de publicarla.

## Versiones y archivos pesados
- Los `.joblib` solo cargan bien con las **mismas versiones** de `pandas`, `scikit-learn`, `xgboost` y
  `lightgbm` que figuran en el manifiesto. Use esas versiones en el repositorio de la app.
- GitHub rechaza archivos de mas de 100 MB. Lo que supere `--max-mb` (50 por defecto) queda fuera de
  `models/`; publiquelo como *Release* y descarguelo en `models/tiempo_real/` antes de cargar el predictor.
  El manifiesto marca esos archivos con `"ubicacion": "release"` y trae su SHA-256 para comprobar la descarga.

## Como actualizarlos
`03_entrenar_modelos.py` -> `04_explicabilidad_shap.py` -> `06_exportar_para_app.py`. Si cambia la lista de
variables (como al agregar `Tipo_via`) hay que repetir desde `01_limpiar_dataset.py`: los `.joblib` viejos no
sirven con el formulario nuevo. Si la verificacion
de extremo a extremo falla, el script no publica nada.
