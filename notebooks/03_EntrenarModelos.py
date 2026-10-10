"""
=============================================================================
ENTRENAMIENTO Y EVALUACION DE LOS 4 MODELOS
=============================================================================
Proyecto : Sistema de prediccion en tiempo real y apoyo a la decision
           para la gestion de incidentes viales
Entrada  : los .npz generados por el script de preparacion
           (output/preparado/datos_tiempo_real.npz y datos_informacion_completa.npz)
Salida   : modelos entrenados (.pkl), tabla comparativa (.csv) y matrices
           de confusion (.png) en output/modelos/

-----------------------------------------------------------------------------
UBICACION EN EL PIPELINE
-----------------------------------------------------------------------------
  Script limpieza -> Script preparacion -> ESTE SCRIPT 

  Este script NO va a tocar el CSV ni repite imputacion/IQR/encoding/
  SMOTE: todo eso ya quedo resuelto y guardado en los .npz. Aqui solo se
  carga, se entrena y se evalua. Si mas adelante quieren ajustar un
  hiperparametro o agregar una metrica, no hace falta re-correr todo el
  pipeline de preparacion, solo este script.

-----------------------------------------------------------------------------
DOS DETALLES TECNICOS IMPORTANTES
-----------------------------------------------------------------------------
  1. Clases 1-4 vs 0-3: Random Forest y Regresion Logistica de scikit-learn
     aceptan las clases de Severity tal cual (1, 2, 3, 4). XGBoost y
     LightGBM esperan que las clases empiecen en 0. Por eso, SOLO para
     esos dos modelos, se les resta 1 a las etiquetas antes de entrenar y
     se les suma 1 de vuelta a las predicciones antes de evaluar. Todas
     las metricas finales se calculan siempre en el espacio original (1-4).

  2. class_weight NO se usa en ningun modelo: el X_train que se carga ya
     paso por SMOTE en el script de preparacion (las 4 clases estan al
     25% cada una). Combinar SMOTE con class_weight='balanced' corregiria
     el desbalance dos veces, lo cual puede sobre-corregir y perjudicar
     el desempeno en la clase mayoritaria real.

-----------------------------------------------------------------------------
POR QUE F1-MACRO Y AUC-MACRO, NO SOLO ACCURACY
-----------------------------------------------------------------------------
  Con un desbalance de ~92 a 1 (visto en el EDA), un modelo que siempre
  prediga Severity 2 tendria ~79% de exactitud sin haber aprendido nada.
  F1-macro calcula el F1 de cada clase por separado y promedia esos 4
  numeros dandoles el mismo peso, sin importar cuantos ejemplos tenga
  cada una. AUC-macro (one-vs-rest) complementa esto mirando que tan bien
  ordena el modelo sus probabilidades, no solo su decision final.

-----------------------------------------------------------------------------
REQUISITOS
-----------------------------------------------------------------------------
  pip install xgboost lightgbm scikit-learn joblib matplotlib seaborn
=============================================================================
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, accuracy_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# 1. CONFIGURACION
# =============================================================================

# Las rutas son relativas a la ubicacion de este script (carpeta notebooks/),
# no a la carpeta desde donde se ejecute. Lee lo que genero el Script 2 en
# notebooks/output/preparado/ y guarda sus resultados en notebooks/output/modelos/.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output" / "preparado"
OUTPUT_DIR = BASE_DIR / "output" / "modelos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"No se encontro la carpeta de datos preparados: {INPUT_DIR}\n"
        "Ejecuta primero 02_preparar_datos.py para generarla."
    )

SEMILLA = 42
ESCENARIOS = ["tiempo_real", "informacion_completa"]
CLASES_ORIGINALES = [1, 2, 3, 4]
NOMBRES_CLASES = [f"Severity {c}" for c in CLASES_ORIGINALES]

# Prueba rapida antes de comprometerse a la corrida completa: con
# MODO_RAPIDO=True cada modelo usa muchos menos arboles/iteraciones, para
# confirmar en pocos minutos que todo el script corre sin errores antes de
# lanzar la version final (CORRALO EN RAPIDO PRIMERO PARA VER SI ESTA BIEN).
MODO_RAPIDO = False
N_ESTIMATORS = 50 if MODO_RAPIDO else 300
MAX_ITER_LOGISTICA = 200 if MODO_RAPIDO else 1000


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# =============================================================================
# 2. DEFINICION DE LOS 4 MODELOS
# =============================================================================

def construir_modelos():
    """Instancias NUEVAS cada vez que se llama, para no reutilizar un
    modelo ya entrenado de un escenario en el otro."""
    return {
        "Random_Forest": {
            "modelo": RandomForestClassifier(
                n_estimators=N_ESTIMATORS, random_state=SEMILLA, n_jobs=-1,
            ),
            "necesita_offset": False,
        },
        "XGBoost": {
            "modelo": XGBClassifier(
                objective="multi:softprob", num_class=4,
                n_estimators=N_ESTIMATORS, random_state=SEMILLA, n_jobs=-1,
                eval_metric="mlogloss",
            ),
            "necesita_offset": True,
        },
        "LightGBM": {
            "modelo": LGBMClassifier(
                objective="multiclass", num_class=4,
                n_estimators=N_ESTIMATORS, random_state=SEMILLA, n_jobs=-1,
                verbose=-1,
            ),
            "necesita_offset": True,
        },
        "Regresion_Logistica": {
            # sklearn moderno usa perdida multinomial automaticamente con
            # mas de 2 clases y el solver por defecto (lbfgs); no hace
            # falta el parametro multi_class (esta deprecado).
            "modelo": LogisticRegression(
                max_iter=MAX_ITER_LOGISTICA, random_state=SEMILLA,
            ),
            "necesita_offset": False,
        },
    }


# =============================================================================
# 3. ENTRENAMIENTO Y EVALUACION DE UNA COMBINACION (modelo, escenario)
# =============================================================================

def entrenar_y_evaluar(nombre_modelo, config, X_train, y_train, X_test, y_test, escenario):
    print(f"\n--- {nombre_modelo} | {escenario} ---")
    t0 = time.time()

    modelo = config["modelo"]

    if config["necesita_offset"]:
        print("  Ajustando temporalmente Severity 1-4 -> 0-3 (lo exige este "
              "modelo); se revierte antes de evaluar.")
        y_train_ajustado = y_train - 1
    else:
        y_train_ajustado = y_train

    modelo.fit(X_train, y_train_ajustado)

    y_pred = modelo.predict(X_test)
    if config["necesita_offset"]:
        y_pred = y_pred + 1
    y_proba = modelo.predict_proba(X_test)  # columnas ya en orden ascendente 1,2,3,4

    # ---- Metricas resumen ----
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr",
                               average="macro", labels=CLASES_ORIGINALES)

    tiempo = time.time() - t0
    print(f"  F1-macro:  {f1_macro:.4f}")
    print(f"  AUC-macro: {auc_macro:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}  (referencia; no usar sola por el desbalance)")
    print(f"  Tiempo:    {tiempo:.1f} s")

    # ---- Reporte por clase ----
    reporte = classification_report(
        y_test, y_pred, labels=CLASES_ORIGINALES, target_names=NOMBRES_CLASES,
        output_dict=True, zero_division=0,
    )
    for c, nombre in zip(CLASES_ORIGINALES, NOMBRES_CLASES):
        print(f"    {nombre}: precision={reporte[nombre]['precision']:.3f}  "
              f"recall={reporte[nombre]['recall']:.3f}  "
              f"f1={reporte[nombre]['f1-score']:.3f}")

    # ---- Matriz de confusion ----
    matriz = confusion_matrix(y_test, y_pred, labels=CLASES_ORIGINALES)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=NOMBRES_CLASES, yticklabels=NOMBRES_CLASES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"{nombre_modelo} - {escenario}")
    plt.tight_layout()
    ruta_matriz = os.path.join(OUTPUT_DIR, f"matriz_confusion_{nombre_modelo}_{escenario}.png")
    plt.savefig(ruta_matriz, dpi=110)
    plt.close(fig)

    # ---- Guardar el modelo entrenado (lo necesitara el script de SHAP) ----
    ruta_modelo = os.path.join(OUTPUT_DIR, f"modelo_{nombre_modelo}_{escenario}.pkl")
    joblib.dump(modelo, ruta_modelo)

    fila = {
        "Modelo": nombre_modelo,
        "Escenario": escenario,
        "F1_macro": round(f1_macro, 4),
        "AUC_macro": round(auc_macro, 4),
        "Precision_macro": round(precision_macro, 4),
        "Recall_macro": round(recall_macro, 4),
        "Accuracy": round(accuracy, 4),
        "Tiempo_segundos": round(tiempo, 1),
    }
    for c, nombre in zip(CLASES_ORIGINALES, NOMBRES_CLASES):
        fila[f"F1_Severity_{c}"] = round(reporte[nombre]["f1-score"], 4)
        fila[f"Recall_Severity_{c}"] = round(reporte[nombre]["recall"], 4)

    return fila


# =============================================================================
# 4. CARGA DE DATOS PREPARADOS
# =============================================================================

titulo("PASO 1 - CARGA DE DATOS PREPARADOS")
if MODO_RAPIDO:
    print("MODO_RAPIDO = True: usando pocos arboles/iteraciones solo para "
          "verificar que el script corre sin errores. Poner en False para "
          "la corrida final que va al informe.\n")

datos = {}
for escenario in ESCENARIOS:
    ruta = os.path.join(INPUT_DIR, f"datos_{escenario}.npz")
    with np.load(ruta) as npz:
        datos[escenario] = {
            "X_train": npz["X_train"], "y_train": npz["y_train"],
            "X_test": npz["X_test"], "y_test": npz["y_test"],
        }
    d = datos[escenario]
    print(f"{escenario}: X_train {d['X_train'].shape}, X_test {d['X_test'].shape}")


# =============================================================================
# 5. ENTRENAMIENTO DE LOS 4 MODELOS EN LOS 2 ESCENARIOS
# =============================================================================

titulo("PASO 2 - ENTRENAMIENTO (4 modelos x 2 escenarios = 8 corridas)")
tiempo_inicio_total = time.time()

resultados = []
for escenario in ESCENARIOS:
    d = datos[escenario]
    modelos = construir_modelos()
    for nombre_modelo, config in modelos.items():
        fila = entrenar_y_evaluar(
            nombre_modelo, config,
            d["X_train"], d["y_train"], d["X_test"], d["y_test"],
            escenario,
        )
        resultados.append(fila)

print(f"\nTiempo total de entrenamiento: {(time.time() - tiempo_inicio_total) / 60:.1f} min")

df_resultados = pd.DataFrame(resultados)
ruta_tabla = os.path.join(OUTPUT_DIR, "tabla_comparativa_modelos.csv")
df_resultados.to_csv(ruta_tabla, index=False)


# =============================================================================
# 6. TABLA COMPARATIVA
# =============================================================================

titulo("PASO 3 - TABLA COMPARATIVA (las 8 combinaciones)")
columnas_resumen = ["Modelo", "Escenario", "F1_macro", "AUC_macro", "Accuracy", "Tiempo_segundos"]
print(df_resultados[columnas_resumen].to_string(index=False))
print(f"\nTabla completa guardada en: {ruta_tabla}")


# =============================================================================
# 7. PERDIDA DE DESEMPENIO AL OPERAR EN TIEMPO REAL (Pregunta de investigacion 4)
# =============================================================================

titulo("PASO 4 - PERDIDA DE DESEMPENIO: INFORMACION COMPLETA vs TIEMPO REAL")
print("Esta es la cifra central de la pregunta de investigacion 4: cuanto")
print("rendimiento pierde cada modelo por operar solo con informacion")
print("disponible al momento del reporte, en vez de con el registro completo.\n")

tabla_perdida = pd.DataFrame(index=df_resultados["Modelo"].unique())
for metrica in ["F1_macro", "AUC_macro"]:
    pivote = df_resultados.pivot(index="Modelo", columns="Escenario", values=metrica)
    tabla_perdida[f"{metrica}_tiempo_real"] = pivote["tiempo_real"]
    tabla_perdida[f"{metrica}_info_completa"] = pivote["informacion_completa"]
    tabla_perdida[f"{metrica}_perdida"] = pivote["informacion_completa"] - pivote["tiempo_real"]
    tabla_perdida[f"{metrica}_perdida_%"] = (
        tabla_perdida[f"{metrica}_perdida"] / pivote["informacion_completa"] * 100
    )

print(tabla_perdida.round(4).to_string())
ruta_perdida = os.path.join(OUTPUT_DIR, "perdida_tiempo_real.csv")
tabla_perdida.round(4).to_csv(ruta_perdida)
print(f"\nTabla guardada en: {ruta_perdida}")

print("\nUna perdida de F1-macro pequenia sugiere que el sistema en tiempo")
print("real es casi tan bueno como el de referencia (buena noticia para el")
print("caso de uso real). Una perdida grande sugiere que la informacion")
print("posterior al reporte (Distance(mi)) aporta bastante, y que hay un")
print("costo real por tener que decidir con informacion parcial.")


# =============================================================================
# 8. CIERRE
# =============================================================================

titulo("PASO 5 - RESUMEN FINAL")
print(f"Modelos entrenados guardados en: {OUTPUT_DIR}/modelo_<nombre>_<escenario>.pkl")
print(f"Matrices de confusion guardadas en: {OUTPUT_DIR}/matriz_confusion_*.png")
print(f"Tabla comparativa: {ruta_tabla}")
print(f"Tabla de perdida por tiempo real: {ruta_perdida}")
print("\nEstos 8 modelos guardados son la entrada del siguiente script (SHAP")
print("+ tablero interactivo): no hace falta reentrenar nada, solo cargarlos.")
print("=" * 70)
