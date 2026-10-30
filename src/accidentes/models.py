"""
MODELOS BASE
=============================================================================
Responsabilidad unica: construir los 4 modelos del proyecto y obtener su
probabilidad de "critico". Nada de integracion ni de metricas aqui.

    Random_Forest         conjunto de arboles que votan (bagging)
    XGBoost               arboles que se corrigen unos a otros (boosting)
    LightGBM              otro boosting, mas rapido (crecimiento por hojas)
    Regresion_Logistica   modelo base lineal: P = sigmoide(theta^T x)

Los 4 se entrenan con el MISMO conjunto balanceado (SMOTENC) y se evaluan
sobre las MISMAS filas reales. Los hiperparametros viven en config.py.
=============================================================================
"""

import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# La regresion logistica puede avisar que no convergio del todo en 300
# iteraciones; para un modelo base eso es aceptable y solo ensucia la consola.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parametros_de_peso(nombre, peso):
    """Peso extra para los casos criticos (la clase 1), segun la API de cada modelo."""
    if nombre in ("XGBoost", "LightGBM"):
        return {"scale_pos_weight": float(peso)}
    return {"class_weight": {0: 1.0, 1: float(peso)}}


def construir_modelo(nombre, hiperparametros, semilla, usar_gpu=False, peso_criticos=None):
    params = dict(hiperparametros[nombre])
    if peso_criticos is not None and peso_criticos != 1:
        params.update(parametros_de_peso(nombre, peso_criticos))
    if nombre == "Random_Forest":
        return RandomForestClassifier(random_state=semilla, **params)
    if nombre == "XGBoost":
        if usar_gpu:
            params["device"] = "cuda"
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=semilla, **params)
    if nombre == "LightGBM":
        return LGBMClassifier(objective="binary", random_state=semilla, **params)
    if nombre == "Regresion_Logistica":
        return LogisticRegression(random_state=semilla, **params)
    raise ValueError(f"Modelo desconocido: {nombre}")


def probabilidad_critico(modelo, X, lote=500_000):
    """P(critico) por lotes, para no duplicar en memoria matrices grandes."""
    salida = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), lote):
        salida[i:i + lote] = modelo.predict_proba(X[i:i + lote])[:, 1]
    return salida
