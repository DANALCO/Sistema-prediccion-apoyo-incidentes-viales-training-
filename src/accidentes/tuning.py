"""
AJUSTE DE HIPERPARAMETROS (sin tocar el test)
=============================================================================
Responsabilidad unica: probar configuraciones de XGBoost y LightGBM y medir si
detectan mejor que la configuracion actual de config.py. No guarda modelos ni
decide nada por si mismo: devuelve numeros para que el equipo decida.

REGLA DE ORO (la misma del resto del proyecto)
    train        entrena cada candidato
    val_ajuste   ELIGE: parada temprana (cuantos arboles) y ranking de candidatos
    val_umbral   CONFIRMA: el mejor candidato debe ganar tambien aqui, una
                 particion que NO se uso para elegir
    test         NO se lee. El script que usa este modulo ni siquiera la carga.

POR QUE ESTOS CAMBIOS (el sintoma: PR-AUC 0.231 en train y 0.139 en test)
    El modelo memoriza el entrenamiento. Se prueban arboles mas cortos, hojas
    con mas ejemplos minimos, submuestreo de filas y columnas, penalizacion L2 y
    PARADA TEMPRANA: se deja de agregar arboles cuando val_ajuste deja de mejorar.
=============================================================================
"""

import time
import warnings

import numpy as np
from lightgbm import early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

from accidentes import config as cfg
from accidentes.models import construir_modelo, probabilidad_critico

# LightGBM 4.7 marca `eval_set` como obsoleto (sugiere eval_X/eval_y), pero `eval_set`
# funciona en todas las versiones; se silencia solo ese aviso para no ensuciar la consola.
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")

MODELOS_AJUSTABLES = ("XGBoost", "LightGBM")

# Valores a sortear. Todos regularizan mas que la configuracion actual (o la igualan).
ESPACIOS = {
    "XGBoost": dict(
        max_depth=[3, 4, 5, 6, 8],
        learning_rate=[0.05, 0.1],
        min_child_weight=[10, 30, 100, 300],
        subsample=[0.5, 0.7, 0.8],
        colsample_bytree=[0.5, 0.7, 0.8],
        reg_lambda=[1, 10, 50],
        gamma=[0, 1, 5],
    ),
    "LightGBM": dict(
        num_leaves=[7, 15, 31, 63],
        learning_rate=[0.03, 0.05, 0.1],
        min_child_samples=[100, 300, 1000, 3000],
        subsample=[0.5, 0.7, 0.8],
        colsample_bytree=[0.5, 0.7, 0.8],
        reg_lambda=[0, 10, 50],
    ),
}
# Primer candidato de cada modelo: una version claramente mas regularizada. Va primero
# para que, aunque se interrumpa el ajuste pronto, ya haya un candidato razonable.
CONSERVADORAS = {
    "XGBoost": dict(max_depth=4, learning_rate=0.1, min_child_weight=100, subsample=0.7,
                    colsample_bytree=0.7, reg_lambda=10, gamma=0),
    "LightGBM": dict(num_leaves=15, learning_rate=0.05, min_child_samples=1000, subsample=0.7,
                     colsample_bytree=0.7, reg_lambda=10),
}
MAX_ARBOLES = {"XGBoost": 1000, "LightGBM": 1500}   # tope de la parada temprana
PACIENCIA = 50                                       # arboles sin mejora antes de parar


# ----------------------------------------------------------------------------
def cargar_sin_test(escenario, estrategia):
    """Carga SOLO train y las dos validaciones. Las claves de test no se leen nunca
    (ni las matrices ni las etiquetas ni las posiciones)."""
    ruta = cfg.DIR_PREPARADO / f"datos_{escenario}.npz"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute antes pipeline/02_preparar_datos.py")
    clave = "train_bal" if estrategia == "SMOTENC" else "train_real"
    with np.load(ruta) as z:
        if f"X_{clave}" not in z.files:
            raise ValueError(f"{ruta.name} no trae {clave} (revise GUARDAR_TRAIN_REAL en config.py)")
        datos = {clave: (z[f"X_{clave}"], z[f"y_{clave}"])}
        for particion in ("val_ajuste", "val_umbral"):
            datos[particion] = (z[f"X_{particion}"], z[f"y_{particion}"])
    return datos


# ----------------------------------------------------------------------------
def configuraciones(nombre, n, semilla):
    """Lista [(id, parametros)] de n candidatos, sin repetidos. Determinista: con la
    misma semilla, pedir mas candidatos conserva los anteriores (permite reanudar)."""
    espacio = ESPACIOS[nombre]
    rng = np.random.default_rng(semilla)
    lista = [("conservadora", dict(CONSERVADORAS[nombre]))]
    vistos = {tuple(sorted(lista[0][1].items()))}
    intentos = 0
    while len(lista) < n and intentos < 10_000:
        intentos += 1
        params = {k: valores[int(rng.integers(len(valores)))] for k, valores in espacio.items()}
        clave = tuple(sorted(params.items()))
        if clave in vistos:
            continue
        vistos.add(clave)
        lista.append((f"c{len(lista):02d}", params))
    return lista[:n]


def entrenar_actual(nombre, X, y, peso, semilla):
    """La configuracion ACTUAL de config.py, tal cual la usa el script 03 (sin parada temprana)."""
    t0 = time.time()
    modelo = construir_modelo(nombre, cfg.HIPERPARAMETROS, semilla, cfg.XGBOOST_USAR_GPU, peso_criticos=peso)
    modelo.fit(X, y)
    return modelo, cfg.HIPERPARAMETROS[nombre]["n_estimators"], False, time.time() - t0


def entrenar_con_parada(nombre, params, X, y, X_val, y_val, peso, semilla):
    """Entrena un candidato y se detiene cuando el PR-AUC en val_ajuste deja de mejorar.
    Devuelve (modelo, arboles_usados, llego_al_tope, segundos)."""
    t0 = time.time()
    hp = {nombre: {**cfg.HIPERPARAMETROS[nombre], **params, "n_estimators": MAX_ARBOLES[nombre]}}
    modelo = construir_modelo(nombre, hp, semilla, cfg.XGBOOST_USAR_GPU, peso_criticos=peso)
    if nombre == "XGBoost":
        modelo.set_params(eval_metric="aucpr", early_stopping_rounds=PACIENCIA)
        modelo.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        usados = int(modelo.best_iteration) + 1
        entrenados = int(modelo.get_booster().num_boosted_rounds())
    elif nombre == "LightGBM":
        modelo.set_params(metric="average_precision")
        modelo.fit(X, y, eval_set=[(X_val, y_val)],
                   callbacks=[early_stopping(PACIENCIA, first_metric_only=True, verbose=False)])
        usados = int(modelo.best_iteration_)
        entrenados = int(modelo.booster_.current_iteration())
    else:
        raise ValueError(f"Solo se ajustan {MODELOS_AJUSTABLES}, no {nombre}")
    return modelo, usados, entrenados >= MAX_ARBOLES[nombre], time.time() - t0


# ----------------------------------------------------------------------------
def recall_precision_en_fraccion(y, score, fraccion):
    """Marcando el `fraccion` de incidentes con mayor puntaje: (recall, precision)."""
    y = np.asarray(y)
    k = max(int(round(len(y) * fraccion)), 1)
    top = np.argpartition(-np.asarray(score), k - 1)[:k]
    tp = int(y[top].sum())
    return tp / max(int(y.sum()), 1), tp / k


def medir(modelo, particiones, X_train_muestra, y_train_muestra, fraccion_alertas):
    """Metricas que no dependen del umbral, en val_ajuste (eligio) y val_umbral (confirma).
    Devuelve (fila_de_metricas, puntajes_val_umbral)."""
    (Xa, ya), (Xu, yu) = particiones["val_ajuste"], particiones["val_umbral"]
    pa, pu = probabilidad_critico(modelo, Xa), probabilidad_critico(modelo, Xu)
    pt = probabilidad_critico(modelo, X_train_muestra)
    rec, pre = recall_precision_en_fraccion(yu, pu, fraccion_alertas)
    fila = {
        "PR_AUC_train": average_precision_score(y_train_muestra, pt),
        "PR_AUC_val_ajuste": average_precision_score(ya, pa),
        "ROC_AUC_val_ajuste": roc_auc_score(ya, pa),
        "PR_AUC_val_umbral": average_precision_score(yu, pu),
        "ROC_AUC_val_umbral": roc_auc_score(yu, pu),
        "Recall_en_alertas": rec,
        "Precision_en_alertas": pre,
    }
    fila["Brecha_train_vs_ajuste"] = fila["PR_AUC_train"] - fila["PR_AUC_val_ajuste"]
    return fila, pu.astype("float32")


def diferencia_pr_auc_bootstrap(y, score_a, score_b, n_boot=200, semilla=42, alfa=0.05):
    """PR-AUC(a) - PR-AUC(b) con intervalo por bootstrap PAREADO (mismos incidentes
    remuestreados para ambos). Devuelve (diferencia, limite_inferior, limite_superior).

    Ojo: remuestrea incidentes como si fueran independientes; entre semanas hay
    deriva, asi que la incertidumbre real es algo MAYOR que este intervalo."""
    y = np.asarray(y)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(semilla)
    n = len(y)
    difs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        difs[i] = average_precision_score(y[idx], score_a[idx]) - average_precision_score(y[idx], score_b[idx])
    base = average_precision_score(y, score_a) - average_precision_score(y, score_b)
    return float(base), float(np.quantile(difs, alfa / 2)), float(np.quantile(difs, 1 - alfa / 2))


MEJORA_MINIMA = 0.005   # PR-AUC: por debajo de esto la diferencia es del tamano del ruido entre periodos


def veredicto(delta, inferior, superior, minimo=MEJORA_MINIMA):
    """Lectura de la diferencia (candidato - actual) en val_umbral: el intervalo debe excluir
    el 0 Y la mejora debe ser relevante (>= minimo). Que el intervalo excluya el 0 solo dice
    que la diferencia no es CERO, no que sea util."""
    if inferior > 0 and delta >= minimo:
        return "MEJORA confirmada y relevante"
    if inferior > 0:
        return f"MEJORA ESTADISTICA PERO PEQUENA (menos de {minimo}): no justifica cambiar la configuracion"
    if superior < 0:
        return "EMPEORA (el intervalo queda por debajo de 0)"
    return "SIN DIFERENCIA CLARA (el intervalo incluye el 0)"
