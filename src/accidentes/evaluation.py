"""
EVALUACION
=============================================================================
Responsabilidad unica: medir. No entrena ni combina nada.

Dos familias de metricas, y la diferencia importa:

  EN EL UMBRAL (dependen del umbral elegido)
      exactitud     aciertos / total. ES LA DE LA RUBRICA. Se acompana SIEMPRE
                    del baseline (la exactitud de decir "todo normal").
      recall        de los criticos reales, cuantos detecta.
      precision     de las alertas, cuantas eran criticos reales.
      F1, balanced accuracy, % de incidentes marcados.

  INDEPENDIENTES DEL UMBRAL (miden que tan bien ORDENA el riesgo)
      ROC-AUC, PR-AUC y su "lift" (veces la tasa base).

CURVA DE UMBRAL: `curva_de_umbral` ordena los incidentes una vez y calcula las
metricas en cada umbral posible. Alimenta el control de umbral de la aplicacion.

Con tan pocos criticos, un modelo que siempre responde "normal" saca ~97% de
exactitud sin detectar nada (el ejemplo del fraude de la clase). Por eso la
exactitud nunca se reporta sola.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from accidentes.ensemble import ZONAS


def metricas_en_umbral(y, score, umbral):
    y = np.asarray(y)
    pred = np.asarray(score) >= umbral
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    n = len(y)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Exactitud": (tp + tn) / n,
        "Baseline": 1 - y.mean(),
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Balanced_acc": (recall + especificidad) / 2,
        "Alertas_%": pred.mean() * 100,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def metricas_independientes(y, score):
    y = np.asarray(y)
    pr_auc = average_precision_score(y, score)
    return {
        "PR_AUC": pr_auc,
        "Lift_PR_AUC": pr_auc / y.mean(),
        "ROC_AUC": roc_auc_score(y, score),
    }


def puntos_rubrica(exactitud, niveles):
    """Puntos del criterio de aplicacion segun la exactitud (sin contar las
    funcionalidades, que la rubrica exige en conjunto: 8, 9 o 10)."""
    for minimo, puntos in niveles:
        if exactitud >= minimo:
            return puntos
    return 0


def curva_de_umbral(y, score, umbral_oficial=None, niveles=None, fracciones=None):
    """Que pasa en cada umbral posible: tabla precalculada para el control de umbral de la app.

    Una fila por cantidad de alertas (de 0.25% a 30% de los incidentes, de a 0.25%)
    mas una fila EXACTA para el umbral oficial. El umbral de cada fila es el
    puntaje del ultimo incidente marcado; si hay empates en ese puntaje se marcan
    todos (igual que `score >= umbral`), por eso las filas repetidas se descartan.

    Todo sale de ordenar los incidentes UNA vez, sin reentrenar nada: con las k
    alertas de mayor puntaje, TP_k es la suma acumulada de criticos reales.

    Columnas *_x1000 = "por cada 1,000 incidentes", la forma mas facil de explicar
    el costo de mover el umbral: criticos detectados, falsas alarmas, criticos perdidos.
    `Puntos_solo_exactitud` son los puntos de la rubrica por la exactitud sola (las
    funcionalidades se exigen aparte).
    """
    from accidentes import config as cfg

    niveles = cfg.NIVELES_RUBRICA if niveles is None else niveles
    y = np.asarray(y)
    score = np.asarray(score, dtype="float64")
    n, positivos = len(y), int(y.sum())
    negativos = n - positivos
    orden = np.argsort(-score, kind="stable")
    ordenado, ys = score[orden], y[orden]
    tp_acum = np.cumsum(ys)

    if fracciones is None:
        fracciones = np.round(np.arange(0.0025, 0.3001, 0.0025), 4)
    ks = np.clip(np.round(np.asarray(fracciones) * n).astype(int), 1, n)
    k_oficial = int((score >= umbral_oficial).sum()) if umbral_oficial is not None else 0
    if k_oficial > 0:
        ks = np.append(ks, k_oficial)

    filas, vistos = [], set()
    for k in np.unique(ks):
        umbral = float(ordenado[k - 1])
        # empates: todos los incidentes con puntaje >= umbral quedan marcados
        marcados = int(np.searchsorted(-ordenado, -umbral, side="right"))
        if marcados in vistos:
            continue
        vistos.add(marcados)
        tp = int(tp_acum[marcados - 1])
        fp = marcados - tp
        fn = positivos - tp
        tn = negativos - fp
        recall = tp / positivos if positivos else 0.0
        precision = tp / marcados if marcados else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        especificidad = tn / negativos if negativos else 0.0
        exactitud = (tp + tn) / n
        es_oficial = bool(k_oficial > 0 and marcados == k_oficial)
        filas.append({
            "Umbral": float(umbral_oficial) if es_oficial else umbral,
            "Alertas_%": marcados / n * 100,
            "Exactitud": exactitud,
            "Baseline": 1 - positivos / n,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Balanced_acc": (recall + especificidad) / 2,
            "Criticos_detectados_x1000": tp / n * 1000,
            "Falsas_alarmas_x1000": fp / n * 1000,
            "Criticos_perdidos_x1000": fn / n * 1000,
            "Es_umbral_oficial": es_oficial,
            "Puntos_solo_exactitud": puntos_rubrica(exactitud, niveles),
        })
    return pd.DataFrame(filas)


def tabla_zonas(y, zonas):
    """Cuantos incidentes y que % de criticos reales hay en cada zona.
    Debe CRECER de 'Normal' a 'Critico': eso demuestra que la vigilancia significa algo."""
    y = np.asarray(y)
    zonas = np.asarray(zonas)
    filas = []
    for zona in ZONAS:
        m = zonas == zona
        n = int(m.sum())
        criticos = int(y[m].sum()) if n else 0
        pct = criticos / n * 100 if n else 0.0
        filas.append({
            "Zona": zona, "Incidentes": n, "Pct_incidentes": n / len(y) * 100,
            "Criticos": criticos, "Pct_critico_en_zona": pct,
            "Veces_la_tasa_base": pct / (y.mean() * 100) if y.mean() else 0.0,
        })
    return pd.DataFrame(filas)


def comparar_escenarios(tabla_tiempo_real, tabla_completa, particion="test"):
    """Pregunta de investigacion 4: cuanto se pierde al operar en tiempo real."""
    columnas = ["Exactitud", "Recall", "Precision", "F1", "PR_AUC", "ROC_AUC"]
    a = tabla_tiempo_real[tabla_tiempo_real["Particion"] == particion].set_index("Metodo")[columnas]
    b = tabla_completa[tabla_completa["Particion"] == particion].set_index("Metodo")[columnas]
    comun = a.index.intersection(b.index)
    salida = pd.DataFrame(index=comun)
    for c in columnas:
        salida[f"{c}_tiempo_real"] = a.loc[comun, c]
        salida[f"{c}_completa"] = b.loc[comun, c]
        salida[f"{c}_perdida"] = b.loc[comun, c] - a.loc[comun, c]
    return salida.reset_index()
