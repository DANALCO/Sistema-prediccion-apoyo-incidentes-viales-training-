"""
INTEGRACION DE LOS 4 MODELOS
=============================================================================
Responsabilidad unica: combinar las probabilidades de los 4 modelos base en
una sola prediccion y fijar la regla de decision.

PROBLEMA QUE RESUELVE LA CALIBRACION. Los modelos se entrenan con datos
balanceados 50/50 (SMOTENC), asi que sus probabilidades creen que la mitad
de los incidentes son criticos, cuando en la realidad son ~2-3%. Antes de
combinar hay que llevarlas a la escala real, usando datos REALES de
validacion (val_ajuste). Dos formas:

  1. Calibracion de Platt por modelo:  P_cal = sigmoide(a * logit(P) + b)
     y luego PROMEDIO (simple o ponderado) de las probabilidades calibradas.
  2. STACKING: una regresion logistica aprende como usar los 4 modelos:
         P = sigmoide(b0 + b1*z_RF + b2*z_XGB + b3*z_LGBM + b4*z_LR),
         z_m = logit(P_m)
     Sus coeficientes son los PESOS de cada modelo; su intercepto corrige el
     desplazamiento causado por SMOTENC. Es la misma sigmoide de la clase.

REGLA DE DECISION. critico  <=>  puntaje >= tau.  tau se elige en val_umbral
(otra parte de la validacion, posterior a val_ajuste): el MAYOR recall cuyo
accuracy en validacion siga siendo >= EXACTITUD_MINIMA_VALIDACION.

VIGILANCIA. Una franja de casos "en el limite" a ambos lados de tau. No es
una tercera clase de la evaluacion: la decision evaluada siempre es
puntaje >= tau.
=============================================================================
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ZONAS = ("Normal", "Vigilancia (bajo el umbral)", "Vigilancia (sobre el umbral)", "Critico")


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


class CalibradorPlatt:
    """P_cal = sigmoide(a * logit(P) + b), ajustado con datos reales."""

    def fit(self, p_crudo, y):
        self.modelo_ = LogisticRegression(C=1e6, max_iter=1000)
        self.modelo_.fit(logit(p_crudo).reshape(-1, 1), y)
        self.a_ = float(self.modelo_.coef_[0, 0])
        self.b_ = float(self.modelo_.intercept_[0])
        return self

    def transformar(self, p_crudo):
        return self.modelo_.predict_proba(logit(p_crudo).reshape(-1, 1))[:, 1]


class Apilador:
    """Stacking: regresion logistica sobre los logits de los modelos base."""

    def __init__(self, nombres, C=1.0):
        self.nombres = list(nombres)
        self.C = C

    def _matriz(self, p_crudos):
        return np.column_stack([logit(p_crudos[n]) for n in self.nombres])

    def fit(self, p_crudos, y):
        self.modelo_ = LogisticRegression(C=self.C, max_iter=1000)
        self.modelo_.fit(self._matriz(p_crudos), y)
        return self

    def transformar(self, p_crudos):
        return self.modelo_.predict_proba(self._matriz(p_crudos))[:, 1]

    @property
    def coeficientes(self):
        coef = {"intercepto": float(self.modelo_.intercept_[0])}
        coef.update({n: float(c) for n, c in zip(self.nombres, self.modelo_.coef_[0])})
        return coef


def pesos_por_pr_auc(y, scores):
    """Pesos del promedio ponderado: proporcionales al PR-AUC en val_ajuste."""
    pr = {n: average_precision_score(y, s) for n, s in scores.items()}
    total = sum(pr.values())
    return {n: v / total for n, v in pr.items()}


def seleccionar_umbral(score, y, exactitud_minima):
    """Mayor numero de alertas cuya exactitud sigue siendo >= exactitud_minima.

    Se ordenan los incidentes por puntaje y se marca el top-k. La exactitud
    con k alertas es (N - k - P + 2*TP_k) / N. Se toma el k justo antes de
    la PRIMERA vez que la exactitud cae por debajo del minimo.
    """
    score = np.asarray(score, dtype="float64")
    y = np.asarray(y)
    n, positivos = len(y), int(y.sum())
    orden = np.argsort(-score, kind="stable")
    ordenado, ys = score[orden], y[orden]
    k = np.arange(1, n + 1)
    exactitud = (n - k - positivos + 2 * np.cumsum(ys)) / n
    violaciones = np.flatnonzero(exactitud < exactitud_minima)
    k_sel = int(violaciones[0]) if len(violaciones) else n
    umbral = float(ordenado[k_sel - 1]) if k_sel > 0 else float(ordenado[0]) + 1e-9
    return umbral


def umbrales_vigilancia(score_val, umbral, factor_inferior, factor_superior):
    """Bordes de la franja de vigilancia, fijados por VOLUMEN en validacion.

    inferior: puntaje que marcaria factor_inferior (1.5) veces los incidentes
              que marca el umbral.
    superior: puntaje que marcaria factor_superior (2/3) de los que marca el umbral.
    """
    ordenado = np.sort(np.asarray(score_val, dtype="float64"))[::-1]
    k = int((score_val >= umbral).sum())
    if k == 0:
        return umbral, umbral
    k_inf = min(len(ordenado), int(np.ceil(k * factor_inferior)))
    k_sup = max(1, int(round(k * factor_superior)))
    inferior = min(float(ordenado[k_inf - 1]), umbral)
    superior = max(float(ordenado[k_sup - 1]), umbral)
    return inferior, superior


def asignar_zona(score, inferior, umbral, superior):
    score = np.asarray(score)
    return np.select(
        [score < inferior, score < umbral, score < superior],
        [ZONAS[0], ZONAS[1], ZONAS[2]],
        default=ZONAS[3],
    )
