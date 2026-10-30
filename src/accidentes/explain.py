"""
EXPLICABILIDAD DEL MODELO INTEGRADO (SHAP aditivo)
=============================================================================
Responsabilidad unica: explicar POR QUE el modelo integrado (stacking) asigno
cierto riesgo a un incidente. No entrena nada.

IDEA CLAVE: el stacking es una regresion logistica sobre los logits de los
modelos base,

    logit(P) = b0 + b_RF*z_RF + b_XGB*z_XGB + b_LGBM*z_LGBM + b_LR*z_LR ,

asi que el integrado es una SUMA PONDERADA. Como los valores SHAP son aditivos,
la explicacion del integrado es la suma ponderada de las explicaciones de cada
modelo, en escala logit:

    phi_j(integrado) = sum_m  b_m * phi_j(modelo m)
    base(integrado)  = b0 + sum_m  b_m * base(modelo m)
    logit(P)         = base + sum_j phi_j          <- se verifica en cada corrida

Como se obtiene phi de cada modelo:
    XGBoost, LightGBM   contribuciones nativas del arbol (exactas, en log-odds)
    Regresion logistica phi_j = coef_j * (x_j - media_j)   (exacto, lineal)
    Random Forest       TreeSHAP en escala de PROBABILIDAD, reescalado a logit
                        con el factor  (logit(p) - logit(p0)) / (p - p0), que
                        conserva la suma exactamente. Si se desactiva, su
                        aporte total se muestra como una sola barra
                        "Random_Forest (no desglosado)".

La explicacion se agrega por VARIABLE ORIGINAL (todas las columnas one-hot de
State se suman en "State"), que es lo que entiende un operador.

La aplicacion usa la misma clase para mostrar la explicacion de un incidente.
=============================================================================
"""

import numpy as np
import pandas as pd

from accidentes.ensemble import logit

NOMBRE_RF_AGREGADO = "Random_Forest (no desglosado)"


def variable_original(nombre_feature, categoricas):
    """'State_CA' -> 'State'; las demas columnas se llaman como su variable."""
    for c in categoricas:
        if nombre_feature.startswith(f"{c}_"):
            return c
    return nombre_feature


def grupo_de_variable(variable, grupos):
    for grupo, variables in grupos.items():
        if variable in variables:
            return grupo
    return "Otros"


class ExplicadorStacking:

    def __init__(self, modelos, apilador, nombres_features, categoricas, incluir_rf=True):
        self.modelos = modelos
        self.apilador = apilador
        self.nombres_features = list(nombres_features)
        self.categoricas = list(categoricas)
        self.incluir_rf = incluir_rf
        self.beta = apilador.coeficientes
        self.medias_ = None
        self.z_rf_medio_ = None
        self._explicador_rf = None

    # ------------------------------------------------------------------
    def fit(self, X_fondo, medias=None):
        """Calcula lo que necesita de los datos de referencia (solo train):
        las medias que usa la regresion logistica y, si se desglosa, el TreeSHAP de RF."""
        self.medias_ = np.asarray(medias if medias is not None else X_fondo.mean(axis=0), dtype="float64")
        if self.incluir_rf:
            import shap   # solo se necesita para Random Forest
            self._explicador_rf = shap.TreeExplainer(self.modelos["Random_Forest"])
        else:
            # Sin desglose: la referencia de RF es su logit medio sobre datos de entrenamiento
            p = self.modelos["Random_Forest"].predict_proba(np.asarray(X_fondo, dtype="float32"))[:, 1]
            self.z_rf_medio_ = float(np.mean(logit(p)))
        return self

    @classmethod
    def desde_estado(cls, modelos, apilador, nombres_features, categoricas, estado):
        """Reconstruye el explicador con el estado que guardo estado() (lo usa la aplicacion)."""
        e = cls(modelos, apilador, nombres_features, categoricas, incluir_rf=estado["incluir_rf"])
        e.medias_ = np.asarray(estado["medias_lr"], dtype="float64")
        if e.incluir_rf:
            import shap
            e._explicador_rf = shap.TreeExplainer(modelos["Random_Forest"])
        else:
            e.z_rf_medio_ = estado["z_rf_medio"]
        return e

    def estado(self):
        """Lo minimo para reconstruir el explicador en otro lugar (por ejemplo, la app)."""
        return {"incluir_rf": self.incluir_rf, "medias_lr": self.medias_.tolist(),
                "z_rf_medio": self.z_rf_medio_}

    # ------------------------------------------------------------------
    def _contrib_xgboost(self, X):
        import xgboost as xgb
        c = self.modelos["XGBoost"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return c[:, :-1].astype("float64"), float(c[0, -1])

    def _contrib_lightgbm(self, X):
        c = self.modelos["LightGBM"].booster_.predict(X, pred_contrib=True)
        return c[:, :-1].astype("float64"), float(c[0, -1])

    def _contrib_logistica(self, X):
        lr = self.modelos["Regresion_Logistica"]
        coef = lr.coef_[0].astype("float64")
        phi = coef * (X.astype("float64") - self.medias_)
        base = float(lr.intercept_[0] + coef @ self.medias_)
        return phi, base

    def _contrib_rf(self, X):
        """Devuelve (phi_logit, base_logit, z_rf) para Random Forest."""
        rf = self.modelos["Random_Forest"]
        p = rf.predict_proba(X)[:, 1].astype("float64")
        z = logit(p)
        if not self.incluir_rf:
            return None, None, z
        valores = self._explicador_rf.shap_values(X, check_additivity=False)
        if isinstance(valores, list):
            phi_p = np.asarray(valores[1], dtype="float64")
        else:
            valores = np.asarray(valores, dtype="float64")
            phi_p = valores[:, :, 1] if valores.ndim == 3 else valores
        esperado = np.atleast_1d(self._explicador_rf.expected_value)
        p0 = float(esperado[1] if len(esperado) > 1 else esperado[0])
        denominador = p - p0
        pendiente = np.where(np.abs(denominador) < 1e-9, 1.0 / (p0 * (1 - p0)),
                             (z - float(logit(p0))) / np.where(denominador == 0, 1, denominador))
        return phi_p * pendiente[:, None], float(logit(p0)), z

    # ------------------------------------------------------------------
    def contribuciones(self, X):
        """Devuelve un diccionario con:
             phi    (n, f) aporte de cada columna al logit del integrado
             extra  (n,)   aporte agregado de RF si no se desglosa (o None)
             base   valor base (logit)
             logit  (n,)   logit del integrado reconstruido como base + suma de aportes
             logit_real (n,) logit del integrado calculado directamente (para verificar)
        """
        X = np.asarray(X, dtype="float32")
        b = self.beta
        phi_xgb, base_xgb = self._contrib_xgboost(X)
        phi_lgb, base_lgb = self._contrib_lightgbm(X)
        phi_lr, base_lr = self._contrib_logistica(X)
        phi_rf, base_rf, z_rf = self._contrib_rf(X)

        phi = b["XGBoost"] * phi_xgb + b["LightGBM"] * phi_lgb + b["Regresion_Logistica"] * phi_lr
        base = b["intercepto"] + b["XGBoost"] * base_xgb + b["LightGBM"] * base_lgb \
            + b["Regresion_Logistica"] * base_lr
        extra = None
        if self.incluir_rf:
            phi = phi + b["Random_Forest"] * phi_rf
            base += b["Random_Forest"] * base_rf
        else:
            # Sin desglose: todo el aporte de RF respecto de su media, como una barra
            extra = b["Random_Forest"] * (z_rf - self.z_rf_medio_)
            base += b["Random_Forest"] * self.z_rf_medio_

        reconstruido = base + phi.sum(axis=1) + (extra if extra is not None else 0.0)

        # logit del integrado calculado directamente con los modelos (control)
        z = {
            "Random_Forest": z_rf,
            "XGBoost": logit(self.modelos["XGBoost"].predict_proba(X)[:, 1]),
            "LightGBM": logit(self.modelos["LightGBM"].predict_proba(X)[:, 1]),
            "Regresion_Logistica": logit(self.modelos["Regresion_Logistica"].predict_proba(X)[:, 1]),
        }
        real = b["intercepto"] + sum(b[m] * z[m] for m in z)
        return {"phi": phi, "extra": extra, "base": float(base),
                "logit": reconstruido, "logit_real": real}

    # ------------------------------------------------------------------
    def por_variable(self, resultado):
        """Agrega las columnas one-hot en su variable original. DataFrame (n x variables)."""
        phi = resultado["phi"]
        df = pd.DataFrame(phi, columns=self.nombres_features)
        agrupado = df.T.groupby([variable_original(n, self.categoricas)
                                 for n in self.nombres_features], sort=False).sum().T
        if resultado["extra"] is not None:
            agrupado[NOMBRE_RF_AGREGADO] = resultado["extra"]
        return agrupado


# ---------------------------------------------------------------------------
# Tablas y graficos
# ---------------------------------------------------------------------------
def tabla_importancia(por_var, es_real, es_critico, grupos):
    """Importancia global por variable: media del valor absoluto del aporte."""
    real = por_var[es_real]
    crit = por_var[es_critico]
    tabla = pd.DataFrame({
        "Variable": por_var.columns,
        "Grupo": [grupo_de_variable(v, grupos) for v in por_var.columns],
        "Aporte_abs_medio_real": real.abs().mean().to_numpy(),
        "Aporte_abs_medio_criticos": crit.abs().mean().to_numpy(),
        "Aporte_medio_criticos": crit.mean().to_numpy(),
    })
    tabla = tabla.sort_values("Aporte_abs_medio_real", ascending=False).reset_index(drop=True)
    tabla.insert(0, "Ranking", np.arange(1, len(tabla) + 1))
    return tabla


def tabla_por_grupo(por_var, es_real, es_critico, grupos):
    """Aporte NETO de cada grupo tematico (se suman las variables del grupo y luego se toma el valor absoluto)."""
    filas = []
    for grupo, variables in grupos.items():
        cols = [v for v in variables if v in por_var.columns]
        if not cols:
            continue
        neto = por_var[cols].sum(axis=1)
        filas.append({
            "Grupo": grupo, "Variables": len(cols),
            "Aporte_abs_medio_real": float(neto[es_real].abs().mean()),
            "Aporte_abs_medio_criticos": float(neto[es_critico].abs().mean()),
            "Aporte_medio_criticos": float(neto[es_critico].mean()),
        })
    tabla = pd.DataFrame(filas).sort_values("Aporte_abs_medio_real", ascending=False).reset_index(drop=True)
    return tabla


def grafico_barras(series, titulo, ruta, etiqueta_x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = series.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(series) + 1.4))
    ax.barh(series.index, series.values, color="#2b6cb0")
    ax.set_xlabel(etiqueta_x)
    ax.set_title(titulo, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafico_contribuciones(contribuciones, base_logit, titulo, ruta, max_barras=10, etiquetas=None):
    """Barras de lo que subio (rojo) o bajo (azul) el riesgo de UN incidente."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = contribuciones.copy()
    orden = c.abs().sort_values(ascending=False)
    principales = c[orden.index[:max_barras]]
    resto = c.drop(index=principales.index).sum()
    if len(c) > max_barras:
        principales = pd.concat([principales, pd.Series({f"otras {len(c) - max_barras} variables": resto})])
    principales = principales.iloc[::-1]
    nombres = [etiquetas.get(k, k) if etiquetas else k for k in principales.index]
    colores = ["#c53030" if v > 0 else "#2b6cb0" for v in principales.values]

    sigmoide = lambda z: 1 / (1 + np.exp(-z))
    final = base_logit + c.sum()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(principales) + 1.8))
    ax.barh(nombres, principales.values, color=colores)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Aporte al riesgo (log-odds): rojo sube, azul baja")
    ax.set_title(f"{titulo}\nRiesgo base {sigmoide(base_logit) * 100:.1f}%  ->  riesgo final "
                 f"{sigmoide(final) * 100:.1f}%", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
