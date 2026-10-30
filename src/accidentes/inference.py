"""
INFERENCIA: PREDECIR, VALIDAR Y EXPLICAR INCIDENTES NUEVOS
=============================================================================
Responsabilidad unica: dado un incidente recien reportado (lo que llena el
formulario de la aplicacion), devolver su riesgo, su nivel de alerta y la
explicacion. Es el UNICO punto de entrada que debe usar la app: asi aplica
exactamente las mismas transformaciones, el mismo stacking y el mismo umbral
que se evaluaron al entrenar. La app no debe reimplementar nada de esto.

    from accidentes.inference import PredictorIncidentes
    predictor = PredictorIncidentes.cargar("models", "tiempo_real")
    resultado = predictor.predecir(incidentes)      # DataFrame, una fila por incidente
    resultado = predictor.predecir(incidentes, umbral=0.30)   # control de umbral de la app
    explicacion, base = predictor.explicar(incidentes)

ENTRADA (una fila por incidente; nombres de columna EXACTOS)
  Obligatorias : Start_Time (fecha y hora del reporte), State, Start_Lat, Start_Lng
  Opcionales   : Temperature(F), Humidity(%), Pressure(in), Visibility(mi),
                 Wind_Speed(mph), Precipitation(in), Wind_Direction,
                 Weather_Condition (uno de los grupos de opciones_formulario()),
                 Sunrise_Sunset, y las 13 columnas de infraestructura (si/no).
                 Tipo_via (uno de opciones_formulario()['Tipo_via']: Interestatal, Carretera_US,
                 Ruta_estatal, Autopista, Carretera, Local, Otro). Si la app prefiere pedir el
                 nombre de la calle, puede enviar `Street` (texto libre) en lugar de Tipo_via:
                 se clasifica con la MISMA regla del entrenamiento (features.clasificar_via).
  Si falta un opcional, se estima con la mediana de entrenamiento y se cuenta en
  'Campos_estimados' (la prediccion es entonces menos fiable). Si falta un
  obligatorio o un valor es fisicamente imposible, se lanza ValueError.

SALIDA DE predecir()
  Probabilidad       riesgo estimado de impacto critico (0 a 1)
  Es_critico         decision evaluada: Probabilidad >= umbral (el oficial de decision.json,
                     o el que pase la app en `umbral=`; ver la curva de umbral para
                     saber que cuesta cada valor)
  Nivel              Normal / Vigilancia (bajo el umbral) / Vigilancia (sobre el
                     umbral) / Critico
  Nivel_simple       Normal / Vigilancia / Critico (para mostrar en pantalla)
  Campos_estimados   cuantos opcionales faltaban
  Prob_<modelo>      probabilidad cruda de cada uno de los 4 modelos
  Aporte_<modelo>    cuanto aporta cada modelo al riesgo final (escala logit);
                     'Aporte_intercepto' + la suma de aportes = logit(Probabilidad)
=============================================================================
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.ensemble import asignar_zona, logit
from accidentes.explain import ExplicadorStacking
from accidentes.features import agregar_variables_temporales, tipos_de_via
from accidentes.models import probabilidad_critico

COLUMNAS_OBLIGATORIAS = ["Start_Time", "State", "Start_Lat", "Start_Lng"]
COLUMNAS_OPCIONALES = (cfg.CLIMA_NUMERICO + ["Wind_Direction", "Weather_Condition", "Sunrise_Sunset"]
                       + cfg.INFRAESTRUCTURA + ["Tipo_via"])
COLUMNAS_ENTRADA = COLUMNAS_OBLIGATORIAS + COLUMNAS_OPCIONALES
NIVEL_SIMPLE = {
    "Normal": "Normal",
    "Vigilancia (bajo el umbral)": "Vigilancia",
    "Vigilancia (sobre el umbral)": "Vigilancia",
    "Critico": "Critico",
}
LIMITES_COORDENADAS = {"Start_Lat": (-90, 90), "Start_Lng": (-180, 180)}


class PredictorIncidentes:

    def __init__(self, preprocesador, modelos, apilador, decision, nombres_features,
                 estado_explicador=None):
        self.preprocesador = preprocesador
        self.modelos = modelos
        self.apilador = apilador
        self.decision = decision
        self.nombres_features = list(nombres_features)
        self.estado_explicador = estado_explicador
        self._explicador = None

    @classmethod
    def cargar(cls, directorio, escenario="tiempo_real"):
        """Carga los artefactos que deja pipeline/06_exportar_para_app.py en <directorio>/<escenario>/."""
        d = Path(directorio) / escenario
        if not (d / "decision.json").exists():
            raise FileNotFoundError(f"No hay artefactos en {d}. Ejecute pipeline/06_exportar_para_app.py")
        modelos = {n: joblib.load(d / f"modelo_{n}.joblib") for n in cfg.NOMBRES_MODELOS}
        with open(d / "decision.json", encoding=cfg.ENCODING) as f:
            decision = json.load(f)
        estado = None
        if (d / "explicador_estado.json").exists():
            with open(d / "explicador_estado.json", encoding=cfg.ENCODING) as f:
                estado = json.load(f)
        nombres = (d / "features.txt").read_text(encoding=cfg.ENCODING).split("\n")
        return cls(joblib.load(d / "preprocesador.joblib"), modelos, joblib.load(d / "apilador.joblib"),
                   decision, nombres, estado)

    # ------------------------------------------------------------------
    def opciones_formulario(self):
        """Valores validos de cada campo categorico (para los desplegables del formulario)."""
        return {c: list(v) for c, v in self.preprocesador.categorias_.items()
                if c in ("State", "Wind_Direction", "Weather_Condition", "Sunrise_Sunset", "Tipo_via")}

    def validar(self, incidentes):
        """Devuelve (tabla de entrada completa, campos_estimados por fila) o lanza ValueError."""
        faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in incidentes.columns]
        if faltan:
            raise ValueError(f"Faltan columnas obligatorias: {faltan}")
        vacias = [c for c in COLUMNAS_OBLIGATORIAS if incidentes[c].isna().any()]
        if vacias:
            raise ValueError(f"Hay valores vacios en campos obligatorios: {vacias}")
        limites = {**cfg.LIMITES_FISICOS, **LIMITES_COORDENADAS}
        for col, (minimo, maximo) in limites.items():
            if col in incidentes.columns:
                fuera = incidentes[col].notna() & ~incidentes[col].between(minimo, maximo)
                if fuera.any():
                    raise ValueError(f"{col} fuera del rango posible [{minimo}, {maximo}]")
        tabla = incidentes.copy()
        if "Street" in tabla.columns:                  # la calle solo completa un Tipo_via que falta
            derivado = tipos_de_via(tabla["Street"])   # misma regla que en el entrenamiento
            if "Tipo_via" in tabla.columns:
                tabla["Tipo_via"] = tabla["Tipo_via"].astype(object).where(tabla["Tipo_via"].notna(), derivado)
            else:
                tabla["Tipo_via"] = derivado
        if "Tipo_via" in tabla.columns:
            validos = set(self.preprocesador.categorias_["Tipo_via"])
            invalidos = sorted(set(tabla["Tipo_via"].dropna().astype(str)) - validos)
            if invalidos:
                raise ValueError(f"Tipo_via no valido: {invalidos}. Opciones: {sorted(validos)}")
        for col in COLUMNAS_OPCIONALES:
            if col not in tabla.columns:
                tabla[col] = np.nan
        tabla["Start_Time"] = pd.to_datetime(tabla["Start_Time"])
        estimados = tabla[COLUMNAS_OPCIONALES].isna().sum(axis=1).to_numpy()
        return tabla[COLUMNAS_ENTRADA], estimados

    def preparar(self, incidentes):
        tabla, estimados = self.validar(incidentes)
        X = self.preprocesador.transformar(agregar_variables_temporales(tabla))
        return X, estimados

    # ------------------------------------------------------------------
    def predecir(self, incidentes, umbral=None):
        """Riesgo, decision y nivel de cada incidente.

        umbral: None usa el umbral oficial (el elegido en validacion, decision.json).
        Un valor entre 0 y 1 lo reemplaza (control de umbral de la app). La franja de
        vigilancia se reescala en la misma proporcion, de modo que siga rodeando al
        umbral. La probabilidad y los aportes NO cambian: solo la decision y el nivel.
        """
        d = self.decision
        if umbral is None:
            tau = d["umbral"]
        else:
            tau = float(umbral)
            if not 0 < tau < 1:
                raise ValueError(f"El umbral debe estar entre 0 y 1 (recibido {umbral})")
        X, estimados = self.preparar(incidentes)
        crudas = {n: probabilidad_critico(m, X) for n, m in self.modelos.items()}
        beta = self.apilador.coeficientes
        aportes = {n: beta[n] * logit(p) for n, p in crudas.items()}
        z = beta["intercepto"] + sum(aportes.values())
        prob = 1 / (1 + np.exp(-z))

        if tau == d["umbral"]:
            inferior, superior = d["umbral_vigilancia_inferior"], d["umbral_vigilancia_superior"]
        else:
            factor = tau / d["umbral"]
            inferior = factor * d["umbral_vigilancia_inferior"]
            superior = factor * d["umbral_vigilancia_superior"]
        zonas = asignar_zona(prob, inferior, tau, superior)
        salida = pd.DataFrame({
            "Probabilidad": prob,
            "Es_critico": prob >= tau,
            "Nivel": zonas,
            "Nivel_simple": [NIVEL_SIMPLE[z_] for z_ in zonas],
            "Campos_estimados": estimados,
        }, index=incidentes.index)
        for n in self.modelos:
            salida[f"Prob_{n}"] = crudas[n]
        salida["Aporte_intercepto"] = beta["intercepto"]
        for n in self.modelos:
            salida[f"Aporte_{n}"] = aportes[n]
        return salida

    def explicar(self, incidentes):
        """Devuelve (aporte de cada variable al logit del riesgo, valor base en logit)."""
        if self.estado_explicador is None:
            raise RuntimeError("No hay explicador_estado.json; ejecute pipeline/04_explicabilidad_shap.py")
        if self._explicador is None:
            self._explicador = ExplicadorStacking.desde_estado(
                self.modelos, self.apilador, self.nombres_features, cfg.CATEGORICAS, self.estado_explicador)
        X, _ = self.preparar(incidentes)
        resultado = self._explicador.contribuciones(X)
        por_variable = self._explicador.por_variable(resultado)
        por_variable.index = incidentes.index
        return por_variable, resultado["base"]
