"""
=============================================================================
EJEMPLO NUMERICO DEL MODELO MATEMATICO  -  un incidente real, paso a paso
=============================================================================
Calcula, para UN incidente real, cada cifra que aparece en el documento del
modelo matematico (secciones 4 a 11): la preparacion, la probabilidad cruda de
los cuatro modelos, la calibracion de Platt, el promedio, el stacking, la
decision con su zona y los aportes SHAP. Tambien imprime los parametros de
Platt (a, b) de cada modelo.

Solo LEE: usa los artefactos de models/<escenario>/ (script 06) y la tabla
limpia. No entrena nada. Por defecto NO lee el test: el ejemplo sale de
val_umbral, para no mirar el test al escoger un caso ilustrativo.

El incidente se escoge sin azar y sin mirar lo bonito del resultado: entre los
incidentes de la particion que fueron criticos y cayeron en la zona Critico, el
que tiene la probabilidad MEDIANA. Es decir, un acierto tipico, no el mejor.

Con --particion test se ilustra el mismo calculo con un incidente del conjunto
de prueba (que el modelo nunca vio). La regla de escogencia es la misma y esta
fijada de antemano; el caso solo ilustra: ninguna decision del proyecto (umbral,
bordes, hiperparametros, balanceo) se toma con el.

Uso (desde la raiz del repositorio):
    python eda/ejemplo_modelo_matematico.py
    python eda/ejemplo_modelo_matematico.py --escenario informacion_completa
    python eda/ejemplo_modelo_matematico.py --particion val_ajuste
    python eda/ejemplo_modelo_matematico.py --particion test
=============================================================================
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.ensemble import logit
from accidentes.inference import COLUMNAS_ENTRADA, PredictorIncidentes
from accidentes.storage import leer_tabla
from accidentes.utils import titulo


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def escoger_incidente(predictor, entrada_criticos):
    """Indice (dentro de entrada_criticos) del acierto tipico: zona Critico, probabilidad mediana."""
    res = predictor.predecir(entrada_criticos)
    candidatos = res.index[res["Nivel"] == "Critico"]
    if len(candidatos) == 0:
        raise SystemExit("Ningun incidente critico cae en la zona Critico en esta particion.")
    p = res.loc[candidatos, "Probabilidad"].sort_values()
    return p.index[len(p) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", default="models", help="carpeta con <escenario>/ (por defecto models)")
    ap.add_argument("--escenario", default="tiempo_real")
    ap.add_argument("--particion", default="val_umbral", choices=["val_umbral", "val_ajuste", "test"])
    ap.add_argument("--arriba", type=int, default=6, help="cuantas variables de SHAP mostrar")
    args = ap.parse_args()

    predictor = PredictorIncidentes.cargar(args.modelos, args.escenario)
    d = predictor.decision
    nombres = list(cfg.NOMBRES_MODELOS)

    datos = np.load(cfg.DIR_PREPARADO / f"datos_{args.escenario}.npz")
    pos = datos[f"pos_{args.particion}"]
    y = datos[f"y_{args.particion}"]
    tabla = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))
    criticos = np.flatnonzero(y == 1)
    entrada = tabla[COLUMNAS_ENTRADA].iloc[pos[criticos]].reset_index(drop=True)
    del tabla

    elegido = escoger_incidente(predictor, entrada)
    fila = entrada.loc[[elegido]].reset_index(drop=True)

    titulo(f"EJEMPLO NUMERICO - {args.escenario}  (incidente real de {args.particion}, critico, acierto tipico)")
    if args.particion == "test":
        print("Caso del conjunto de prueba: el modelo nunca lo vio. Solo ilustra; con el no se decidio nada.")

    # [1] El incidente tal como llega al formulario ---------------------------------------
    print("\n[1] Campos del incidente (los que conoce el operador al reportarlo)")
    for c in COLUMNAS_ENTRADA:
        v = fila.at[0, c]
        print(f"    {c:20s} {'(vacio)' if pd.isna(v) else v}")

    # [2] Preparacion -------------------------------------------------------------------
    X, estimados = predictor.preparar(fila)
    vector = np.asarray(X, dtype="float64")[0]
    print(f"\n[2] Preparacion -> vector x de {len(vector)} componentes  "
          f"(campos opcionales vacios, estimados por mediana del estado: {int(estimados[0])})")
    activos = [(n, v) for n, v in zip(predictor.nombres_features, vector) if v != 0]
    print(f"    Componentes distintos de cero: {len(activos)} de {len(vector)}")
    for n, v in activos:
        print(f"    {n:40s} {v:+.4f}")

    # [3] Cuatro modelos + Platt ---------------------------------------------------------
    res = predictor.predecir(fila).iloc[0]
    platt = d["calibradores_platt"]
    print("\n[3] Probabilidad cruda, logit y calibracion de Platt de cada modelo")
    print(f"    {'Modelo':22s} {'P_cruda':>9s} {'z=logit':>9s} {'a':>9s} {'b':>9s} {'P_cal':>9s}")
    p_cal = {}
    for n in nombres:
        p_cruda = float(res[f"Prob_{n}"])
        z = float(logit(p_cruda))
        a, b = platt[n]["a"], platt[n]["b"]
        p_cal[n] = float(sigmoide(a * z + b))
        print(f"    {n:22s} {p_cruda:9.4f} {z:+9.4f} {a:+9.4f} {b:+9.4f} {p_cal[n]:9.4f}")

    # [4] Integracion -------------------------------------------------------------------
    omega = d["pesos_promedio_ponderado"]
    prom = float(np.mean([p_cal[n] for n in nombres]))
    pond = float(sum(omega[n] * p_cal[n] for n in nombres))
    beta = d["coeficientes_apilador"]
    print("\n[4] Integracion de los cuatro modelos")
    print(f"    Promedio simple       = media de las 4 P_cal                  = {prom:.4f}")
    print("    Promedio ponderado    = " + " + ".join(f"{omega[n]:.4f}*{p_cal[n]:.4f}" for n in nombres)
          + f" = {pond:.4f}")
    print("    Stacking (escala logit):")
    print(f"      intercepto beta0                                         {beta['intercepto']:+.4f}")
    total = beta["intercepto"]
    for n in nombres:
        z = float(logit(res[f"Prob_{n}"]))
        termino = beta[n] * z
        total += termino
        print(f"      {n:22s} beta={beta[n]:+.4f} * z={z:+.4f} = {termino:+.4f}")
    p_final = float(sigmoide(total))
    print(f"      logit p(x) = {total:+.4f}   ->   p(x) = {p_final:.4f}   "
          f"(el predictor entrega {res['Probabilidad']:.4f})")

    # [5] Decision ----------------------------------------------------------------------
    tau, inf, sup = d["umbral"], d["umbral_vigilancia_inferior"], d["umbral_vigilancia_superior"]
    print("\n[5] Decision y zona")
    print(f"    tau = {tau:.5f}   bordes de vigilancia = [{inf:.5f}, {sup:.5f}]")
    print(f"    p(x) = {p_final:.4f}  ->  {'critico' if p_final >= tau else 'no critico'}; zona: {res['Nivel']}")

    # [6] Explicacion -------------------------------------------------------------------
    try:
        aportes, base = predictor.explicar(fila)
    except RuntimeError as e:
        print(f"\n[6] Explicacion omitida: {e}")
    else:
        a = aportes.iloc[0]
        orden = a.abs().sort_values(ascending=False).index
        print(f"\n[6] Explicacion SHAP: valor base (logit) = {base:+.4f}; {len(a)} variables")
        for v in orden[:args.arriba]:
            print(f"    {v:20s} {a[v]:+.4f}")
        resto = a.drop(orden[:args.arriba]).sum()
        print(f"    {'(las demas)':20s} {resto:+.4f}")
        print(f"    base + suma de aportes = {float(base) + float(a.sum()):+.4f}   "
              f"(logit del integrado: {total:+.4f}; diferencia {abs(float(base) + float(a.sum()) - total):.1e})")

    # [7] Parametros de Platt -----------------------------------------------------------
    print("\n[7] Parametros de la calibracion de Platt (P_cal = sigmoide(a*logit(P) + b))")
    for n in nombres:
        print(f"    {n:22s} a = {platt[n]['a']:+.4f}   b = {platt[n]['b']:+.4f}")
    print("\nCopie desde la linea [1] hasta aqui y peguela en el chat.")


if __name__ == "__main__":
    main()
