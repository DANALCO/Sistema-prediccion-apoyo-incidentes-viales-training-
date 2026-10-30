"""
=============================================================================
CASOS REALES DE VALIDACION PARA PROBAR LA APP  -  uno por zona y resultado real
=============================================================================
Busca, entre incidentes REALES de validacion, casos que el modelo clasifica en cada una de las cuatro zonas
(Normal, Vigilancia bajo el umbral, Vigilancia sobre el umbral, Critico), separando los que de verdad fueron
criticos de los que no. Sirve para comprobar a mano, en la app, que cada zona se comporta como se espera.

  - Imprime los campos de cada caso listos para copiar en el formulario.
  - Escribe models/reportes/ejemplos_app.json: la app los ofrece en una lista desplegable
    ("Casos reales de validacion") y, al analizarlos, compara el resultado del modelo con lo que paso de verdad.

NO lee el test: usa val_umbral (datos que el modelo no vio al ajustarse), igual que ejemplo_modelo_matematico.py.
De cada grupo (zona x resultado real) se toma el caso de probabilidad MEDIANA: un caso tipico, no el mejor.
Un caso aislado no valida el modelo; la evaluacion esta en test (tabla comparativa y curva de umbral).

Uso (desde la raiz del repositorio, con models/ ya exportado por pipeline/06):
    python eda/ejemplos_para_app.py
    python eda/ejemplos_para_app.py --particion val_ajuste
=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.ensemble import ZONAS
from accidentes.inference import COLUMNAS_ENTRADA, PredictorIncidentes
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

TROZO = 40_000          # incidentes por llamada a predecir (para no llenar la memoria)
# Decimales con los que el formulario de la app muestra cada campo. Se redondea ANTES de predecir, de modo que
# lo que se copia en el formulario da exactamente la misma probabilidad que se imprime aqui.
DECIMALES = {"Start_Lat": 5, "Start_Lng": 5, "Temperature(F)": 1, "Humidity(%)": 0, "Pressure(in)": 2,
             "Visibility(mi)": 1, "Wind_Speed(mph)": 1, "Precipitation(in)": 2}
ID_ZONA = {"Normal": "normal", "Vigilancia (bajo el umbral)": "vigilancia_bajo",
           "Vigilancia (sobre el umbral)": "vigilancia_sobre", "Critico": "critico"}


def a_json(v):
    """Valor serializable: NaN -> None, fecha -> texto sin segundos, numpy -> python."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.replace(second=0, microsecond=0).isoformat()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


def imprimir_caso(i, caso, entrada):
    real = "SI fue critico" if caso["real_critico"] else "NO fue critico"
    print(f"\n--- Caso {i}: zona {caso['zona']}  |  en los datos: {real}  |  riesgo estimado {caso['probabilidad']:.1%}")
    print(f"    (hay {caso['casos_en_el_grupo']:,} casos asi en la particion; este es el de probabilidad mediana)")
    print(f"    Fecha y hora : {entrada['Start_Time']}")
    print(f"    Estado       : {entrada['State']}")
    print(f"    Latitud      : {entrada['Start_Lat']}")
    print(f"    Longitud     : {entrada['Start_Lng']}")
    for col in cfg.CLIMA_NUMERICO + ["Weather_Condition", "Wind_Direction", "Sunrise_Sunset", "Tipo_via"]:
        v = entrada.get(col)
        print(f"    {col:18s} : {'(no se)' if v is None else v}")
    si = [c for c in cfg.INFRAESTRUCTURA if entrada.get(c) is True]
    no_sabe = [c for c in cfg.INFRAESTRUCTURA if entrada.get(c) is None]
    print(f"    Infraestructura 'Si': {', '.join(si) if si else 'ninguna'}"
          + (f"   |  sin dato: {', '.join(no_sabe)}" if no_sabe else "   |  el resto: No"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", default="models", help="carpeta con <escenario>/ y reportes/ (por defecto models)")
    ap.add_argument("--escenario", default="tiempo_real")
    ap.add_argument("--particion", default="val_umbral", choices=["val_umbral", "val_ajuste"])
    args = ap.parse_args()

    predictor = PredictorIncidentes.cargar(args.modelos, args.escenario)
    datos = np.load(cfg.DIR_PREPARADO / f"datos_{args.escenario}.npz")
    pos, y = datos[f"pos_{args.particion}"], datos[f"y_{args.particion}"]
    tabla = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))
    entradas = tabla[COLUMNAS_ENTRADA].iloc[pos].reset_index(drop=True)
    del tabla
    for col, decimales in DECIMALES.items():
        entradas[col] = entradas[col].astype("float64").round(decimales)

    titulo(f"CASOS REALES DE VALIDACION - {args.escenario}, {args.particion} ({len(entradas):,} incidentes)")
    prob, zona = [], []
    for ini in range(0, len(entradas), TROZO):
        r = predictor.predecir(entradas.iloc[ini:ini + TROZO])
        prob.append(r["Probabilidad"].to_numpy())
        zona.append(r["Nivel"].to_numpy())
    prob, zona = np.concatenate(prob), np.concatenate(zona)

    print("\nResumen por zona (esta particion):")
    for z in ZONAS:
        m = zona == z
        n, c = int(m.sum()), int(y[m].sum())
        print(f"  {z:30s} {n:8,d} incidentes   {c:6,d} criticos   {100 * c / max(n, 1):5.1f} % criticos")

    casos, i = [], 0
    for z in ZONAS:
        for real in (1, 0):
            idx = np.flatnonzero((zona == z) & (y == real))
            if len(idx) == 0:
                continue
            medio = idx[np.argsort(prob[idx])[len(idx) // 2]]
            fila = entradas.iloc[medio]
            entrada = {c: a_json(fila[c]) for c in COLUMNAS_ENTRADA}
            i += 1
            caso = {"id": f"{i}_{ID_ZONA[z]}_{'real_critico' if real else 'real_no_critico'}", "zona": z,
                    "real_critico": bool(real), "probabilidad": float(prob[medio]),
                    "casos_en_el_grupo": int(len(idx)), "particion": args.particion, "entrada": entrada}
            casos.append(caso)
            imprimir_caso(i, caso, entrada)

    destino = Path(args.modelos) / "reportes" / "ejemplos_app.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding=cfg.ENCODING) as f:
        json.dump(casos, f, ensure_ascii=False, indent=1)
    print(f"\n{len(casos)} casos -> {destino}")
    print("La app los ofrece en 'Casos reales de validacion' (pagina Analizar incidente). Reinicie la app si estaba abierta.")


if __name__ == "__main__":
    main()
