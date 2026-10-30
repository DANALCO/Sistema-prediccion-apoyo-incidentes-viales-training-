"""
=============================================================================
SCRIPT 05  -  ANALISIS DE ERRORES DEL MODELO INTEGRADO
=============================================================================
Entrada : outputs/modelos/<escenario>/   (script 03)
          outputs/limpio/                (script 01)
Salida  : eda/figures/errores_por_segmento_<escenario>.csv
          eda/figures/errores_estabilidad_mensual_<escenario>.csv
          eda/figures/errores_por_estado_<escenario>.png
          eda/figures/errores_estabilidad_mensual_<escenario>.png

Este script responde tres preguntas sobre el TEST:
  1. DONDE falla: en que estados, horas, climas... el modelo detecta menos
     (recall bajo), se equivoca mas en sus alertas (precision baja) o concentra
     los criticos que no detecta.
  2. SI el rendimiento es parejo: si el recall cambia mucho entre estados, el
     modelo funciona mejor en unas regiones que en otras.
  3. SI el modelo se mantiene en el tiempo: tasa de criticos, de alertas, recall
     y exactitud mes a mes (deriva).

Se usa la decision final (modelo integrado + umbral fijado en validacion).

Uso (desde la raiz del repositorio):   python pipeline/05_analisis_errores.py
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
from accidentes.analysis import (grafico_estabilidad, grafico_recall_precision,
                                 tabla_estabilidad_mensual, tabla_segmentos)
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

MIN_CRITICOS_SEGMENTO = 200     # para hablar del recall de un segmento hacen falta criticos suficientes


def main():
    parser = argparse.ArgumentParser(description="Analisis de errores del modelo integrado.")
    parser.add_argument("--escenario", choices=list(cfg.ESCENARIOS), default="tiempo_real")
    args = parser.parse_args()
    esc = args.escenario

    titulo(f"ANALISIS DE ERRORES - {esc}")
    directorio = cfg.DIR_MODELOS / esc
    if not (directorio / "decision.json").exists():
        raise FileNotFoundError(f"No hay modelos en {directorio}. Ejecute antes pipeline/03_entrenar_modelos.py")
    with open(directorio / "decision.json", encoding=cfg.ENCODING) as f:
        decision = json.load(f)
    puntajes = np.load(directorio / "puntajes.npz")
    score, y = puntajes[f"{cfg.METODO_FINAL}__test"], puntajes["y__test"]
    pos_test = np.load(cfg.DIR_PREPARADO / f"datos_{esc}.npz")["pos_test"]

    umbral = decision["umbral"]
    pred = score >= umbral
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    recall, precision = tp / (tp + fn), tp / max(tp + fp, 1)
    print(f"Modelo: {decision['metodo_final']} | estrategia: {decision.get('estrategia_balanceo', '?')} "
          f"| umbral {umbral:.5f}")
    print(f"Test: {len(y):,} incidentes | criticos {int(y.sum()):,} ({y.mean() * 100:.2f}%)")
    print(f"  Aciertos criticos (TP) {tp:>9,} | Falsas alarmas (FP) {fp:>9,}")
    print(f"  Criticos perdidos (FN) {fn:>9,} | Normales bien (TN)   {tn:>9,}")
    print(f"  Recall {recall:.3f} | Precision {precision:.3f} | "
          f"Exactitud {(tp + tn) / len(y):.4f} (baseline {1 - y.mean():.4f})")

    # --- Datos originales de los incidentes del test ---
    columnas = ["Start_Time", "State", "Hora", "Dia_semana", "Mes", "Weather_Condition", "Estacion",
                "Es_hora_pico", "Es_fin_de_semana", "Sunrise_Sunset"] + cfg.INFRAESTRUCTURA
    tabla_limpia = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))
    df = tabla_limpia[columnas].iloc[pos_test].reset_index(drop=True)
    del tabla_limpia

    # --- Errores por segmento ---
    segmentos = ["State", "Hora", "Dia_semana", "Mes", "Weather_Condition", "Estacion", "Es_hora_pico",
                 "Es_fin_de_semana", "Sunrise_Sunset"] + cfg.INFRAESTRUCTURA
    tablas = [tabla_segmentos(df, y, pred, v, maximo=15 if v == "State" else None) for v in segmentos]
    todo = pd.concat(tablas, ignore_index=True)
    cfg.DIR_EDA.mkdir(parents=True, exist_ok=True)
    todo.round(4).to_csv(cfg.DIR_EDA / f"errores_por_segmento_{esc}.csv", index=False, encoding=cfg.ENCODING)

    estados = tablas[0]
    titulo("ERRORES POR ESTADO (los 15 con mas incidentes)")
    print(estados[["Categoria", "Incidentes", "Criticos", "Tasa_critico_%", "Alertas_%", "Recall",
                   "Precision", "Pct_de_los_FN"]].round(3).to_string(index=False))
    suficientes = estados[estados["Criticos"] >= MIN_CRITICOS_SEGMENTO]
    if len(suficientes) >= 2:
        mejor, peor = suficientes.loc[suficientes["Recall"].idxmax()], suficientes.loc[suficientes["Recall"].idxmin()]
        print(f"\nRecall por estado (con >= {MIN_CRITICOS_SEGMENTO} criticos): "
              f"de {peor['Recall']:.3f} ({peor['Categoria']}) a {mejor['Recall']:.3f} ({mejor['Categoria']}); "
              f"global {recall:.3f}.")
        print("Si el recall cambia mucho entre estados, el modelo funciona mejor en unas regiones que en otras.")

    titulo("DONDE SE CONCENTRAN LOS CRITICOS QUE EL MODELO NO DETECTA (falsos negativos)")
    for v in ["Hora", "Weather_Condition", "Es_hora_pico", "Sunrise_Sunset"]:
        t = tablas[segmentos.index(v)].sort_values("Pct_de_los_FN", ascending=False).head(4)
        print(f"\n{v}:")
        print(t[["Categoria", "Incidentes", "Criticos", "Recall", "Pct_de_los_FN", "Pct_de_los_FP"]]
              .round(3).to_string(index=False))

    # --- Estabilidad en el tiempo ---
    mensual = tabla_estabilidad_mensual(df["Start_Time"], y, pred)
    mensual.round(4).to_csv(cfg.DIR_EDA / f"errores_estabilidad_mensual_{esc}.csv", index=False,
                            encoding=cfg.ENCODING)
    titulo("ESTABILIDAD MES A MES EN EL TEST (deriva)")
    print(mensual[["Mes", "Incidentes", "Tasa_critico_%", "Alertas_%", "Recall", "Precision", "Exactitud",
                   "Baseline_exactitud"]].round(4).to_string(index=False))
    print("\nSi la tasa de criticos o de alertas sube o baja con el tiempo, el umbral fijado en validacion")
    print("deja de ser el ideal y conviene recalibrarlo con datos recientes.")

    # --- Graficos ---
    grafico_recall_precision(estados, f"Recall y precision por estado - {esc}",
                             cfg.DIR_EDA / f"errores_por_estado_{esc}.png", recall, precision)
    grafico_estabilidad(mensual, f"Criticos y alertas por mes (test) - {esc}",
                        cfg.DIR_EDA / f"errores_estabilidad_mensual_{esc}.png")

    print("\n" + "=" * 70)
    print("ANALISIS DE ERRORES COMPLETADO")
    print(f"Resultados en {cfg.DIR_EDA}")
    print("=" * 70)


if __name__ == "__main__":
    main()
