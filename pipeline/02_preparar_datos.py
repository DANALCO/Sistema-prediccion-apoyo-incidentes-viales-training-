"""
=============================================================================
SCRIPT 02  -  PREPARACION DE DATOS PARA MODELADO  (orquestador)
=============================================================================
Entrada : outputs/limpio/us_accidents_limpio.parquet (lo genera el script 01)
Salida  : outputs/preparado/  (matrices, preprocesadores y tablas de trazabilidad)

Este script solo ORQUESTA; cada paso vive en su modulo (SRP):
    splitting.py       -> particion temporal (train / validacion / test)
    preprocessing.py   -> imputacion, IQR, escalado, codificacion (solo train aprende)
    balancing.py       -> SMOTENC (solo sobre train)
    scenarios.py       -> arma y guarda cada escenario

ORDEN DE OPERACIONES (regla de oro: train aprende, val/test solo transforman):
  1. Cargar la tabla limpia.
  2. Dividir POR FECHA, una sola vez, para ambos escenarios.
  3. Para cada escenario:
       a. fit del preprocesador SOLO con train
       b. transformar train, validacion y test con lo aprendido
       c. SMOTENC UNICAMENTE sobre train
     Validacion y test conservan la distribucion real.

Salidas por escenario (tiempo_real, informacion_completa):
    datos_<escenario>.npz              matrices X/y de cada particion
    preprocesador_<escenario>.joblib   para la aplicacion
    features_<escenario>.txt           nombres de columnas (para SHAP)
Tablas comunes:
    particion_temporal.csv             filas, fechas y tasa de criticos por particion
    composicion_clases.csv             clases antes y despues del balanceo

Uso (desde la raiz del repositorio):   python pipeline/02_preparar_datos.py
=============================================================================
"""

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import pandas as pd

from accidentes import config as cfg
from accidentes.scenarios import guardar_escenario, preparar_escenario
from accidentes.splitting import dividir_temporal, resumen_particion
from accidentes.storage import leer_tabla
from accidentes.utils import titulo


def main():
    os.makedirs(cfg.DIR_PREPARADO, exist_ok=True)

    # ------------------------------------------------------------------
    titulo("PASO 1 - CARGA DE LA TABLA LIMPIA")
    df = leer_tabla(os.path.join(cfg.DIR_LIMPIO, "us_accidents_limpio"))
    print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
    if not df["Start_Time"].is_monotonic_increasing:
        raise RuntimeError("La tabla limpia no esta ordenada por fecha; vuelva a ejecutar 01_limpiar_dataset.py")

    # ------------------------------------------------------------------
    titulo("PASO 2 - PARTICION TEMPORAL (una sola vez para ambos escenarios)")
    posiciones = dividir_temporal(df["Start_Time"], cfg.CORTE_TRAIN,
                                  cfg.CORTE_VAL_AJUSTE, cfg.CORTE_TEST)
    tabla = resumen_particion(df, posiciones, "Start_Time", cfg.COL_OBJETIVO)
    print(tabla.to_string(index=False))
    ruta_tabla = os.path.join(cfg.DIR_PREPARADO, "particion_temporal.csv")
    tabla.to_csv(ruta_tabla, index=False, encoding=cfg.ENCODING)
    print(f"\n  -> {ruta_tabla}")
    print("  'Baseline_exactitud' es la exactitud de decir siempre 'normal'.")
    print("  Validacion y test NO se balancean: conservan la distribucion real.")

    # ------------------------------------------------------------------
    composiciones = []
    for nombre, cfg_escenario in cfg.ESCENARIOS.items():
        resultado = preparar_escenario(nombre, cfg_escenario, df, posiciones)
        titulo(f"GUARDADO - {nombre}")
        guardar_escenario(nombre, resultado, posiciones, cfg.DIR_PREPARADO)
        composiciones.extend(resultado["composicion"])
        del resultado
        gc.collect()

    ruta_comp = os.path.join(cfg.DIR_PREPARADO, "composicion_clases.csv")
    pd.DataFrame(composiciones).to_csv(ruta_comp, index=False, encoding=cfg.ENCODING)

    print("\n" + "=" * 70)
    print("PREPARACION COMPLETADA")
    print(f"Tabla de composicion de clases -> {ruta_comp}")
    print("Ambos escenarios usan las mismas filas y la misma particion temporal;")
    print("solo cambian las columnas de entrada (Distance(mi) solo en el completo).")
    print("Siguiente paso: script 03 -> entrenar los 4 modelos, calibrar, integrar.")
    print("=" * 70)


if __name__ == "__main__":
    main()
