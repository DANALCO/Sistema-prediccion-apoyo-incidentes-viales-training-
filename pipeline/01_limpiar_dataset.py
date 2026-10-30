"""
=============================================================================
SCRIPT 01  -  LIMPIEZA DEL DATASET  (orquestador)
=============================================================================
Proyecto : Sistema de prediccion en tiempo real y apoyo a la decision para
           la gestion de incidentes viales (US Accidents, Moosavi [23]).
Entregable 2 -> criterio "Construccion del Dataset".

Este script solo ORQUESTA: cada paso vive en su modulo (SRP).

    ingestion.py  -> lee el CSV y recorta al alcance (Source1, desde 2021)
    cleaning.py   -> fechas, objetivo valido, duplicados, valores imposibles
    features.py   -> variables temporales, clima agrupado, tipo de via, objetivo binario
    storage.py    -> guarda la tabla limpia

Salidas (carpeta outputs/limpio):
    us_accidents_limpio.parquet     tabla limpia (la lee el script 02)
    evidencia_fuente_anio.csv       Severity por fuente y anio: JUSTIFICA el alcance
    bitacora_limpieza.csv           filas perdidas por paso
    mapa_weather_condition.csv      categoria original -> grupo

Que NO hace (a proposito): imputar, recortar IQR, codificar, escalar ni
balancear. Eso aprende parametros de los datos y va en el script 02, solo con
train. Hacerlo aqui seria data leakage.

Uso (desde la raiz del repositorio):   python pipeline/01_limpiar_dataset.py
Requisitos: pip install -r requirements.txt
=============================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

from accidentes import cleaning, features
from accidentes import config as cfg
from accidentes.ingestion import DataIngester
from accidentes.storage import guardar_tabla
from accidentes.utils import Bitacora, titulo


def main():
    if not cfg.RUTA_CSV_ORIGINAL.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV original en: {cfg.RUTA_CSV_ORIGINAL}\n"
            "Descarguelo de Kaggle (ver data/README.md) y copielo a data/, o defina la "
            "variable de entorno ACCIDENTES_CSV con su ruta."
        )
    os.makedirs(cfg.DIR_LIMPIO, exist_ok=True)
    bitacora = Bitacora()

    # ------------------------------------------------------------------
    titulo("PASO 1 - INGESTA (solo lectura y recorte al alcance)")
    print(f"Archivo : {cfg.RUTA_CSV_ORIGINAL}")
    print(f"Alcance : {cfg.FUENTE}, Start_Time >= {cfg.FECHA_INICIO}")
    ingester = DataIngester(cfg.RUTA_CSV_ORIGINAL, cfg.COLUMNAS_A_LEER, cfg.ENCODING)
    df, evidencia, filas_csv = ingester.leer_alcance(cfg.FUENTE, cfg.FECHA_INICIO)
    filas_alcance = len(df)
    print(f"\nFilas del CSV completo : {filas_csv:,}")
    print(f"Filas en el alcance    : {filas_alcance:,} ({filas_alcance / filas_csv * 100:.1f}%)")

    ruta_evidencia = os.path.join(cfg.DIR_LIMPIO, "evidencia_fuente_anio.csv")
    evidencia.to_csv(ruta_evidencia, index=False, encoding=cfg.ENCODING)
    print("\nEVIDENCIA DEL ALCANCE: Severity (%) por fuente y anio, en TODO el archivo.")
    print("Muestra que 'Severity' no significa lo mismo en todas las fuentes ni anios.")
    print(evidencia.to_string(index=False))
    print(f"\n  -> {ruta_evidencia}")

    # ------------------------------------------------------------------
    titulo("PASO 2 - INSPECCION INICIAL (alcance)")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    print("Valores nulos por columna:")
    if nulos.empty:
        print("  (ninguna columna tiene nulos)")
    for col, n in nulos.items():
        print(f"  {col:<25} {n:>12,}  ({n / filas_alcance * 100:6.2f}%)")
    print("\nDistribucion de Severity en el alcance:")
    for clase, n in df[cfg.COL_SEVERIDAD].value_counts().sort_index().items():
        print(f"  Severity {int(clase)}: {n:>12,}  ({n / filas_alcance * 100:6.2f}%)")

    # ------------------------------------------------------------------
    titulo("PASO 3 - LIMPIEZA")
    df = cleaning.normalizar_booleanas(df, cfg.INFRAESTRUCTURA)
    df = cleaning.convertir_fechas(df)
    df = cleaning.eliminar_fechas_invalidas(df, bitacora)
    df = cleaning.validar_objetivo(df, bitacora, cfg.COL_SEVERIDAD)
    # Tipo_via se deriva del nombre de la calle ANTES de quitar duplicados y el texto se descarta
    # enseguida. Al decidir que filas son duplicadas se ignora Tipo_via: asi sobrevive
    # exactamente el mismo conjunto de filas que cuando la tabla no traia esta columna.
    df = features.agregar_tipo_via(df)
    por_tipo = df["Tipo_via"].value_counts(dropna=False, normalize=True) * 100
    print("Tipo_via (derivado de Street; el texto se descarta):")
    for tipo, pct in por_tipo.items():
        etiqueta = tipo if isinstance(tipo, str) else "(vacio)"       # el preprocesador lo manda a 'Desconocido'
        print(f"  {etiqueta:<14} {pct:5.1f}%")
    df = cleaning.eliminar_duplicados(df, bitacora, ignorar=("Tipo_via",))
    print("\nValores fisicamente imposibles -> NaN (no se borran filas):")
    df, _ = cleaning.marcar_valores_imposibles(df, cfg.LIMITES_FISICOS)

    # ------------------------------------------------------------------
    titulo("PASO 4 - VARIABLES DETERMINISTAS (sin parametros aprendidos)")
    df = features.agregar_variables_temporales(df)
    print("  Creadas: Hora, Dia_semana, Mes, Anio, Es_fin_de_semana, Es_hora_pico, Estacion")
    df, mapa_clima = features.consolidar_weather_condition(df)
    n_orig = len(mapa_clima)
    n_grupos = mapa_clima["Grupo"].nunique()
    print(f"  Weather_Condition: {n_orig} categorias originales -> {n_grupos} grupos")
    df = features.agregar_objetivo_binario(df)
    tasa = df[cfg.COL_OBJETIVO].mean()
    print(f"  {cfg.COL_OBJETIVO} = Severity >= {cfg.SEVERIDAD_MINIMA_CRITICA}: "
          f"{df[cfg.COL_OBJETIVO].sum():,} criticos ({tasa * 100:.2f}%)")
    print(f"  Baseline de exactitud ('todo normal'): {(1 - tasa) * 100:.2f}%")

    df = cleaning.agregar_duracion(df)
    corr = df[["Duracion_min", cfg.COL_OBJETIVO]].corr().iloc[0, 1]
    print(f"  Duracion_min (solo trazabilidad, NO es predictor): "
          f"correlacion con {cfg.COL_OBJETIVO} = {corr:.3f}")

    for col in cfg.COLS_CATEGORICAS_CRUDAS + ["Estacion"]:
        df[col] = df[col].astype("category")

    # ------------------------------------------------------------------
    titulo("PASO 5 - OPTIMIZACION DE MEMORIA")
    df, antes, despues = cleaning.optimizar_memoria(df)
    print(f"Memoria antes: {antes:,.1f} MB | despues: {despues:,.1f} MB "
          f"(ahorro {(1 - despues / antes) * 100:.1f}%)")

    # ------------------------------------------------------------------
    titulo("PASO 6 - RESUMEN Y VERIFICACIONES")
    resumen = bitacora.a_dataframe()
    print(resumen.to_string(index=False))
    filas_finales = len(df)
    print(f"\nFilas en el alcance: {filas_alcance:,} | finales: {filas_finales:,} "
          f"({filas_finales / filas_alcance * 100:.2f}% conservado)")

    print(f"\n{cfg.COL_OBJETIVO} por anio (estabilidad dentro del alcance):")
    por_anio = df.groupby("Anio", observed=True)[cfg.COL_OBJETIVO].agg(["size", "mean"])
    for anio, fila in por_anio.iterrows():
        print(f"  {anio}: {int(fila['size']):>10,} filas | criticos {fila['mean'] * 100:5.2f}%")

    print(f"\nRequisito de la rubrica (mas de {cfg.MIN_REGISTROS_RUBRICA:,} registros): "
          f"{'CUMPLE' if filas_finales > cfg.MIN_REGISTROS_RUBRICA else 'NO CUMPLE'}")

    esperadas = set(cfg.DISCRETAS + cfg.CATEGORICAS + cfg.BOOLEANAS + cfg.COORDENADAS
                    + cfg.CLIMA_NUMERICO + cfg.INFO_POSTERIOR
                    + [cfg.COL_OBJETIVO, "Start_Time"])
    faltan = sorted(esperadas - set(df.columns))
    if faltan:
        raise RuntimeError(f"Faltan columnas que necesita el script 02: {faltan}")
    print("Verificacion: la tabla trae todas las columnas que necesita el script 02.")

    # La tabla se guarda ORDENADA por fecha: asi las posiciones de fila que
    # guarda el script 02 quedan alineadas con esta tabla.
    df = df.sort_values("Start_Time", kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    titulo("PASO 7 - GUARDADO")
    ruta = guardar_tabla(df, os.path.join(cfg.DIR_LIMPIO, "us_accidents_limpio"))
    print(f"Tabla limpia  -> {ruta} ({filas_finales:,} filas, {df.shape[1]} columnas)")

    ruta_bitacora = os.path.join(cfg.DIR_LIMPIO, "bitacora_limpieza.csv")
    resumen.to_csv(ruta_bitacora, index=False, encoding=cfg.ENCODING)
    print(f"Bitacora      -> {ruta_bitacora}")

    ruta_mapa = os.path.join(cfg.DIR_LIMPIO, "mapa_weather_condition.csv")
    mapa_clima.to_csv(ruta_mapa, index=False, encoding=cfg.ENCODING)
    print(f"Mapa de clima -> {ruta_mapa}")

    print("\n" + "=" * 70)
    print("LIMPIEZA COMPLETADA. Siguiente paso: python pipeline/02_preparar_datos.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
