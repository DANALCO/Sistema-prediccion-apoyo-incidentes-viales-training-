"""
=============================================================================
DIAGNOSTICO  -  el TIPO DE VIA (derivado de Street) ayuda a detectar criticos?
=============================================================================
Hipotesis: un choque en una interestatal afecta mas al trafico que uno en una
calle local, y el tipo de via se conoce al reportar el incidente. La columna
Street se descarto por cardinalidad como texto; aqui se convierte en pocas
categorias con una regla fija (accidentes.features.clasificar_via).

Que hace:
  [1] Alinea Street con las MISMAS filas ya preparadas (relee el CSV con la misma
      limpieza y comprueba, columna por columna, que coincide con la tabla limpia;
      si algo no coincide, se detiene). El resultado queda en cache.
  [2] Tabla de riesgo por tipo de via (train, val_ajuste y val_umbral).
  [3] A/B: el LightGBM ACTUAL contra el mismo modelo + el tipo de via, entrenados
      igual (mismos pesos y parametros), con varias semillas para separar el efecto
      real del azar de LightGBM.
  [4] Veredicto con intervalo bootstrap y el mismo criterio de relevancia del ajuste.

Protocolo: usa train y las 2 validaciones. NO lee el test (ni matrices, ni etiquetas,
ni posiciones). No modifica el pipeline: si ayuda, se incorpora despues, a proposito.

ESTADO: REGISTRO HISTORICO. El diagnostico dio AYUDA en las dos validaciones (PR-AUC en val_umbral
0.153 -> 0.190; recall al 5% de alertas 0.351 -> 0.415) y el tipo de via YA forma parte del pipeline
(features.agregar_tipo_via, llamado desde 01). Por eso este script se niega a correr sobre datos
preparados que ya lo traen: compararia "con" contra "con". Las tablas que produjo se conservan en
eda/figures/ para el informe.

Entrada : outputs/preparado/datos_<escenario>.npz (02), outputs/limpio/ (01), el CSV original
Salida  : eda/figures/tipo_via_riesgo.csv y tipo_via_ab.csv
Uso (desde la raiz del repositorio):
    python eda/diagnostico_tipo_via.py
    python eda/diagnostico_tipo_via.py --recargar      # vuelve a leer el CSV (si cambia la regla o la tabla limpia)
=============================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from accidentes import cleaning, features, tuning
from accidentes import config as cfg
from accidentes.ingestion import DataIngester
from accidentes.models import construir_modelo, probabilidad_critico
from accidentes.storage import guardar_tabla, leer_tabla
from accidentes.training import datos_de_entrenamiento
from accidentes.utils import Bitacora, titulo

RUTA_CACHE = os.path.join(cfg.DIR_PREPARADO, "street_alineada")


def leer_posiciones(escenario):
    """Posiciones (en la tabla limpia) de las dos validaciones. Las de test NO se leen."""
    with np.load(cfg.DIR_PREPARADO / f"datos_{escenario}.npz") as z:
        return {"val_ajuste": z["pos_val_ajuste"], "val_umbral": z["pos_val_umbral"]}


def construir_cache(n_filas):
    """Relee el CSV con la MISMA limpieza que el script 01 y verifica la alineacion."""
    print("    Releyendo el CSV con la misma limpieza que 01 (unos minutos)...")
    ingester = DataIngester(cfg.RUTA_CSV_ORIGINAL, list(dict.fromkeys(cfg.COLUMNAS_A_LEER + ["Street"])), cfg.ENCODING)
    df, _, _ = ingester.leer_alcance(cfg.FUENTE, cfg.FECHA_INICIO)
    bitacora = Bitacora()
    df = cleaning.normalizar_booleanas(df, cfg.INFRAESTRUCTURA)
    df = cleaning.convertir_fechas(df)
    df = cleaning.eliminar_fechas_invalidas(df, bitacora)
    df = cleaning.validar_objetivo(df, bitacora, cfg.COL_SEVERIDAD)
    # Mismo des-duplicado que 01, pero SIN contar Street: si se contara, podrian cambiar las filas.
    df = df[~df.drop(columns="Street").duplicated()].reset_index(drop=True)
    df = df.sort_values("Start_Time", kind="stable").reset_index(drop=True)
    critico = (df[cfg.COL_SEVERIDAD].to_numpy() >= cfg.SEVERIDAD_MINIMA_CRITICA).astype("int8")

    tabla = leer_tabla(os.path.join(cfg.DIR_LIMPIO, "us_accidents_limpio"))
    controles = {
        "mismo numero de filas": len(tabla) == len(df),
    }
    if controles["mismo numero de filas"]:
        controles["Start_Time identica"] = bool((tabla["Start_Time"].to_numpy() == df["Start_Time"].to_numpy()).all())
        controles["coordenadas identicas"] = bool(
            np.allclose(tabla["Start_Lat"].to_numpy("float64"), df["Start_Lat"].to_numpy("float64"), atol=1e-4, equal_nan=True)
            and np.allclose(tabla["Start_Lng"].to_numpy("float64"), df["Start_Lng"].to_numpy("float64"), atol=1e-4, equal_nan=True))
        # etiquetas: solo train + validaciones (las filas de test no se comparan)
        controles["etiquetas identicas (train y validaciones)"] = bool(
            (tabla[cfg.COL_OBJETIVO].to_numpy()[:n_filas] == critico[:n_filas]).all())
    for nombre, ok in controles.items():
        print(f"    {'OK   ' if ok else 'FALLA'} {nombre}")
    if not all(controles.values()):
        sys.exit("\nLas filas releidas NO coinciden con la tabla limpia: las posiciones del script 02 no se pueden "
                 "usar. Ejecute 01 y 02 de nuevo con el mismo CSV y repita.")
    del tabla
    # Solo se conserva hasta el final de val_umbral: el resto (test) no se guarda ni se usa.
    salida = pd.DataFrame({"Street": df["Street"].iloc[:n_filas].astype("object").to_numpy(),
                           cfg.COL_OBJETIVO: critico[:n_filas]})
    guardar_tabla(salida, RUTA_CACHE)
    return salida


def cargar_calles(posiciones, y_train, y_ajuste, y_umbral, recargar):
    n_train = len(y_train)
    pos_aj, pos_um = posiciones["val_ajuste"], posiciones["val_umbral"]
    if not (np.array_equal(pos_aj, np.arange(n_train, n_train + len(pos_aj)))
            and np.array_equal(pos_um, np.arange(pos_aj[-1] + 1, pos_aj[-1] + 1 + len(pos_um)))):
        sys.exit("Las particiones no son contiguas en la tabla ordenada por fecha; revise 02.")
    n_filas = int(pos_um[-1]) + 1

    cache = None
    if not recargar and any(os.path.exists(RUTA_CACHE + e) for e in (".parquet", ".csv")):
        cache = leer_tabla(RUTA_CACHE)
        y = cache[cfg.COL_OBJETIVO].to_numpy()
        coincide = (len(cache) == n_filas and (y[:n_train] == y_train).all()
                    and (y[pos_aj] == y_ajuste).all() and (y[pos_um] == y_umbral).all())
        print("    Cache de Street encontrada y alineada con las etiquetas." if coincide
              else "    La cache no coincide con las etiquetas actuales: se reconstruye.")
        if not coincide:
            cache = None
    if cache is None:
        cache = construir_cache(n_filas)
    return features.tipos_de_via(cache["Street"]).to_numpy(), cache["Street"]


def tabla_de_riesgo(tipos, indices, y):
    """Filas y % de criticos por tipo de via en cada particion."""
    base = {p: y[p].mean() for p in y}
    filas = []
    for tipo in features.TIPOS_VIA:
        fila = {"Tipo_via": tipo}
        for p in ("train", "val_ajuste", "val_umbral"):
            m = tipos[indices[p]] == tipo
            fila[f"Filas_{p}"] = int(m.sum())
            fila[f"Pct_filas_{p}"] = m.mean() * 100
            fila[f"Tasa_critico_{p}_%"] = y[p][m].mean() * 100 if m.any() else np.nan
        fila["Veces_tasa_base_train"] = fila["Tasa_critico_train_%"] / (base["train"] * 100) if fila["Filas_train"] else np.nan
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("Tasa_critico_train_%", ascending=False, na_position="last")


def con_tipo_via(X, tipos):
    """Anade una columna 0/1 por tipo de via a la matriz ya preparada."""
    extra = np.stack([(tipos == t) for t in features.TIPOS_VIA], axis=1).astype("float32")
    return np.hstack([X.astype("float32", copy=False), extra])


def entrenar_y_medir(X, y, peso, particiones, semilla):
    modelo = construir_modelo("LightGBM", cfg.HIPERPARAMETROS, semilla, False, peso_criticos=peso)
    modelo.fit(X, y)
    return modelo, {p: probabilidad_critico(modelo, Xp) for p, (Xp, _) in particiones.items()}


def main():
    parser = argparse.ArgumentParser(description="Diagnostico: el tipo de via ayuda a detectar criticos?")
    parser.add_argument("--escenario", choices=list(cfg.ESCENARIOS), default="tiempo_real")
    parser.add_argument("--estrategia", choices=[e for e in cfg.ESTRATEGIAS_BALANCEO if e != "SMOTENC"],
                        default=cfg.ESTRATEGIA_FINAL)
    parser.add_argument("--semillas", type=int, default=3, help="semillas por variante")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--alertas", type=float, default=0.05)
    parser.add_argument("--recargar", action="store_true", help="vuelve a leer el CSV original")
    args = parser.parse_args()

    ruta_nombres = cfg.DIR_PREPARADO / f"features_{args.escenario}.txt"
    if ruta_nombres.exists() and any(n.startswith("Tipo_via_") for n in
                                     ruta_nombres.read_text(encoding=cfg.ENCODING).split("\n")):
        sys.exit("El tipo de via YA es parte del pipeline (features_%s.txt trae columnas Tipo_via_*): este "
                 "diagnostico compara 'sin' contra 'con' y ya no es posible.\nSus resultados historicos "
                 "estan en eda/figures/tipo_via_riesgo.csv y tipo_via_ab.csv." % args.escenario)

    titulo("DIAGNOSTICO: TIPO DE VIA (derivado de Street)")
    print("Particiones usadas: train, val_ajuste, val_umbral.  Test: NO SE LEE.\n")

    print("[1] Alineacion de Street con las filas ya preparadas")
    datos = tuning.cargar_sin_test(args.escenario, args.estrategia)
    X_ent, y_ent, peso = datos_de_entrenamiento(datos, args.estrategia)
    (X_aj, y_aj), (X_um, y_um) = datos["val_ajuste"], datos["val_umbral"]
    posiciones = leer_posiciones(args.escenario)
    tipos, calles = cargar_calles(posiciones, y_ent, y_aj, y_um, args.recargar)
    n_train = len(y_ent)
    indices = {"train": np.arange(n_train), "val_ajuste": posiciones["val_ajuste"],
               "val_umbral": posiciones["val_umbral"]}
    y = {"train": y_ent, "val_ajuste": y_aj, "val_umbral": y_um}

    # ------------------------------------------------------------------
    titulo("[2] RIESGO POR TIPO DE VIA (% de incidentes criticos dentro de cada tipo)")
    riesgo = tabla_de_riesgo(tipos, indices, y)
    vista = riesgo[["Tipo_via", "Filas_train", "Pct_filas_train", "Tasa_critico_train_%", "Veces_tasa_base_train",
                    "Tasa_critico_val_ajuste_%", "Tasa_critico_val_umbral_%"]].copy()
    vista.columns = ["Tipo_via", "Filas_train", "%_filas", "Critico_train_%", "x_tasa_base", "Critico_ajuste_%", "Critico_umbral_%"]
    print(vista.round(2).to_string(index=False))
    print(f"\nTasa base en train: {y_ent.mean() * 100:.2f}% | val_ajuste: {y_aj.mean() * 100:.2f}% | val_umbral: {y_um.mean() * 100:.2f}%")
    print("Si el orden de los tipos se mantiene entre train y las validaciones, el patron es estable.")
    otros = pd.Series(calles.iloc[:n_train][tipos[:n_train] == "Otro"]).value_counts().head(15)
    if len(otros):
        pct_otro = (tipos[:n_train] == "Otro").mean() * 100
        print(f"\n'Otro' = {pct_otro:.1f}% de train (reglas sin cubrir). Los 15 nombres mas frecuentes:")
        for nombre, n in otros.items():
            print(f"    {n:>8,}  {nombre}")
    os.makedirs(cfg.DIR_EDA, exist_ok=True)
    riesgo.round(4).to_csv(cfg.DIR_EDA / "tipo_via_riesgo.csv", index=False, encoding=cfg.ENCODING)

    # ------------------------------------------------------------------
    titulo(f"[3] A/B: LightGBM ACTUAL vs LightGBM + tipo de via  ({args.semillas} semillas por variante)")
    print(f"Mismos datos, parametros de config.py y pesos de clase (x{peso:.1f}); solo cambia la presencia de las columnas de tipo de via.\n")
    particiones_a = {"val_ajuste": (X_aj, y_aj), "val_umbral": (X_um, y_um)}
    particiones_b = {"val_ajuste": (con_tipo_via(X_aj, tipos[indices["val_ajuste"]]), y_aj),
                     "val_umbral": (con_tipo_via(X_um, tipos[indices["val_umbral"]]), y_um)}
    X_b = con_tipo_via(X_ent, tipos[:n_train])
    puntajes = {"A": {p: [] for p in particiones_a}, "B": {p: [] for p in particiones_a}}
    filas, ganancia_nueva = [], None
    for i in range(args.semillas):
        semilla = cfg.SEMILLA + i
        for variante, X, parts in (("A", X_ent, particiones_a), ("B", X_b, particiones_b)):
            t0 = time.time()
            modelo, p = entrenar_y_medir(X, y_ent, peso, parts, semilla)
            for part, valores in p.items():
                puntajes[variante][part].append(valores)
            if variante == "B" and i == 0:
                g = modelo.booster_.feature_importance("gain")
                ganancia_nueva = g[-len(features.TIPOS_VIA):].sum() / g.sum() * 100
            filas.append({"Variante": "A actual" if variante == "A" else "B + tipo de via", "Semilla": semilla,
                          "PR_AUC_val_ajuste": average_precision_score(y_aj, p["val_ajuste"]),
                          "PR_AUC_val_umbral": average_precision_score(y_um, p["val_umbral"]),
                          "Segundos": time.time() - t0})
            print(f"    semilla {semilla} | {filas[-1]['Variante']:<16} PR-AUC ajuste {filas[-1]['PR_AUC_val_ajuste']:.4f} | "
                  f"umbral {filas[-1]['PR_AUC_val_umbral']:.4f} | {filas[-1]['Segundos']:.0f} s")
            del modelo
    ab = pd.DataFrame(filas)

    # ------------------------------------------------------------------
    titulo("[4] RESULTADO (puntajes promediados entre semillas)")
    prom = {v: {p: np.mean(puntajes[v][p], axis=0) for p in particiones_a} for v in puntajes}
    resumen, rel, est = [], [], []
    for part, y_part in (("val_ajuste", y_aj), ("val_umbral", y_um)):
        a, b = prom["A"][part], prom["B"][part]
        delta, lo, hi = tuning.diferencia_pr_auc_bootstrap(y_part, b, a, args.n_bootstrap, cfg.SEMILLA)
        rec_a, pre_a = tuning.recall_precision_en_fraccion(y_part, a, args.alertas)
        rec_b, pre_b = tuning.recall_precision_en_fraccion(y_part, b, args.alertas)
        resumen.append({"Particion": part,
                        "PR_AUC_A": average_precision_score(y_part, a), "PR_AUC_B": average_precision_score(y_part, b),
                        "Delta_PR_AUC": delta, "IC95_inf": lo, "IC95_sup": hi,
                        "ROC_AUC_A": roc_auc_score(y_part, a), "ROC_AUC_B": roc_auc_score(y_part, b),
                        "Recall_A": rec_a, "Recall_B": rec_b, "Precision_A": pre_a, "Precision_B": pre_b})
        rel.append(lo > 0 and delta >= tuning.MEJORA_MINIMA)
        est.append(lo > 0)
        print(f"  {part:<11} PR-AUC {resumen[-1]['PR_AUC_A']:.4f} -> {resumen[-1]['PR_AUC_B']:.4f} "
              f"| diferencia {delta:+.4f} (IC 95%: {lo:+.4f} a {hi:+.4f}) "
              f"| recall al marcar {args.alertas:.0%}: {rec_a:.3f} -> {rec_b:.3f}")
        print(f"              => {tuning.veredicto(delta, lo, hi)}")
    seed_sd = ab.groupby("Variante")[["PR_AUC_val_ajuste", "PR_AUC_val_umbral"]].std().mean().mean()
    print(f"\n  Variacion entre semillas (desviacion tipica del PR-AUC): ~{seed_sd:.4f}")
    print(f"  Parte de la ganancia total del modelo B que usan las columnas de tipo de via: {ganancia_nueva:.1f}%")
    print("  Referencia: la fila 'A actual' debe parecerse a la fila 'actual' de LightGBM en el ajuste de hiperparametros.")
    pd.DataFrame(resumen).round(5).to_csv(cfg.DIR_EDA / "tipo_via_ab.csv", index=False, encoding=cfg.ENCODING)

    titulo("CONCLUSION")
    if all(rel):
        print("AYUDA: el tipo de via mejora el ranking de forma relevante en AMBAS validaciones.")
        print("Siguiente paso: incorporarlo al pipeline (features.py, 02, formulario de la app) y reentrenar 03 -> 04 -> 06.")
    elif any(est):
        print("AYUDA POCO O DE FORMA INCONSISTENTE: hay una diferencia estadistica pero pequena o que no se repite")
        print("en las dos validaciones. No justifica agregar un campo al formulario por si sola.")
    else:
        print("NO AYUDA: con esta regla de tipo de via el modelo no detecta mejor. El limite sigue siendo la informacion")
        print("disponible. Es un resultado valido y se documenta como hipotesis descartada.")
    print(f"\nTablas: {cfg.DIR_EDA / 'tipo_via_riesgo.csv'} y tipo_via_ab.csv")


if __name__ == "__main__":
    main()
