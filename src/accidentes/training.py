"""
ENTRENAMIENTO DE UN ESCENARIO (de las matrices a los resultados)
=============================================================================
Responsabilidad unica: coordinar, para UN escenario, el flujo
    4 modelos base -> probabilidades -> calibracion -> integracion -> umbral
    -> metricas -> vigilancia -> guardado
usando los modulos models, ensemble y evaluation (que son los que saben HACER
cada cosa; aqui solo se ordenan los pasos).

QUE DATOS USA CADA PASO (regla de oro: nada se aprende con datos posteriores)
    train (SMOTENC)  entrena los 4 modelos base
    val_ajuste       calibracion de Platt, pesos del promedio y stacking
    val_umbral       umbral de decision y franja de vigilancia
    test             SOLO se mide. Nunca se usa para decidir nada.

Los hiperparametros estan fijos en config.py. No se deben ajustar mirando la
columna 'test': eso convertiria el test en un segundo conjunto de validacion.
=============================================================================
"""

import json
import time

import joblib
import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.ensemble import (Apilador, CalibradorPlatt, asignar_zona,
                                 pesos_por_pr_auc, seleccionar_umbral,
                                 umbrales_vigilancia)
from accidentes.evaluation import (metricas_en_umbral, metricas_independientes,
                                   puntos_rubrica, tabla_zonas)
from accidentes.models import construir_modelo, probabilidad_critico
from accidentes.utils import titulo

PARTICIONES_PUNTAJE = ("train_real", "val_ajuste", "val_umbral", "test")
PARTICIONES_REPORTE = ("train_real", "val_umbral", "test")


# ----------------------------------------------------------------------------
def cargar_datos_escenario(nombre):
    ruta = cfg.DIR_PREPARADO / f"datos_{nombre}.npz"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute antes pipeline/02_preparar_datos.py")
    z = np.load(ruta)
    datos = {"train_bal": (z["X_train_bal"], z["y_train_bal"])}
    if "X_train_real" in z.files:
        datos["train_real"] = (z["X_train_real"], z["y_train_real"])
    for particion in ("val_ajuste", "val_umbral", "test"):
        datos[particion] = (z[f"X_{particion}"], z[f"y_{particion}"])
    return datos


def entrenar_modelos_base(X, y, directorio, peso_criticos=None):
    modelos = {}
    for nombre in cfg.NOMBRES_MODELOS:
        t0 = time.time()
        modelo = construir_modelo(nombre, cfg.HIPERPARAMETROS, cfg.SEMILLA, cfg.XGBOOST_USAR_GPU,
                                  peso_criticos=peso_criticos)
        modelo.fit(X, y)
        modelos[nombre] = modelo
        joblib.dump(modelo, directorio / f"modelo_{nombre}.joblib",
                    compress=3 if nombre == "Random_Forest" else 0)
        print(f"    {nombre:<22} entrenado en {time.time() - t0:6.0f} s")
    return modelos


def probabilidades_crudas(modelos, datos):
    p = {nombre: {} for nombre in modelos}
    for nombre, modelo in modelos.items():
        for particion in PARTICIONES_PUNTAJE:
            if particion in datos:
                p[nombre][particion] = probabilidad_critico(modelo, datos[particion][0])
    return p


def construir_puntajes(p_crudas, y_val_ajuste, particiones):
    """Calibra cada modelo y construye los puntajes de los 7 metodos."""
    nombres = list(p_crudas)
    calibradores = {n: CalibradorPlatt().fit(p_crudas[n]["val_ajuste"], y_val_ajuste) for n in nombres}
    calibrada = {n: {p: calibradores[n].transformar(p_crudas[n][p]) for p in particiones}
                 for n in nombres}

    puntajes = {n: calibrada[n] for n in nombres}
    pesos = pesos_por_pr_auc(y_val_ajuste, {n: calibrada[n]["val_ajuste"] for n in nombres})
    puntajes["Promedio"] = {p: np.mean([calibrada[n][p] for n in nombres], axis=0) for p in particiones}
    puntajes["Promedio_ponderado"] = {p: sum(pesos[n] * calibrada[n][p] for n in nombres)
                                      for p in particiones}

    apilador = Apilador(nombres).fit({n: p_crudas[n]["val_ajuste"] for n in nombres}, y_val_ajuste)
    puntajes["Stacking"] = {p: apilador.transformar({n: p_crudas[n][p] for n in nombres})
                            for p in particiones}
    return puntajes, calibradores, apilador, pesos


def evaluar_metodos(puntajes, datos):
    """Umbral por metodo (en val_umbral) y metricas en train_real, val_umbral y test."""
    umbrales, filas = {}, []
    for metodo in cfg.METODOS:
        s = puntajes[metodo]
        umbral = seleccionar_umbral(s["val_umbral"], datos["val_umbral"][1],
                                    cfg.EXACTITUD_MINIMA_VALIDACION)
        umbrales[metodo] = umbral
        for particion in PARTICIONES_REPORTE:
            if particion not in datos:
                continue
            y = datos[particion][1]
            fila = {"Metodo": metodo, "Particion": particion, "Umbral": umbral}
            fila.update(metricas_en_umbral(y, s[particion], umbral))
            fila.update(metricas_independientes(y, s[particion]))
            filas.append(fila)
    return pd.DataFrame(filas), umbrales


def calcular_zonas(puntajes, datos, umbral):
    s = puntajes[cfg.METODO_FINAL]
    inferior, superior = umbrales_vigilancia(
        s["val_umbral"], umbral, cfg.FACTOR_VIGILANCIA_INFERIOR, cfg.FACTOR_VIGILANCIA_SUPERIOR)
    tablas = []
    for particion in ("val_umbral", "test"):
        zonas = asignar_zona(s[particion], inferior, umbral, superior)
        t = tabla_zonas(datos[particion][1], zonas)
        t.insert(0, "Particion", particion)
        tablas.append(t)
    return inferior, superior, pd.concat(tablas, ignore_index=True)


def datos_de_entrenamiento(datos, estrategia):
    """(X, y, peso_criticos) que corresponden a cada estrategia de balanceo."""
    if estrategia == "SMOTENC":
        X, y = datos["train_bal"]
        return X, y, None
    if "train_real" not in datos:
        raise ValueError(f"La estrategia {estrategia} necesita el train real (GUARDAR_TRAIN_REAL = True en config.py)")
    X, y = datos["train_real"]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if estrategia == "Sin_balancear":
        return X, y, None
    if estrategia == "Pesos_moderados":
        return X, y, float(np.sqrt(n_neg / n_pos))
    if estrategia == "Pesos_completos":
        return X, y, float(n_neg / n_pos)
    raise ValueError(f"Estrategia desconocida: {estrategia}. Opciones: {cfg.ESTRATEGIAS_BALANCEO}")


def comparar_balanceo(datos):
    """LightGBM con cada estrategia de balanceo, medido SOLO en val_umbral
    (el test no se toca para decidir esto). Compare sobre todo PR_AUC y ROC_AUC:
    el umbral se elige en esta misma particion para todas las estrategias."""
    y_aj, y_um = datos["val_ajuste"][1], datos["val_umbral"][1]
    filas = []
    for estrategia in cfg.ESTRATEGIAS_BALANCEO:
        try:
            X, y, peso = datos_de_entrenamiento(datos, estrategia)
        except ValueError:
            continue
        t0 = time.time()
        modelo = construir_modelo("LightGBM", cfg.HIPERPARAMETROS, cfg.SEMILLA, peso_criticos=peso)
        modelo.fit(X, y)
        calibrador = CalibradorPlatt().fit(probabilidad_critico(modelo, datos["val_ajuste"][0]), y_aj)
        s = calibrador.transformar(probabilidad_critico(modelo, datos["val_umbral"][0]))
        umbral = seleccionar_umbral(s, y_um, cfg.EXACTITUD_MINIMA_VALIDACION)
        fila = {"Entrenamiento": estrategia, "Peso_criticos": peso if peso else 1.0,
                "Filas_entrenamiento": len(y)}
        fila.update(metricas_en_umbral(y_um, s, umbral))
        fila.update(metricas_independientes(y_um, s))
        fila["Segundos"] = time.time() - t0
        filas.append(fila)
        print(f"    {estrategia:<16} PR-AUC {fila['PR_AUC']:.4f} | ROC-AUC {fila['ROC_AUC']:.4f} "
              f"| {fila['Segundos']:.0f} s")
    columnas = ["Entrenamiento", "Peso_criticos", "Filas_entrenamiento", "PR_AUC", "ROC_AUC",
                "Recall", "Precision", "Exactitud", "Alertas_%"]
    return pd.DataFrame(filas)[columnas]


def comparar_balanceo_escenario(nombre):
    """Solo la comparacion de balanceo (unos minutos), sin reentrenar los 4 modelos."""
    titulo(f"COMPARACION DEL BALANCEO - {nombre}  (LightGBM, medido en val_umbral)")
    directorio = cfg.DIR_MODELOS / nombre
    directorio.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos_escenario(nombre)
    tabla = comparar_balanceo(datos)
    print()
    print(tabla.round(4).to_string(index=False))
    mejor = tabla.sort_values("PR_AUC", ascending=False).iloc[0]
    print(f"\nMejor PR-AUC en val_umbral: {mejor['Entrenamiento']} ({mejor['PR_AUC']:.4f}).")
    print("Para usarla en el modelo final: cambie ESTRATEGIA_FINAL en config.py (o use --estrategia).")
    tabla.to_csv(directorio / "comparacion_balanceo.csv", index=False, encoding=cfg.ENCODING)
    return tabla


# ----------------------------------------------------------------------------
def guardar_resultados(directorio, nombre, tabla, zonas, balanceo, calibradores, apilador,
                       pesos, umbral, inferior, superior, puntajes, datos):
    tabla.to_csv(directorio / "tabla_comparativa.csv", index=False, encoding=cfg.ENCODING)
    zonas.to_csv(directorio / "zonas_vigilancia.csv", index=False, encoding=cfg.ENCODING)
    if balanceo is not None:
        balanceo.to_csv(directorio / "comparacion_balanceo.csv", index=False, encoding=cfg.ENCODING)
    joblib.dump(calibradores, directorio / "calibradores.joblib")
    joblib.dump(apilador, directorio / "apilador.joblib")

    decision = {
        "escenario": nombre,
        "metodo_final": cfg.METODO_FINAL,
        "estrategia_balanceo": cfg.ESTRATEGIA_FINAL,
        "umbral": umbral,
        "umbral_vigilancia_inferior": inferior,
        "umbral_vigilancia_superior": superior,
        "exactitud_minima_validacion": cfg.EXACTITUD_MINIMA_VALIDACION,
        "coeficientes_apilador": apilador.coeficientes,
        "pesos_promedio_ponderado": pesos,
        "calibradores_platt": {n: {"a": c.a_, "b": c.b_} for n, c in calibradores.items()},
    }
    with open(directorio / "decision.json", "w", encoding=cfg.ENCODING) as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    arreglos = {}
    for metodo, por_particion in puntajes.items():
        for particion, valores in por_particion.items():
            arreglos[f"{metodo}__{particion}"] = valores.astype("float32")
    for particion in ("val_ajuste", "val_umbral", "test"):
        arreglos[f"y__{particion}"] = datos[particion][1]
    np.savez_compressed(directorio / "puntajes.npz", **arreglos)


def imprimir_resumen(nombre, tabla, apilador, pesos, umbral, inferior, superior, zonas, balanceo):
    test = tabla[tabla["Particion"] == "test"].set_index("Metodo").loc[list(cfg.METODOS)]
    titulo(f"RESULTADOS EN TEST - {nombre}")
    print("Exactitud va SIEMPRE junto al baseline (exactitud de decir 'todo normal').")
    print(f"Umbral de cada metodo: mayor recall con exactitud >= "
          f"{cfg.EXACTITUD_MINIMA_VALIDACION:.0%} en val_umbral.\n")
    mostrar = test[["Exactitud", "Baseline", "Recall", "Precision", "F1", "Alertas_%",
                    "PR_AUC", "ROC_AUC", "Balanced_acc", "Umbral"]].round(4)
    print(mostrar.to_string())

    titulo(f"TRAIN (real) CONTRA TEST - {nombre}  (diferencias grandes = sobreajuste)")
    if (tabla["Particion"] == "train_real").any():
        tr = tabla[tabla["Particion"] == "train_real"].set_index("Metodo").loc[list(cfg.METODOS)]
        comp = pd.DataFrame({
            "Exactitud_train": tr["Exactitud"], "Exactitud_test": test["Exactitud"],
            "F1_train": tr["F1"], "F1_test": test["F1"],
            "PR_AUC_train": tr["PR_AUC"], "PR_AUC_test": test["PR_AUC"],
        }).round(4)
        print(comp.to_string())
    else:
        print("(no se guardo el train real; GUARDAR_TRAIN_REAL = False)")

    titulo(f"INTEGRACION - {nombre}")
    print("Coeficientes del STACKING (su peso sobre cada modelo, en escala logit):")
    for k, v in apilador.coeficientes.items():
        print(f"    {k:<22} {v:+.4f}")
    print("\nPesos del PROMEDIO PONDERADO (proporcionales al PR-AUC en val_ajuste):")
    for k, v in pesos.items():
        print(f"    {k:<22} {v:.4f}")

    titulo(f"VIGILANCIA - {nombre}  (metodo final: {cfg.METODO_FINAL})")
    print(f"Umbral de decision (critico si puntaje >= tau): {umbral:.5f}")
    print(f"Franja de vigilancia: [{inferior:.5f}, {superior:.5f}]\n")
    if inferior == umbral == superior:
        print("AVISO: la franja de vigilancia quedo vacia porque el umbral marca muy pocos incidentes.\n")
    print(zonas.round(2).to_string(index=False))
    print("\n'Pct_critico_en_zona' debe CRECER de Normal a Critico; si no, la franja no aporta.")

    if balanceo is not None:
        titulo(f"BALANCEO - {nombre}  (LightGBM, medido en val_umbral; estrategia usada: {cfg.ESTRATEGIA_FINAL})")
        print(balanceo.round(4).to_string(index=False))
        print("\nOjo: el umbral se eligio en esta misma particion; compare sobre todo PR_AUC y ROC_AUC.")

    final = test.loc[cfg.METODO_FINAL]
    puntos = puntos_rubrica(final["Exactitud"], cfg.NIVELES_RUBRICA)
    titulo(f"CRITERIO DE EXACTITUD DE LA RUBRICA - {nombre}")
    print(f"Modelo final ({cfg.METODO_FINAL}) en test: exactitud {final['Exactitud']:.4f} "
          f"| baseline {final['Baseline']:.4f}")
    print(f"Recall {final['Recall']:.3f} | Precision {final['Precision']:.3f} | F1 {final['F1']:.3f}")
    if final["Recall"] < 0.02 or final["Alertas_%"] < 0.1:
        print("\nADVERTENCIA: el modelo final casi no marca incidentes como criticos. Su exactitud")
        print("es la de 'todo normal' y NO demuestra que el modelo sirva. No la reporte como logro.")
        print("Revise EXACTITUD_MINIMA_VALIDACION en config.py y los resultados de validacion.")
    else:
        print(f"Segun solo la exactitud: {puntos} pts "
              f"(la rubrica exige ademas 8/9/10 funcionalidades).")
        print("Reporte siempre baseline, recall, F1 y PR-AUC: la exactitud sola no dice si el modelo sirve.")


def entrenar_escenario(nombre):
    """Flujo completo de un escenario. Devuelve la tabla comparativa."""
    titulo(f"ESCENARIO: {nombre}")
    directorio = cfg.DIR_MODELOS / nombre
    directorio.mkdir(parents=True, exist_ok=True)
    t_inicio = time.time()

    print("\n[1] Carga de las matrices preparadas")
    datos = cargar_datos_escenario(nombre)
    for particion, (X, y) in datos.items():
        print(f"    {particion:<11} {X.shape[0]:>10,} filas x {X.shape[1]} columnas | "
              f"criticos {y.mean() * 100:5.2f}%")

    X_ent, y_ent, peso = datos_de_entrenamiento(datos, cfg.ESTRATEGIA_FINAL)
    detalle_peso = f" | peso de los criticos x{peso:.1f}" if peso else ""
    print(f"\n[2] Entrenamiento de los 4 modelos base (estrategia: {cfg.ESTRATEGIA_FINAL}; "
          f"{len(y_ent):,} filas{detalle_peso})")
    modelos = entrenar_modelos_base(X_ent, y_ent, directorio, peso)

    print("\n[3] Probabilidades crudas")
    p_crudas = probabilidades_crudas(modelos, datos)
    particiones = [p for p in PARTICIONES_PUNTAJE if p in datos]

    print("\n[4] Calibracion y combinacion (con val_ajuste)")
    puntajes, calibradores, apilador, pesos = construir_puntajes(
        p_crudas, datos["val_ajuste"][1], particiones)

    print("\n[5] Umbral por metodo (con val_umbral) y metricas")
    tabla, umbrales = evaluar_metodos(puntajes, datos)
    tabla.insert(0, "Estrategia", cfg.ESTRATEGIA_FINAL)
    umbral = umbrales[cfg.METODO_FINAL]

    print("\n[6] Franja de vigilancia")
    inferior, superior, zonas = calcular_zonas(puntajes, datos, umbral)

    balanceo = None
    if cfg.COMPARAR_BALANCEO:
        print("\n[7] Comparacion de estrategias de balanceo (LightGBM, en val_umbral)")
        balanceo = comparar_balanceo(datos)

    guardar_resultados(directorio, nombre, tabla, zonas, balanceo, calibradores, apilador,
                       pesos, umbral, inferior, superior, puntajes, datos)
    imprimir_resumen(nombre, tabla, apilador, pesos, umbral, inferior, superior, zonas, balanceo)
    print(f"\nTiempo del escenario: {(time.time() - t_inicio) / 60:.1f} min | salidas en {directorio}")
    return tabla
