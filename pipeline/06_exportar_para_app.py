"""
=============================================================================
SCRIPT 06  -  EXPORTAR LOS ARTEFACTOS PARA LA APLICACION WEB
=============================================================================
La aplicacion vive en OTRO repositorio y solo necesita los artefactos finales
del modelo. Este script los reune, los VERIFICA y deja constancia de sus
versiones y tamanos.

Entrada : outputs/preparado/, outputs/modelos/<escenario>/  (scripts 02, 03 y 04),
          shap/tablas/ y eda/figures/ (scripts 04, 05 y patrones; opcionales)
Salida  : outputs/exportacion/<escenario>/        paquete COMPLETO (no se versiona)
          models/<escenario>/                      artefactos <= --max-mb (se versionan)
          models/reportes/                         tablas pequenas para los paneles de la app
          models/manifest_<escenario>.json         versiones, tamanos, SHA-256, verificacion

PASOS
  1. Reune los archivos con los nombres que espera accidentes.inference.
  2. VERIFICA el paquete: carga PredictorIncidentes, predice 2,000 incidentes de
     test partiendo SOLO de los campos del formulario y compara con los puntajes
     del entrenamiento; prueba campos vacios y valores imposibles; comprueba la
     explicacion. Si algo no coincide, NO publica nada en models/.
  3. Genera la CURVA DE UMBRAL (alertas, exactitud, recall, precision, falsas
     alarmas y criticos perdidos en cada umbral) para el control de umbral de la
     app, y la valida contra la tabla del script 03.
  4. Copia a models/ lo que cabe en GitHub (limite duro: 100 MB por archivo).
     Lo mas pesado queda en outputs/exportacion/ para publicarlo como *Release*.
  5. Escribe el manifiesto.

Uso (desde la raiz del repositorio):
    python pipeline/06_exportar_para_app.py
    python pipeline/06_exportar_para_app.py --max-mb 30
=============================================================================
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.evaluation import curva_de_umbral
from accidentes.features import TIPOS_VIA
from accidentes.inference import COLUMNAS_ENTRADA, PredictorIncidentes
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

PAQUETES = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "joblib", "shap", "imbalanced-learn"]
TOLERANCIA = 1e-5
# La curva se calcula con los puntajes guardados en float32 (puntajes.npz). Un empate de
# redondeo puede mover UNA alerta respecto a la tabla del script 03 (que usa float64):
# eso cambia las metricas en ~1e-5 a 1e-4. Un error de formula las movera mucho mas.
TOLERANCIA_CURVA = 1e-3
FRACCIONES_MUESTRA = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def versiones():
    salida = {"python": sys.version.split()[0]}
    for p in PAQUETES:
        try:
            salida[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            salida[p] = "no instalado"
    return salida


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def archivos_del_modelo(esc):
    """(origen, nombre en el paquete, obligatorio)"""
    m = cfg.DIR_MODELOS / esc
    lista = [
        (cfg.DIR_PREPARADO / f"preprocesador_{esc}.joblib", "preprocesador.joblib", True),
        (cfg.DIR_PREPARADO / f"features_{esc}.txt", "features.txt", True),
        (m / "apilador.joblib", "apilador.joblib", True),
        (m / "decision.json", "decision.json", True),
        (m / "explicador_estado.json", "explicador_estado.json", False),
    ]
    lista += [(m / f"modelo_{n}.joblib", f"modelo_{n}.joblib", True) for n in cfg.NOMBRES_MODELOS]
    return lista


def reportes(esc):
    """Tablas pequenas que alimentan los paneles de la aplicacion (todas opcionales)."""
    return [
        (cfg.DIR_MODELOS / esc / "tabla_comparativa.csv", f"tabla_comparativa_{esc}.csv"),
        (cfg.DIR_MODELOS / esc / "zonas_vigilancia.csv", f"zonas_vigilancia_{esc}.csv"),
        (cfg.DIR_MODELOS / esc / "comparacion_balanceo.csv", f"comparacion_balanceo_{esc}.csv"),
        (cfg.DIR_MODELOS / esc / "curva_umbral.csv", f"curva_umbral_{esc}.csv"),
        (cfg.DIR_MODELOS / "perdida_tiempo_real.csv", "perdida_tiempo_real.csv"),
        (cfg.DIR_SHAP_TABLAS / f"importancia_global_{esc}.csv", f"importancia_global_{esc}.csv"),
        (cfg.DIR_SHAP_TABLAS / f"importancia_por_grupo_{esc}.csv", f"importancia_por_grupo_{esc}.csv"),
        (cfg.DIR_EDA / "patrones_de_riesgo.csv", "patrones_de_riesgo.csv"),
        (cfg.DIR_EDA / f"errores_por_segmento_{esc}.csv", f"errores_por_segmento_{esc}.csv"),
        (cfg.DIR_EDA / f"errores_estabilidad_mensual_{esc}.csv", f"errores_estabilidad_mensual_{esc}.csv"),
    ]


def verificar(paquete, esc, n_incidentes):
    """Prueba de extremo a extremo: formulario -> riesgo, comparado con el entrenamiento."""
    resultados, ok = {}, True
    predictor = PredictorIncidentes.cargar(paquete, esc)

    datos = np.load(cfg.DIR_PREPARADO / f"datos_{esc}.npz")
    pos_test = datos["pos_test"]
    puntajes = np.load(cfg.DIR_MODELOS / esc / "puntajes.npz")[f"{cfg.METODO_FINAL}__test"]
    rng = np.random.default_rng(cfg.SEMILLA)
    elegidos = rng.choice(len(pos_test), min(n_incidentes, len(pos_test)), replace=False)
    tabla = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))
    entrada = tabla[COLUMNAS_ENTRADA].iloc[pos_test[elegidos]].reset_index(drop=True)
    del tabla

    # 1) Mismos resultados que el entrenamiento, partiendo solo de los campos del formulario
    res = predictor.predecir(entrada)
    error = np.abs(res["Probabilidad"].to_numpy() - puntajes[elegidos])
    misma_decision = bool(((puntajes[elegidos] >= predictor.decision["umbral"]) == res["Es_critico"].to_numpy()).all())
    resultados["error_maximo_probabilidad"] = float(error.max())
    resultados["incidentes_verificados"] = int(len(entrada))
    paso = error.max() < TOLERANCIA and misma_decision
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} prediccion desde el formulario = puntaje del entrenamiento "
          f"(error maximo {error.max():.2e}, misma decision: {misma_decision}, {len(entrada):,} incidentes)")

    # 2) Aportes por modelo suman el logit final
    logit_final = np.log(res["Probabilidad"] / (1 - res["Probabilidad"]))
    suma = res["Aporte_intercepto"] + res[[f"Aporte_{n}" for n in cfg.NOMBRES_MODELOS]].sum(axis=1)
    paso = float(np.abs(logit_final - suma).max()) < 1e-6
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} intercepto + aportes de los 4 modelos = logit del riesgo")

    # 3) Campos opcionales vacios: no rompe y lo avisa
    vacia = entrada.head(5).copy()
    vacia[cfg.CLIMA_NUMERICO + ["Wind_Direction", "Weather_Condition"]] = np.nan
    r = predictor.predecir(vacia)
    paso = bool((r["Campos_estimados"] >= len(cfg.CLIMA_NUMERICO) + 2).all() and r["Probabilidad"].between(0, 1).all())
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} campos opcionales vacios se estiman y se cuentan (Campos_estimados)")

    # 3b) Tipo de via: desplegable, nombre de calle en texto libre y valores invalidos
    opciones = predictor.opciones_formulario().get("Tipo_via", [])
    paso = set(opciones) == set(TIPOS_VIA) | {cfg.VALOR_DESCONOCIDO}
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} opciones_formulario() ofrece Tipo_via: {', '.join(opciones)}")

    calles = {"Interestatal": "I-95 N", "Carretera_US": "US-1 S", "Ruta_estatal": "CA-99 S", "Autopista": "Pomona Fwy W",
              "Carretera": "Pacific Coast Hwy", "Local": "W Main St", "Otro": "Broadway"}
    base = entrada.head(len(calles)).copy()
    con_tipo = base.copy(); con_tipo["Tipo_via"] = list(calles)
    con_calle = base.drop(columns="Tipo_via").copy(); con_calle["Street"] = list(calles.values())
    dif = np.abs(predictor.predecir(con_tipo)["Probabilidad"].to_numpy()
                 - predictor.predecir(con_calle)["Probabilidad"].to_numpy()).max()
    sin_tipo = con_tipo.copy(); sin_tipo["Tipo_via"] = np.nan
    extra = (predictor.predecir(sin_tipo)["Campos_estimados"].to_numpy()
             - predictor.predecir(con_tipo)["Campos_estimados"].to_numpy())
    paso = bool(dif < 1e-12 and (extra == 1).all())
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} Tipo_via: enviar la calle (texto) da lo mismo que enviar su tipo "
          f"(dif. {dif:.1e}); si falta, cuenta como campo estimado")
    malo = base.head(2).copy(); malo["Tipo_via"] = "Marte"
    try:
        predictor.predecir(malo)
        paso = False
    except ValueError:
        paso = True
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} Tipo_via no valido -> ValueError claro")

    # 4) Datos invalidos: error claro
    for descripcion, modificar in [
        ("latitud vacia", lambda d: d.__setitem__("Start_Lat", np.nan)),
        ("humedad imposible (150%)", lambda d: d.__setitem__("Humidity(%)", 150.0)),
        ("falta la columna State", lambda d: d.drop(columns="State", inplace=True)),
    ]:
        prueba = entrada.head(2).copy()
        modificar(prueba)
        try:
            predictor.predecir(prueba)
            paso = False
        except ValueError:
            paso = True
        ok &= paso
        print(f"  {'OK   ' if paso else 'FALLA'} {descripcion} -> ValueError claro")

    # 4b) Umbral personalizado (control de umbral de la app): cambia la decision, no el riesgo
    tau_prueba = float(np.quantile(res["Probabilidad"], 0.9))
    r2 = predictor.predecir(entrada, umbral=tau_prueba)
    oficial = predictor.predecir(entrada, umbral=predictor.decision["umbral"])
    sobre_el_umbral = r2["Nivel"].isin(["Vigilancia (sobre el umbral)", "Critico"]).to_numpy()
    paso = bool(
        (r2["Es_critico"].to_numpy() == (res["Probabilidad"].to_numpy() >= tau_prueba)).all()
        and r2["Es_critico"].any() and not r2["Es_critico"].all()
        # el Random Forest suma sus arboles en paralelo: dos llamadas identicas difieren ~1e-16
        and float(np.abs(r2["Probabilidad"].to_numpy() - res["Probabilidad"].to_numpy()).max()) < 1e-12
        and (sobre_el_umbral == r2["Es_critico"].to_numpy()).all()
        and (oficial["Nivel"].to_numpy() == res["Nivel"].to_numpy()).all()
        and (oficial["Es_critico"].to_numpy() == res["Es_critico"].to_numpy()).all()
    )
    for invalido in (0, 1, 1.5, -0.2):
        try:
            predictor.predecir(entrada.head(2), umbral=invalido)
            paso = False
        except ValueError:
            pass
    ok &= paso
    print(f"  {'OK   ' if paso else 'FALLA'} umbral personalizado: cambia Es_critico y Nivel, no la probabilidad; "
          f"umbral fuera de (0, 1) -> ValueError")

    # 5) Explicacion coherente con la prediccion
    if predictor.estado_explicador is not None:
        try:
            expl, base = predictor.explicar(entrada.head(5))
            reconstruido = 1 / (1 + np.exp(-(base + expl.sum(axis=1))))
            paso = float(np.abs(reconstruido.to_numpy() - res["Probabilidad"].head(5).to_numpy()).max()) < 1e-3
            ok &= paso
            print(f"  {'OK   ' if paso else 'FALLA'} explicacion SHAP: base + aportes = riesgo "
                  f"(RF desglosado: {predictor.estado_explicador['incluir_rf']})")
        except ImportError:
            print("  AVISO: no se pudo probar la explicacion (falta instalar shap)")
    else:
        print("  AVISO: no hay explicador_estado.json (ejecute pipeline/04_explicabilidad_shap.py)")
    resultados["todo_ok"] = bool(ok)
    return ok, resultados, predictor


def generar_curva(esc, decision):
    """Curva de umbral (val_umbral y test) para el control de umbral de la app.

    Se calcula con los puntajes del entrenamiento (puntajes.npz), sin reentrenar, y se
    VALIDA: el punto del umbral oficial debe coincidir con la tabla del script 03, que
    calcula las mismas metricas con otro codigo. Devuelve (ok, resultados, curva).
    """
    resultados, ok = {}, True
    puntajes = np.load(cfg.DIR_MODELOS / esc / "puntajes.npz")
    tabla = pd.read_csv(cfg.DIR_MODELOS / esc / "tabla_comparativa.csv", encoding=cfg.ENCODING)
    # puntajes.npz guarda float32: se compara el umbral con esa misma precision
    umbral_f32 = float(np.float32(decision["umbral"]))
    partes, dif_global = [], 0.0
    for particion in ("val_umbral", "test"):
        y = puntajes[f"y__{particion}"]
        curva = curva_de_umbral(y, puntajes[f"{cfg.METODO_FINAL}__{particion}"], umbral_oficial=umbral_f32)
        curva.loc[curva["Es_umbral_oficial"], "Umbral"] = decision["umbral"]

        forma = (int(curva["Es_umbral_oficial"].sum()) == 1
                 and bool(curva["Alertas_%"].is_monotonic_increasing)
                 and bool(curva["Recall"].is_monotonic_increasing)
                 and bool(curva["Falsas_alarmas_x1000"].is_monotonic_increasing)
                 and bool(curva["Criticos_perdidos_x1000"].is_monotonic_decreasing))
        dif = float("inf")
        if forma:
            ref = tabla[(tabla["Metodo"] == cfg.METODO_FINAL) & (tabla["Particion"] == particion)]
            if len(ref) != 1:
                raise ValueError(f"tabla_comparativa.csv no tiene exactamente una fila de {cfg.METODO_FINAL} en {particion}")
            ref, punto = ref.iloc[0], curva[curva["Es_umbral_oficial"]].iloc[0]
            dif = max(abs(punto[c] - ref[c]) for c in ("Exactitud", "Baseline", "Recall", "Precision", "F1", "Balanced_acc"))
            dif = max(dif, abs(punto["Alertas_%"] - ref["Alertas_%"]) / 100)
        paso = forma and dif < TOLERANCIA_CURVA
        ok &= paso
        dif_global = max(dif_global, dif)
        print(f"  {'OK   ' if paso else 'FALLA'} curva en {particion}: {len(curva)} umbrales, ordenada y con un unico punto oficial; "
              f"ese punto = tabla del script 03 (diferencia maxima {dif:.1e})")
        curva.insert(0, "Particion", particion)
        partes.append(curva)

    curva = pd.concat(partes, ignore_index=True)
    # metricas a 6 decimales; el umbral NO se redondea (el oficial debe quedar exacto)
    curva = curva.round({c: 6 for c in curva.select_dtypes("float").columns if c != "Umbral"})
    ruta = cfg.DIR_MODELOS / esc / "curva_umbral.csv"
    curva.to_csv(ruta, index=False, encoding=cfg.ENCODING)
    resultados["curva_umbral_filas"] = int(len(curva))
    resultados["curva_umbral_diferencia_maxima_vs_script_03"] = float(dif_global)
    return ok, resultados, curva


def mostrar_muestra_curva(curva):
    """Imprime unas filas de la curva en test: el trade-off que movera el control de la app."""
    t = curva[curva["Particion"] == "test"].reset_index(drop=True)
    filas = {int((t["Alertas_%"] - f * 100).abs().idxmin()) for f in FRACCIONES_MUESTRA}
    filas.add(int(t.index[t["Es_umbral_oficial"]][0]))
    print(f"\n    Curva en TEST (baseline 'todo normal' = {t['Baseline'].iloc[0]:.4f}); cifras por cada 1,000 incidentes:")
    print(f"    {'Alertas%':>9}{'Umbral':>9}{'Exactitud':>11}{'Recall':>8}{'Precision':>11}"
          f"{'Detect.':>9}{'Falsas':>8}{'Perdid.':>9}{'Pts':>5}")
    for i in sorted(filas):
        r = t.iloc[i]
        marca = "  <- umbral oficial" if r["Es_umbral_oficial"] else ""
        print(f"    {r['Alertas_%']:>9.2f}{r['Umbral']:>9.4f}{r['Exactitud']:>11.4f}{r['Recall']:>8.3f}"
              f"{r['Precision']:>11.3f}{r['Criticos_detectados_x1000']:>9.1f}{r['Falsas_alarmas_x1000']:>8.1f}"
              f"{r['Criticos_perdidos_x1000']:>9.1f}{int(r['Puntos_solo_exactitud']):>5d}{marca}")


def main():
    parser = argparse.ArgumentParser(description="Exporta y verifica los artefactos para la app.")
    # 'informacion_completa' usa Distance(mi), que solo se conoce DESPUES del incidente: el
    # formulario no la tiene, asi que ese escenario es solo de referencia (pregunta 4) y no se exporta.
    parser.add_argument("--escenario", choices=["tiempo_real"], default="tiempo_real")
    parser.add_argument("--max-mb", type=float, default=50.0,
                        help="archivos mas pesados no se copian a models/ (van a un Release)")
    parser.add_argument("--n-verificacion", type=int, default=2000)
    args = parser.parse_args()
    esc = args.escenario

    titulo(f"EXPORTACION PARA LA APP - {esc}")
    paquete = cfg.DIR_SALIDA / "exportacion"
    destino_paquete = paquete / esc
    if destino_paquete.exists():
        shutil.rmtree(destino_paquete)
    destino_paquete.mkdir(parents=True)

    print("[1] Reuniendo los archivos")
    for origen, nombre, obligatorio in archivos_del_modelo(esc):
        if not origen.exists():
            if obligatorio:
                raise FileNotFoundError(f"Falta {origen}. Ejecute antes los scripts 02 y 03.")
            print(f"    (opcional, no existe) {nombre}")
            continue
        shutil.copy2(origen, destino_paquete / nombre)
    print(f"    paquete completo en {destino_paquete}")

    print("\n[2] Verificacion de extremo a extremo (formulario -> riesgo)")
    ok, verificacion, predictor = verificar(paquete, esc, args.n_verificacion)
    if not ok:
        print("\nLA VERIFICACION FALLO: no se publica nada en models/. Revise las lineas marcadas FALLA.")
        sys.exit(1)

    print("\n[3] Curva de umbral (control de umbral de la app)")
    ok_curva, res_curva, curva = generar_curva(esc, predictor.decision)
    verificacion.update(res_curva)
    if not ok_curva:
        print("\nLA CURVA NO COINCIDE CON LA TABLA DEL SCRIPT 03: no se publica nada en models/.")
        sys.exit(1)
    mostrar_muestra_curva(curva)

    print("\n[4] Publicando en models/ lo que cabe en GitHub")
    destino_models = cfg.DIR_MODELOS_APP / esc
    destino_reportes = cfg.DIR_MODELOS_APP / "reportes"
    destino_models.mkdir(parents=True, exist_ok=True)
    destino_reportes.mkdir(parents=True, exist_ok=True)
    archivos, pesados = [], []
    for ruta in sorted(destino_paquete.iterdir()):
        mb = ruta.stat().st_size / 1024 ** 2
        en_release = mb > args.max_mb
        if en_release:
            pesados.append((ruta.name, mb))
        else:
            shutil.copy2(ruta, destino_models / ruta.name)
        archivos.append({"archivo": ruta.name, "mb": round(mb, 2), "sha256": sha256(ruta),
                         "ubicacion": "release" if en_release else f"models/{esc}"})
        print(f"    {ruta.name:<34} {mb:8.2f} MB  -> {'RELEASE (muy pesado)' if en_release else 'models/' + esc}")
    copiados = 0
    for origen, nombre in reportes(esc):
        if origen.exists():
            shutil.copy2(origen, destino_reportes / nombre)
            copiados += 1
    print(f"    {copiados} tablas pequenas -> models/reportes/")

    decision = predictor.decision
    manifiesto = {
        "escenario": esc,
        "creado": datetime.now().isoformat(timespec="seconds"),
        "metodo_final": decision["metodo_final"],
        "estrategia_balanceo": decision.get("estrategia_balanceo"),
        "umbral": decision["umbral"],
        "franja_vigilancia": [decision["umbral_vigilancia_inferior"], decision["umbral_vigilancia_superior"]],
        "versiones": versiones(),
        "verificacion": verificacion,
        "archivos": archivos,
    }
    ruta_manifiesto = cfg.DIR_MODELOS_APP / f"manifest_{esc}.json"
    with open(ruta_manifiesto, "w", encoding=cfg.ENCODING) as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)
    shutil.copy2(ruta_manifiesto, destino_paquete / ruta_manifiesto.name)

    titulo("EXPORTACION COMPLETADA")
    print(f"Manifiesto: {ruta_manifiesto}")
    print("Versiones con las que se guardaron los modelos (la app debe usar las MISMAS):")
    for k, v in manifiesto["versiones"].items():
        print(f"    {k:<18} {v}")
    if pesados:
        print("\nARCHIVOS DEMASIADO PESADOS PARA models/ (publiquelos como Release de GitHub):")
        for nombre, mb in pesados:
            print(f"    {nombre}  ({mb:.1f} MB)  en {destino_paquete}")
        print("La app los descarga y los deja en models/<escenario>/ antes de cargar el predictor.")
    else:
        print("\nTodo cabe en models/: haga `git add models/` y suba el repositorio.")
    print("Congele las versiones:  pip freeze > requirements-lock.txt")


if __name__ == "__main__":
    main()
