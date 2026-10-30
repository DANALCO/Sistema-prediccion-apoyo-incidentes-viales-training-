"""
=============================================================================
AJUSTE DE HIPERPARAMETROS  -  XGBoost y LightGBM (sin tocar el test)
=============================================================================
Pregunta que responde: regularizando XGBoost y LightGBM (que hoy memorizan el
entrenamiento: PR-AUC 0.231 en train vs 0.139 en test), el modelo DETECTA mejor?

Protocolo (la regla de oro del proyecto):
    train        entrena cada candidato
    val_ajuste   elige: cuantos arboles (parada temprana) y el mejor candidato
    val_umbral   CONFIRMA: el candidato elegido debe ganar tambien aqui
    test         NO SE LEE. Este script ni siquiera carga esa particion.

Se compara cada candidato con la configuracion ACTUAL de config.py (entrenada
exactamente como lo hace el script 03) y se mide la diferencia de PR-AUC con un
intervalo bootstrap, para saber si la mejora es real o ruido.

Entrada : outputs/preparado/datos_<escenario>.npz   (script 02)
Salida  : outputs/modelos/<escenario>/ajuste_hiperparametros.csv   (todos los intentos)
          outputs/modelos/<escenario>/ajuste/                      (puntajes, para reanudar)
NO modifica config.py: al final imprime las lineas para pegar SI el resultado lo justifica.

REANUDABLE: si lo interrumpe (Ctrl+C) o se corta, vuelva a ejecutar el mismo
comando y continua donde quedo. Ctrl+C tambien muestra el resumen de lo hecho.

Uso (desde la raiz del repositorio):
    python pipeline/ajustar_hiperparametros.py                      # 12 candidatos por modelo
    python pipeline/ajustar_hiperparametros.py --fraccion-train 0.3 --n-config 6   # prueba rapida
    python pipeline/ajustar_hiperparametros.py --n-config 20        # amplia la busqueda (conserva lo hecho)
    python pipeline/ajustar_hiperparametros.py --reiniciar          # empieza de cero
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes import tuning
from accidentes.training import datos_de_entrenamiento
from accidentes.utils import titulo

TAMANO_MUESTRA_BRECHA = 300_000     # filas de train para medir el PR-AUC de entrenamiento


def submuestrear(X, y, fraccion, semilla):
    if fraccion >= 1:
        return X, y
    rng = np.random.default_rng(semilla)
    idx = np.sort(rng.choice(len(y), int(len(y) * fraccion), replace=False))
    return X[idx], y[idx]


def formatear_parametros(d):
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


def mostrar_resumen(df, modelos, y_umbral, directorio_puntajes, n_boot, alertas):
    titulo("RESUMEN DEL AJUSTE")
    print(f"Elige: PR-AUC en val_ajuste. Confirma: PR-AUC en val_umbral (no se uso para elegir).")
    print(f"Recall/Precision se miden marcando el {alertas:.0%} de incidentes con mayor riesgo, en val_umbral.")
    print("Diferencias de PR-AUC menores a ~0.005 suelen ser ruido: mire el intervalo.\n")
    recomendaciones = {}
    for nombre in modelos:
        t = df[df["Modelo"] == nombre]
        base = t[t["Config"] == "actual"]
        cand = t[t["Config"] != "actual"].sort_values("PR_AUC_val_ajuste", ascending=False)
        if base.empty or cand.empty:
            continue
        base, mejor = base.iloc[0], cand.iloc[0]
        print(f"--- {nombre}: {len(cand)} candidatos probados ---")
        columnas = ["Config", "Arboles", "PR_AUC_val_ajuste", "PR_AUC_val_umbral", "Recall_en_alertas",
                    "Precision_en_alertas", "Brecha_train_vs_ajuste", "Segundos"]
        vista = pd.concat([base.to_frame().T, cand.head(5)])[columnas]
        vista = vista.rename(columns={"PR_AUC_val_ajuste": "PR_ajuste", "PR_AUC_val_umbral": "PR_umbral",
                                      "Recall_en_alertas": "Recall", "Precision_en_alertas": "Precision",
                                      "Brecha_train_vs_ajuste": "Brecha_train"})
        for c in ("PR_ajuste", "PR_umbral", "Recall", "Precision", "Brecha_train"):
            vista[c] = vista[c].astype(float).round(4)
        vista["Segundos"] = vista["Segundos"].astype(float).round(0)
        print(vista.to_string(index=False))
        if bool(mejor["Tope"]):
            print("  AVISO: el mejor candidato llego al tope de arboles; podria mejorar con mas arboles.")

        p_mejor = np.load(directorio_puntajes / f"{nombre}__{mejor['Config']}.npy")
        p_base = np.load(directorio_puntajes / f"{nombre}__actual.npy")
        delta, lo, hi = tuning.diferencia_pr_auc_bootstrap(y_umbral, p_mejor, p_base, n_boot, cfg.SEMILLA)
        dr = mejor["Recall_en_alertas"] - base["Recall_en_alertas"]
        print(f"\n  Mejor candidato: '{mejor['Config']}' ({int(mejor['Arboles'])} arboles)")
        print(f"  PR-AUC en val_umbral, candidato - actual: {delta:+.4f}  (IC 95%: {lo:+.4f} a {hi:+.4f})")
        print(f"  Recall al marcar {alertas:.0%}: {dr:+.3f} | brecha train-vs-validacion: "
              f"{mejor['Brecha_train_vs_ajuste']:.3f} (antes {base['Brecha_train_vs_ajuste']:.3f})")
        print(f"  => {tuning.veredicto(delta, lo, hi)}\n")
        recomendaciones[nombre] = (mejor, delta, lo)

    mejoran = {n: r for n, r in recomendaciones.items() if r[2] > 0 and r[1] >= tuning.MEJORA_MINIMA}
    if mejoran:
        titulo("PARA config.py  (pegue SOLO las lineas de los modelos con MEJORA confirmada y relevante)")
        for nombre, (mejor, _, _) in mejoran.items():
            params = json.loads(mejor["Parametros"])
            fusion = {**cfg.HIPERPARAMETROS[nombre], **params, "n_estimators": int(mejor["Arboles"])}
            print(f'    "{nombre}": dict({formatear_parametros(fusion)}),')
        print("\nDespues: python pipeline/03_entrenar_modelos.py -> 04 -> 06. La decision de adoptar se tomo")
        print("con las validaciones; el test se mira recien con el modelo final, no para elegir entre versiones.")
    else:
        print("Ningun modelo mostro una mejora confirmada y relevante: no hay motivo para cambiar config.py.")
        print("(Es un resultado valido: significa que el limite lo pone la senal de los datos, no el sobreajuste.)")


def main():
    parser = argparse.ArgumentParser(description="Ajusta XGBoost y LightGBM sin usar el test.")
    parser.add_argument("--escenario", choices=list(cfg.ESCENARIOS), default="tiempo_real")
    parser.add_argument("--modelos", nargs="+", choices=list(tuning.MODELOS_AJUSTABLES),
                        default=list(tuning.MODELOS_AJUSTABLES))
    parser.add_argument("--estrategia", choices=cfg.ESTRATEGIAS_BALANCEO, default=cfg.ESTRATEGIA_FINAL)
    parser.add_argument("--n-config", type=int, default=12, help="candidatos por modelo (el 1o es el conservador)")
    parser.add_argument("--fraccion-train", type=float, default=1.0,
                        help="usar solo esta fraccion del train (pruebas rapidas; 1.0 = todo)")
    parser.add_argument("--alertas", type=float, default=0.05,
                        help="fraccion de incidentes marcados para medir recall/precision")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--reiniciar", action="store_true", help="borra los resultados previos")
    args = parser.parse_args()
    if not 0 < args.fraccion_train <= 1:
        parser.error("--fraccion-train debe estar en (0, 1]")

    esc = args.escenario
    directorio = cfg.DIR_MODELOS / esc
    directorio_puntajes = directorio / "ajuste"
    ruta_csv = directorio / "ajuste_hiperparametros.csv"
    directorio_puntajes.mkdir(parents=True, exist_ok=True)

    titulo(f"AJUSTE DE HIPERPARAMETROS - {esc}")
    print(f"Modelos: {', '.join(args.modelos)} | candidatos por modelo: {args.n_config} | "
          f"estrategia: {args.estrategia} | fraccion de train: {args.fraccion_train:.0%}")
    print("Particiones usadas: train, val_ajuste, val_umbral.  Test: NO SE LEE.\n")

    ajustes = {"Estrategia": args.estrategia, "Fraccion_train": args.fraccion_train,
               "Semilla": cfg.SEMILLA, "Alertas_frac": args.alertas}
    if args.reiniciar:
        ruta_csv.unlink(missing_ok=True)
        for f in directorio_puntajes.glob("*.npy"):
            f.unlink()
    filas = []
    if ruta_csv.exists():
        previo = pd.read_csv(ruta_csv, encoding=cfg.ENCODING)
        distintos = [k for k, v in ajustes.items() if k in previo and not (previo[k] == v).all()]
        if distintos:
            sys.exit(f"Los resultados previos se hicieron con otros ajustes ({', '.join(distintos)}). "
                     f"Use --reiniciar para empezar de cero o repita los mismos argumentos.")
        filas = previo.to_dict("records")
        print(f"Se reanuda: {len(filas)} resultados previos en {ruta_csv.name}\n")

    print("[1] Carga de matrices (train + 2 validaciones)")
    datos = tuning.cargar_sin_test(esc, args.estrategia)
    clave = "train_bal" if args.estrategia == "SMOTENC" else "train_real"
    X_tr, y_tr = submuestrear(*datos[clave], args.fraccion_train, cfg.SEMILLA)
    X_ent, y_ent, peso = datos_de_entrenamiento({clave: (X_tr, y_tr)}, args.estrategia)
    particiones = {p: datos[p] for p in ("val_ajuste", "val_umbral")}
    rng = np.random.default_rng(cfg.SEMILLA)
    idx = np.sort(rng.choice(len(y_ent), min(TAMANO_MUESTRA_BRECHA, len(y_ent)), replace=False))
    X_brecha, y_brecha = X_ent[idx], y_ent[idx]
    detalle = f" | peso de los criticos x{peso:.1f}" if peso else ""
    print(f"    train {len(y_ent):,} filas ({y_ent.mean() * 100:.2f}% criticos){detalle}")
    for p, (X, y) in particiones.items():
        print(f"    {p:<11} {len(y):,} filas ({y.mean() * 100:.2f}% criticos)")
    y_umbral = particiones["val_umbral"][1]

    hechos = {(f["Modelo"], f["Config"]) for f in filas
              if (directorio_puntajes / f"{f['Modelo']}__{f['Config']}.npy").exists()}
    filas = [f for f in filas if (f["Modelo"], f["Config"]) in hechos]

    def registrar(nombre, config_id, params, modelo, arboles, tope, segundos):
        fila, puntajes_umbral = tuning.medir(modelo, particiones, X_brecha, y_brecha, args.alertas)
        np.save(directorio_puntajes / f"{nombre}__{config_id}.npy", puntajes_umbral)
        fila = {"Modelo": nombre, "Config": config_id, "Arboles": arboles, "Tope": bool(tope),
                "Segundos": round(segundos, 1), **fila, **ajustes, "Parametros": json.dumps(params)}
        filas[:] = [f for f in filas if (f["Modelo"], f["Config"]) != (nombre, config_id)] + [fila]
        pd.DataFrame(filas).to_csv(ruta_csv, index=False, encoding=cfg.ENCODING)
        return fila

    interrumpido = False
    try:
        for nombre in args.modelos:
            titulo(f"{nombre}")
            if (nombre, "actual") not in hechos:
                print(f"  actual (config.py: {cfg.HIPERPARAMETROS[nombre]['n_estimators']} arboles) ...", end=" ", flush=True)
                modelo, arboles, tope, seg = tuning.entrenar_actual(nombre, X_ent, y_ent, peso, cfg.SEMILLA)
                f = registrar(nombre, "actual", {}, modelo, arboles, tope, seg)
                print(f"PR-AUC ajuste {f['PR_AUC_val_ajuste']:.4f} | umbral {f['PR_AUC_val_umbral']:.4f} | "
                      f"train {f['PR_AUC_train']:.4f} | {seg:.0f} s")
                del modelo
            else:
                print("  actual: ya calculado")

            candidatos = tuning.configuraciones(nombre, args.n_config, cfg.SEMILLA)
            tiempos = []
            for i, (config_id, params) in enumerate(candidatos, start=1):
                if (nombre, config_id) in hechos:
                    continue
                resumen = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  [{i}/{len(candidatos)}] {config_id}: {resumen}\n      ...", end=" ", flush=True)
                modelo, arboles, tope, seg = tuning.entrenar_con_parada(
                    nombre, params, X_ent, y_ent, *particiones["val_ajuste"], peso, cfg.SEMILLA)
                f = registrar(nombre, config_id, params, modelo, arboles, tope, seg)
                tiempos.append(seg)
                faltan = sum(1 for c, _ in candidatos if (nombre, c) not in hechos) - len(tiempos)
                eta = f" | quedan ~{faltan * np.mean(tiempos) / 60:.0f} min" if faltan > 0 else ""
                print(f"{arboles} arboles{' (TOPE)' if tope else ''} | PR-AUC ajuste {f['PR_AUC_val_ajuste']:.4f} | "
                      f"umbral {f['PR_AUC_val_umbral']:.4f} | {seg:.0f} s{eta}")
                del modelo
    except KeyboardInterrupt:
        interrumpido = True
        print("\n\nInterrumpido. Se muestra el resumen de lo hecho; vuelva a ejecutar para continuar.")

    df = pd.DataFrame(filas)
    mostrar_resumen(df, args.modelos, y_umbral, directorio_puntajes, args.n_bootstrap, args.alertas)
    print(f"\nTodos los intentos: {ruta_csv}")
    if interrumpido:
        sys.exit(130)


if __name__ == "__main__":
    main()
