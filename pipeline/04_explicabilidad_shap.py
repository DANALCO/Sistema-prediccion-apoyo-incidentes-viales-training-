"""
=============================================================================
SCRIPT 04  -  EXPLICABILIDAD SHAP DEL MODELO INTEGRADO  (orquestador)
=============================================================================
Entrada : outputs/modelos/<escenario>/   (lo genera el script 03)
          outputs/preparado/              (matrices del script 02)
Salida  : shap/tablas/importancia_global_<escenario>.csv
          shap/tablas/importancia_por_grupo_<escenario>.csv
          shap/figures/shap_global_<escenario>.png       importancia global (distribucion real)
          shap/figures/shap_criticos_<escenario>.png     que empuja hacia "critico"
          shap/figures/shap_grupos_<escenario>.png       clima / tiempo / infraestructura / ubicacion
          shap/figures/shap_ejemplo_{tp,fp,fn}_<escenario>.png   un incidente por tipo de resultado
          outputs/modelos/<escenario>/explicador_estado.json     para la aplicacion

La explicacion es la del MODELO INTEGRADO (stacking): suma ponderada de las
explicaciones de los 4 modelos (ver accidentes/explain.py). En cada corrida se
verifica que  base + suma de aportes = logit del integrado.

Responde las preguntas de investigacion 1 a 3 (clima, patrones temporales,
infraestructura) con numeros: el aporte de cada grupo tematico.

IMPORTANTE AL LEER LOS RESULTADOS: SHAP muestra que usa el MODELO, no que causa
un accidente critico. Un aporte pequeno de una variable es una respuesta
valida ("asociacion debil"), no un error.

Uso (desde la raiz del repositorio):
    python pipeline/04_explicabilidad_shap.py
    python pipeline/04_explicabilidad_shap.py --sin-rf-shap     # mas rapido
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import joblib
import numpy as np
import pandas as pd

from accidentes import config as cfg
from accidentes.explain import (ExplicadorStacking, grafico_barras, grafico_contribuciones,
                                tabla_importancia, tabla_por_grupo)
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

TOLERANCIA_CONTROL = 1e-3


def cargar_artefactos(escenario):
    directorio = cfg.DIR_MODELOS / escenario
    if not (directorio / "decision.json").exists():
        raise FileNotFoundError(f"No hay modelos en {directorio}. Ejecute antes pipeline/03_entrenar_modelos.py")
    modelos = {n: joblib.load(directorio / f"modelo_{n}.joblib") for n in cfg.NOMBRES_MODELOS}
    apilador = joblib.load(directorio / "apilador.joblib")
    with open(directorio / "decision.json", encoding=cfg.ENCODING) as f:
        decision = json.load(f)
    nombres = (cfg.DIR_PREPARADO / f"features_{escenario}.txt").read_text(encoding=cfg.ENCODING).split("\n")
    return directorio, modelos, apilador, decision, nombres


def calcular_por_lotes(explicador, X, lote=500):
    partes = []
    for i in range(0, len(X), lote):
        t0 = time.time()
        partes.append(explicador.contribuciones(X[i:i + lote]))
        print(f"    {min(i + lote, len(X)):>6,} / {len(X):,} incidentes  ({time.time() - t0:.0f} s el ultimo lote)")
    return {
        "phi": np.vstack([p["phi"] for p in partes]),
        "extra": np.concatenate([p["extra"] for p in partes]) if partes[0]["extra"] is not None else None,
        "base": partes[0]["base"],
        "logit": np.concatenate([p["logit"] for p in partes]),
        "logit_real": np.concatenate([p["logit_real"] for p in partes]),
    }


def elegir_casos(score, y, umbral, inferior):
    """Un incidente 'tipico' (cercano a la mediana) de cada tipo de resultado."""
    def mediano(mascara):
        idx = np.flatnonzero(mascara)
        if len(idx) == 0:
            return None
        return int(idx[np.argmin(np.abs(score[idx] - np.median(score[idx])))])
    casos = {
        "tp": mediano((y == 1) & (score >= umbral)),
        "fp": mediano((y == 0) & (score >= umbral)),
        "fn": mediano((y == 1) & (score < inferior)),
    }
    return {k: v for k, v in casos.items() if v is not None}


def etiquetas_de_incidente(fila, variables):
    """'State' -> 'State = VA' usando los valores originales del incidente."""
    etiquetas = {}
    for v in variables:
        if v not in fila.index:
            continue
        valor = fila[v]
        if pd.isna(valor):
            texto = "sin dato"
        elif isinstance(valor, (bool, np.bool_)):
            texto = "Si" if valor else "No"
        elif isinstance(valor, (float, np.floating)):
            texto = f"{valor:.1f}"
        else:
            texto = str(valor)
        etiquetas[v] = f"{v} = {texto}"
    return etiquetas


def main():
    parser = argparse.ArgumentParser(description="SHAP del modelo integrado (stacking).")
    parser.add_argument("--escenario", choices=list(cfg.ESCENARIOS), default="tiempo_real")
    parser.add_argument("--sin-rf-shap", action="store_true",
                        help="no desglosa Random Forest (su aporte sale como una sola barra)")
    parser.add_argument("--n-real", type=int, default=cfg.SHAP_N_REAL)
    parser.add_argument("--n-criticos", type=int, default=cfg.SHAP_N_CRITICOS)
    args = parser.parse_args()
    esc = args.escenario
    rng = np.random.default_rng(cfg.SEMILLA)

    titulo(f"SHAP DEL MODELO INTEGRADO - {esc}")
    directorio, modelos, apilador, decision, nombres = cargar_artefactos(esc)
    print(f"Estrategia de balanceo del modelo: {decision.get('estrategia_balanceo', '?')}")
    print("Pesos del stacking (escala logit):",
          {k: round(v, 3) for k, v in apilador.coeficientes.items()})

    z = np.load(cfg.DIR_PREPARADO / f"datos_{esc}.npz")
    X_test, y_test, pos_test = z["X_test"], z["y_test"], z["pos_test"]
    fondo = z["X_train_real"] if "X_train_real" in z.files else z["X_val_ajuste"]

    # --- Muestra: parte al azar (distribucion real) y parte solo de criticos ---
    n_real = min(args.n_real, len(y_test))
    idx_real = rng.choice(len(y_test), n_real, replace=False)
    pos_criticos = np.flatnonzero(y_test == 1)
    idx_crit = rng.choice(pos_criticos, min(args.n_criticos, len(pos_criticos)), replace=False)
    idx = np.concatenate([idx_real, idx_crit])
    es_real = np.arange(len(idx)) < n_real
    es_critico = ~es_real
    print(f"\nMuestra de test: {n_real:,} al azar (real) + {len(idx_crit):,} criticos")

    # --- Explicador ---
    titulo("CALCULO DE LOS APORTES")
    explicador = ExplicadorStacking(modelos, apilador, nombres, cfg.CATEGORICAS,
                                    incluir_rf=not args.sin_rf_shap)
    paso = max(1, len(fondo) // 20000)
    explicador.fit(fondo[::paso][:20000], medias=fondo[::5].mean(axis=0, dtype="float64"))
    if args.sin_rf_shap:
        print("Random Forest NO se desglosa: su aporte se muestra como una sola barra.")
    resultado = calcular_por_lotes(explicador, X_test[idx])

    error = np.abs(resultado["logit"] - resultado["logit_real"])
    estado = "OK" if error.max() < TOLERANCIA_CONTROL else "AVISO: revisar"
    print(f"\nCONTROL DE ADITIVIDAD  (base + suma de aportes  vs  logit del integrado)")
    print(f"  error maximo {error.max():.2e} | error medio {error.mean():.2e} | {estado}")
    if error.max() >= TOLERANCIA_CONTROL:
        print("  El desglose no suma exactamente. Pruebe --sin-rf-shap para ver si el desvio viene de Random Forest.")

    # --- Tablas ---
    por_var = explicador.por_variable(resultado)
    grupos = cfg.GRUPOS_TEMATICOS
    imp = tabla_importancia(por_var, es_real, es_critico, grupos)
    imp_grupo = tabla_por_grupo(por_var, es_real, es_critico, grupos)

    cfg.DIR_SHAP_TABLAS.mkdir(parents=True, exist_ok=True)
    cfg.DIR_SHAP_FIGURAS.mkdir(parents=True, exist_ok=True)
    imp.to_csv(cfg.DIR_SHAP_TABLAS / f"importancia_global_{esc}.csv", index=False, encoding=cfg.ENCODING)
    imp_grupo.to_csv(cfg.DIR_SHAP_TABLAS / f"importancia_por_grupo_{esc}.csv", index=False, encoding=cfg.ENCODING)

    titulo("IMPORTANCIA POR VARIABLE (aporte absoluto medio al logit del riesgo)")
    print("'real' = incidentes al azar; 'criticos' = solo los que fueron criticos; el signo")
    print("de 'Aporte_medio_criticos' dice si la variable, en promedio, SUBE (+) o BAJA (-) el riesgo.\n")
    print(imp.head(15).round(4).to_string(index=False))

    titulo("IMPORTANCIA POR GRUPO TEMATICO (preguntas de investigacion 1 a 3)")
    print(imp_grupo.round(4).to_string(index=False))
    print("\nUn aporte pequeno es una respuesta valida: indica asociacion debil en este modelo.")

    # --- Figuras globales ---
    top = imp.head(15).set_index("Variable")
    grafico_barras(top["Aporte_abs_medio_real"], f"Importancia global (modelo integrado) - {esc}",
                   cfg.DIR_SHAP_FIGURAS / f"shap_global_{esc}.png", "aporte absoluto medio (log-odds)")
    top_c = imp.sort_values("Aporte_abs_medio_criticos", ascending=False).head(15).set_index("Variable")
    grafico_barras(top_c["Aporte_abs_medio_criticos"], f"Variables que mas pesan en los incidentes criticos - {esc}",
                   cfg.DIR_SHAP_FIGURAS / f"shap_criticos_{esc}.png", "aporte absoluto medio (log-odds)")
    grafico_barras(imp_grupo.set_index("Grupo")["Aporte_abs_medio_real"],
                   f"Aporte neto por grupo tematico - {esc}",
                   cfg.DIR_SHAP_FIGURAS / f"shap_grupos_{esc}.png", "aporte neto absoluto medio (log-odds)")

    # --- Un incidente por tipo de resultado (con valores originales) ---
    titulo("EJEMPLOS DE INCIDENTES")
    puntajes = np.load(directorio / "puntajes.npz")
    score_test = puntajes["Stacking__test"]
    casos = elegir_casos(score_test, y_test, decision["umbral"], decision["umbral_vigilancia_inferior"])
    tabla_limpia = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))
    nombres_casos = {"tp": "Critico detectado (acierto)", "fp": "Falsa alarma",
                     "fn": "Critico que el modelo NO detecto"}
    if casos:
        idx_casos = [casos[k] for k in casos]
        res_c = explicador.contribuciones(X_test[idx_casos])
        pv_c = explicador.por_variable(res_c)
        for i, tipo in enumerate(casos):
            fila = tabla_limpia.iloc[int(pos_test[casos[tipo]])]
            etiquetas = etiquetas_de_incidente(fila, pv_c.columns)
            prob_shap = 1 / (1 + np.exp(-res_c["logit"][i]))
            print(f"  {nombres_casos[tipo]:<34} puntaje del modelo {score_test[casos[tipo]]:.4f} | "
                  f"reconstruido con SHAP {prob_shap:.4f}")
            grafico_contribuciones(pv_c.iloc[i], res_c["base"], nombres_casos[tipo],
                                   cfg.DIR_SHAP_FIGURAS / f"shap_ejemplo_{tipo}_{esc}.png",
                                   etiquetas=etiquetas)
    else:
        print("  (no hubo casos de los tres tipos para ilustrar)")

    # --- Estado del explicador para la app ---
    with open(directorio / "explicador_estado.json", "w", encoding=cfg.ENCODING) as f:
        json.dump(explicador.estado(), f)

    print("\n" + "=" * 70)
    print("EXPLICABILIDAD COMPLETADA")
    print(f"Tablas   -> {cfg.DIR_SHAP_TABLAS}")
    print(f"Graficos -> {cfg.DIR_SHAP_FIGURAS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
