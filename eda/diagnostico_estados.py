"""
=============================================================================
DIAGNOSTICO  -  por que casi no hay alertas en CA, TX o FL, y se puede corregir?
=============================================================================
Observacion (05_analisis_errores, test): VA, NC y GA reciben ~22% de alertas; CA, el estado
con mas incidentes, recibe 0.002% (recall 0.001). Dos explicaciones posibles:

  (A) Las tasas de criticos difieren mucho en MAGNITUD entre estados (0.4% a 9.7%): con un umbral
      global, marcar los puntajes mas altos es lo que maximiza los criticos detectados, y un
      estado de tasa baja casi nunca llega al umbral aunque el modelo ordene bien sus incidentes.
      No hay nada que corregir; cubrir CA es una decision de politica que cuesta recall global.
  (B) El modelo esta DESCALIBRADO por estado (por ejemplo, la tasa de un estado cambio despues
      del entrenamiento) y una correccion por estado mejoraria el ranking.

Que hace (solo train y las 2 validaciones; el test NO se usa):
  [1] Estabilidad en el tiempo de la tasa de criticos de cada estado (train, val_ajuste, val_umbral).
  [2] Calibracion y poder de discriminacion DENTRO de cada estado (val_umbral; val_ajuste al lado).
  [3] Con el MISMO numero de alertas que el umbral oficial, compara en val_umbral:
        - Global (actual)           marcar los puntajes mas altos
        - Recalibrado por estado    ajuste de logit por estado estimado en val_ajuste (con
                                    encogimiento hacia 0); es la unica opcion que podria MEJORAR el ranking
        - Relativo al estado        puntaje / tasa historica del estado (train)
        - Cobertura pareja          el mismo % de alertas en cada estado (politica, no mejora)
  [4] Veredicto con intervalo bootstrap y el mismo criterio de relevancia del ajuste (MEJORA_MINIMA).

Protocolo: la correccion por estado se ESTIMA en val_ajuste y se CONFIRMA en val_umbral. La tabla limpia
se recorta a train + validaciones al leerla; las claves de test del .npz no se tocan.
No modifica el pipeline: si ayuda, se incorpora despues, a proposito.

Entrada : outputs/preparado/datos_<esc>.npz y outputs/modelos/<esc>/ (scripts 02 y 03), outputs/limpio/ (01)
Salida  : eda/figures/estados_calibracion.csv y estados_asignacion.csv
Uso (desde la raiz del repositorio):   python eda/diagnostico_estados.py
=============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from accidentes import config as cfg
from accidentes import tuning
from accidentes.ensemble import logit
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

MIN_AUC = 30          # minimo de criticos y de normales en un estado para calcular su AUC


# ----------------------------------------------------------------------------
def cargar(escenario):
    """Lee SOLO train y las dos validaciones (matrices no: solo etiquetas, posiciones y puntajes)."""
    ruta = cfg.DIR_PREPARADO / f"datos_{escenario}.npz"
    with np.load(ruta) as z:
        if "y_train_real" not in z.files:
            sys.exit(f"{ruta.name} no trae y_train_real (revise GUARDAR_TRAIN_REAL en config.py)")
        y_train = z["y_train_real"]
        pos = {"val_ajuste": z["pos_val_ajuste"], "val_umbral": z["pos_val_umbral"]}
        y = {"val_ajuste": z["y_val_ajuste"], "val_umbral": z["y_val_umbral"]}
    n_train = len(y_train)
    if not (np.array_equal(pos["val_ajuste"], np.arange(n_train, n_train + len(pos["val_ajuste"])))
            and np.array_equal(pos["val_umbral"],
                               np.arange(pos["val_ajuste"][-1] + 1, pos["val_ajuste"][-1] + 1 + len(pos["val_umbral"])))):
        sys.exit("Las particiones no son contiguas en la tabla ordenada por fecha; revise 02.")
    n_filas = int(pos["val_umbral"][-1]) + 1

    directorio = cfg.DIR_MODELOS / escenario
    with np.load(directorio / "puntajes.npz") as z:
        puntajes = {p: z[f"{cfg.METODO_FINAL}__{p}"].astype("float64") for p in ("val_ajuste", "val_umbral")}
        for p in puntajes:
            if not np.array_equal(z[f"y__{p}"], y[p]):
                sys.exit(f"Las etiquetas de puntajes.npz y de datos_{escenario}.npz no coinciden en {p}: "
                         "ejecute 02 y 03 de nuevo.")
    with open(directorio / "decision.json", encoding=cfg.ENCODING) as f:
        umbral = float(json.load(f)["umbral"])
    return y_train, pos, y, n_train, n_filas, puntajes, umbral


def leer_estados(n_filas, y_train, pos, y):
    """Estado de cada fila de train + validaciones, alineado con las etiquetas (si no, se detiene)."""
    base = str(cfg.DIR_LIMPIO / "us_accidents_limpio")
    columnas = ["State", cfg.COL_OBJETIVO]
    tabla = (pd.read_parquet(base + ".parquet", columns=columnas) if os.path.exists(base + ".parquet")
             else leer_tabla(base)[columnas])
    tabla = tabla.iloc[:n_filas]                       # el resto (test) se descarta aqui y no se usa
    estado = tabla["State"].astype(object).fillna(cfg.VALOR_DESCONOCIDO).to_numpy()
    critico = tabla[cfg.COL_OBJETIVO].to_numpy()
    del tabla
    n_train = len(y_train)
    coincide = (np.array_equal(critico[:n_train], y_train)
                and np.array_equal(critico[pos["val_ajuste"]], y["val_ajuste"])
                and np.array_equal(critico[pos["val_umbral"]], y["val_umbral"]))
    if not coincide:
        sys.exit("Las etiquetas de la tabla limpia NO coinciden con las del script 02: ejecute 01 y 02 de nuevo.")
    return {"train": estado[:n_train], "val_ajuste": estado[pos["val_ajuste"]], "val_umbral": estado[pos["val_umbral"]]}


# ----------------------------------------------------------------------------
def recall_por_estado(estado, y, alerta):
    """Serie: recall dentro de cada estado para una mascara de alertas."""
    d = pd.DataFrame({"e": estado, "y": y, "a": alerta & (y == 1)})
    g = d[d["y"] == 1].groupby("e")["a"].agg(["sum", "size"])
    return (g["sum"] / g["size"]).rename("recall")


def auc_dentro(y, s):
    if y.sum() < MIN_AUC or (len(y) - y.sum()) < MIN_AUC:
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def tabla_calibracion(estado, y, s, umbral, estado_aj, y_aj, s_aj):
    """Una fila por estado en val_umbral (O/E = criticos observados / esperados por el modelo)."""
    alerta = s >= umbral
    filas = []
    for e in sorted(set(estado)):
        m = estado == e
        ma = estado_aj == e
        roc, ap = auc_dentro(y[m], s[m])
        esperados_aj = s_aj[ma].sum()
        filas.append({
            "Estado": e, "Incidentes": int(m.sum()), "Criticos": int(y[m].sum()),
            "Tasa_critico_%": y[m].mean() * 100, "Prob_media_%": s[m].mean() * 100,
            "OE_umbral": y[m].sum() / s[m].sum() if s[m].sum() > 0 else np.nan,
            "OE_ajuste": y_aj[ma].sum() / esperados_aj if esperados_aj > 0 else np.nan,
            "ROC_AUC_dentro": roc, "PR_AUC_dentro": ap,
            "Alertas_%": alerta[m].mean() * 100,
            "Recall": (alerta[m] & (y[m] == 1)).sum() / y[m].sum() if y[m].sum() else np.nan,
        })
    return pd.DataFrame(filas).sort_values("Incidentes", ascending=False).reset_index(drop=True)


def ajuste_por_estado(estado_aj, y_aj, s_aj, prior):
    """log de la razon (criticos observados + prior) / (esperados + prior), por estado, en val_ajuste.
    El prior son 'criticos virtuales' que encogen hacia 0 (sin correccion) los estados con pocos datos."""
    d = pd.DataFrame({"e": estado_aj, "o": y_aj, "p": s_aj}).groupby("e").agg(o=("o", "sum"), p=("p", "sum"))
    return np.log((d["o"] + prior) / (d["p"] + prior))


def tasa_historica(estado_tr, y_tr, m=200):
    """Tasa de criticos por estado en train, encogida hacia la tasa global (m incidentes virtuales)."""
    base = y_tr.mean()
    d = pd.DataFrame({"e": estado_tr, "y": y_tr}).groupby("e")["y"].agg(["sum", "size"])
    return (d["sum"] + m * base) / (d["size"] + m)


def marcar_top(score, k):
    idx = np.argpartition(-score, k - 1)[:k]
    mascara = np.zeros(len(score), dtype=bool)
    mascara[idx] = True
    return mascara


def marcar_top_por_estado(score, estado, fraccion):
    mascara = np.zeros(len(score), dtype=bool)
    for e in np.unique(estado):
        idx = np.flatnonzero(estado == e)
        k = int(round(len(idx) * fraccion))
        if k >= 1:
            mascara[idx[np.argpartition(-score[idx], k - 1)[:k]]] = True
    return mascara


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diagnostico: comportamiento del modelo por estado")
    parser.add_argument("--escenario", choices=["tiempo_real"], default="tiempo_real")
    parser.add_argument("--prior", type=float, default=50.0,
                        help="criticos virtuales del encogimiento de la correccion por estado")
    parser.add_argument("--min-criticos", type=int, default=50,
                        help="minimo de criticos en val_umbral para contar un estado en los resumenes")
    parser.add_argument("--min-incidentes", type=int, default=2000,
                        help="minimo de incidentes (en train y en val_umbral) para comparar la estabilidad de la tasa")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()

    titulo("DIAGNOSTICO: COMPORTAMIENTO DEL MODELO POR ESTADO")
    print(f"Modelo: {cfg.METODO_FINAL} | particiones usadas: train, val_ajuste, val_umbral.  Test: NO SE USA.\n")
    y_train, pos, y, n_train, n_filas, s, umbral = cargar(args.escenario)
    estado = leer_estados(n_filas, y_train, pos, y)
    y_aj, y_um, s_aj, s_um = y["val_ajuste"], y["val_umbral"], s["val_ajuste"], s["val_umbral"]
    presupuesto = float((s_um >= umbral).mean())
    print(f"Umbral oficial {umbral:.5f} -> {presupuesto * 100:.2f}% de alertas en val_umbral "
          f"(el mismo volumen se usa en todas las estrategias del paso [3]).")

    # ------------------------------------------------------------------
    titulo("[1] ESTABILIDAD DE LA TASA DE CRITICOS POR ESTADO")
    partes = []
    for p, yy in (("train", y_train), ("val_ajuste", y_aj), ("val_umbral", y_um)):
        g = pd.Series(yy).groupby(estado[p]).agg(["size", "mean"])
        g.columns = [f"n_{p}", f"tasa_{p}"]
        partes.append(g)
    tasas = pd.concat(partes, axis=1).sort_values("n_val_umbral", ascending=False)
    vista = tasas.head(15).copy()
    for p in ("train", "val_ajuste", "val_umbral"):
        vista[f"tasa_{p}"] = (vista[f"tasa_{p}"] * 100).round(2)
    print(vista[["n_val_umbral", "tasa_train", "tasa_val_ajuste", "tasa_val_umbral"]]
          .rename(columns={"n_val_umbral": "Incidentes_val_umbral", "tasa_train": "Critico_train_%",
                           "tasa_val_ajuste": "Critico_ajuste_%", "tasa_val_umbral": "Critico_umbral_%"}).to_string())
    grandes = tasas[(tasas["n_train"] >= args.min_incidentes) & (tasas["n_val_umbral"] >= args.min_incidentes)].dropna()
    print(f"\n  {len(grandes)} estados con >= {args.min_incidentes:,} incidentes en train y en val_umbral.")
    if len(grandes) >= 3:
        rho = grandes["tasa_train"].corr(grandes["tasa_val_umbral"], method="spearman")
        cambio = np.abs(np.log(grandes["tasa_val_umbral"].clip(lower=1e-4) / grandes["tasa_train"].clip(lower=1e-4)))
        print(f"  Correlacion de orden (Spearman) de la tasa train vs val_umbral: {rho:.3f}")
        print(f"  Cambio tipico de la tasa (mediana de |log(umbral/train)|): {cambio.median():.3f}  "
              f"(0.10 = 10% de cambio relativo)")
    else:
        print("  (muy pocos estados para medir la estabilidad)")

    # ------------------------------------------------------------------
    titulo("[2] CALIBRACION Y DISCRIMINACION DENTRO DE CADA ESTADO (val_umbral)")
    cal = tabla_calibracion(estado["val_umbral"], y_um, s_um, umbral, estado["val_ajuste"], y_aj, s_aj)
    print("O/E = criticos observados / criticos que el modelo esperaba (1.00 = calibrado; >1 el modelo se queda corto).")
    print("ROC-AUC dentro = que tan bien ordena el modelo los incidentes de ESE estado (0.5 = al azar).\n")
    print(cal.head(15).round({"Tasa_critico_%": 2, "Prob_media_%": 2, "OE_umbral": 2, "OE_ajuste": 2,
                              "ROC_AUC_dentro": 3, "PR_AUC_dentro": 3, "Alertas_%": 2, "Recall": 3}).to_string(index=False))
    cal.round(5).to_csv(cfg.DIR_EDA / "estados_calibracion.csv", index=False, encoding=cfg.ENCODING)
    grandes_cal = cal[cal["Criticos"] >= args.min_criticos]
    en_rango = grandes_cal[grandes_cal["OE_umbral"].between(0.8, 1.25)]
    roc_global = roc_auc_score(y_um, s_um)
    print(f"\n  Estados con >= {args.min_criticos} criticos en val_umbral: {len(grandes_cal)} "
          f"({grandes_cal['Incidentes'].sum() / len(y_um) * 100:.0f}% de los incidentes).")
    if len(grandes_cal):
        print(f"  Calibrados (O/E entre 0.80 y 1.25): {len(en_rango)} de {len(grandes_cal)}"
              f" ({en_rango['Incidentes'].sum() / grandes_cal['Incidentes'].sum() * 100:.0f}% de sus incidentes).")
    con_ambos = grandes_cal.dropna(subset=["OE_ajuste", "OE_umbral"])
    if len(con_ambos) >= 3:
        persistencia = np.log(con_ambos["OE_ajuste"].clip(lower=1e-3)).corr(np.log(con_ambos["OE_umbral"].clip(lower=1e-3)))
        print(f"  Persistencia de la descalibracion (correlacion de log O/E, val_ajuste vs val_umbral): {persistencia:.3f}"
              "  (cerca de 0: el error de un periodo no se repite en el siguiente)")
    else:
        print("  (muy pocos estados para medir la persistencia de la descalibracion)")
    if grandes_cal["ROC_AUC_dentro"].notna().any():
        print(f"  ROC-AUC dentro del estado: mediana {grandes_cal['ROC_AUC_dentro'].median():.3f} "
              f"(minimo {grandes_cal['ROC_AUC_dentro'].min():.3f}) | global {roc_global:.3f}")
    print("  Un ROC-AUC dentro del estado claramente sobre 0.5 indica que el modelo SI distingue los incidentes graves "
          "dentro de ese estado, aunque no llegue al umbral global.")

    # ------------------------------------------------------------------
    titulo("[3] MISMO NUMERO DE ALERTAS, DISTINTO REPARTO (val_umbral)")
    ajuste = ajuste_por_estado(estado["val_ajuste"], y_aj, s_aj, args.prior)
    tasa_h = tasa_historica(estado["train"], y_train)
    e_um = estado["val_umbral"]
    corr = pd.Series(e_um).map(ajuste).fillna(0.0).to_numpy()      # estado no visto en val_ajuste: sin correccion
    base_h = y_train.mean()
    rel_h = pd.Series(e_um).map(tasa_h).fillna(base_h).to_numpy()
    score = {
        "Global (actual)": s_um,
        "Recalibrado por estado": logit(s_um) + corr,
        "Relativo al estado": s_um / rel_h,
    }
    n, k = len(y_um), int(round(presupuesto * len(y_um)))
    mascaras = {nombre: marcar_top(sc, k) for nombre, sc in score.items()}
    mascaras["Cobertura pareja"] = marcar_top_por_estado(s_um, e_um, presupuesto)

    grandes_nombres = grandes_cal["Estado"].tolist()
    top_estados = cal["Estado"].head(5).tolist()
    filas, por_estado = [], pd.DataFrame({"Estado": cal["Estado"], "Criticos": cal["Criticos"]}).set_index("Estado")
    for nombre, m in mascaras.items():
        tp = int((m & (y_um == 1)).sum())
        rec_e = recall_por_estado(e_um, y_um, m)
        por_estado[f"Recall_{nombre}"] = rec_e
        por_estado[f"Alertas_%_{nombre}"] = pd.Series(m).groupby(e_um).mean() * 100
        fila = {"Estrategia": nombre, "Alertas_%": m.mean() * 100, "Recall": tp / y_um.sum(),
                "Precision": tp / max(int(m.sum()), 1),
                "PR_AUC": average_precision_score(y_um, score[nombre]) if nombre in score else np.nan,
                "ROC_AUC": roc_auc_score(y_um, score[nombre]) if nombre in score else np.nan,
                "Estados_recall>=10%": int((rec_e.reindex(grandes_nombres).fillna(0) >= 0.10).sum())}
        for e in top_estados:
            fila[f"Recall_{e}"] = rec_e.get(e, np.nan)
        filas.append(fila)
    resultado = pd.DataFrame(filas)
    print(f"  Se marcan {k:,} de {n:,} incidentes ({presupuesto * 100:.2f}%) en todas las estrategias. "
          f"Estados contados: los {len(grandes_nombres)} con >= {args.min_criticos} criticos.")
    print(f"  Columnas Recall_<estado>: los 5 estados con mas incidentes ({', '.join(top_estados)}).\n")
    print(resultado.round(4).to_string(index=False))
    resultado.round(5).to_csv(cfg.DIR_EDA / "estados_asignacion.csv", index=False, encoding=cfg.ENCODING)
    por_estado.round(5).to_csv(cfg.DIR_EDA / "estados_asignacion_por_estado.csv", encoding=cfg.ENCODING)

    # ------------------------------------------------------------------
    titulo("[4] RESULTADO")
    a, b = score["Global (actual)"], score["Recalibrado por estado"]
    delta, lo, hi = tuning.diferencia_pr_auc_bootstrap(y_um, b, a, args.n_bootstrap, cfg.SEMILLA)
    rec_a = resultado.loc[resultado["Estrategia"] == "Global (actual)", "Recall"].iloc[0]
    rec_b = resultado.loc[resultado["Estrategia"] == "Recalibrado por estado", "Recall"].iloc[0]
    rec_c = resultado.loc[resultado["Estrategia"] == "Cobertura pareja", "Recall"].iloc[0]
    print(f"  Recalibrado por estado vs global, PR-AUC en val_umbral: {average_precision_score(y_um, a):.4f} -> "
          f"{average_precision_score(y_um, b):.4f} | diferencia {delta:+.4f} (IC 95%: {lo:+.4f} a {hi:+.4f})")
    print(f"  Recall con {presupuesto * 100:.2f}% de alertas: {rec_a:.3f} -> {rec_b:.3f}")
    print(f"  => {tuning.veredicto(delta, lo, hi)}")
    print(f"\n  Costo de repartir las alertas por igual entre estados (cobertura pareja): recall global "
          f"{rec_a:.3f} -> {rec_c:.3f} ({(rec_c - rec_a) * 100:+.1f} puntos).")

    titulo("CONCLUSION")
    if lo > 0 and delta >= tuning.MEJORA_MINIMA:
        print("AYUDA: la correccion por estado, estimada en val_ajuste, mejora el ranking de forma relevante en val_umbral.")
        print("Siguiente paso: guardarla en decision.json (un ajuste de logit por estado), aplicarla en PredictorIncidentes")
        print("y medir el test UNA vez con el pipeline final. Como tiene un parametro por estado, conviene revisar su")
        print("estabilidad con otra particion antes de adoptarla.")
    else:
        print("NO AYUDA: corregir por estado no mejora el ranking de forma relevante. La concentracion de alertas viene")
        print("de la diferencia de MAGNITUD entre las tasas de criticos de los estados, no de un defecto que se corrija")
        print("por estado (aunque haya estados descalibrados, ver O/E, corregirlos casi no cambia el ranking).")
        print("Si la aplicacion necesita cubrir mas estados, es una decision de politica con costo")
        print(f"en recall global (ver 'Cobertura pareja': {(rec_c - rec_a) * 100:+.1f} puntos) y se documenta como limitacion.")
    print(f"\nTablas: {cfg.DIR_EDA / 'estados_calibracion.csv'}, estados_asignacion.csv y estados_asignacion_por_estado.csv")


if __name__ == "__main__":
    main()
