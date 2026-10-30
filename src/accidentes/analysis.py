"""
ANALISIS DESCRIPTIVO Y DE ERRORES
=============================================================================
Responsabilidad unica: producir las TABLAS y GRAFICOS que explican los datos
y los errores del modelo. No entrena nada.

Dos usos:
  1. PATRONES DE RIESGO (sin modelo). Para cada categoria (por ejemplo "lluvia",
     "hora 8", "hay semaforo"): cuantos incidentes hay, que % fue critico,
     cuantas veces la tasa general y un intervalo de confianza del 95%.
     Responde las preguntas de investigacion 1 (clima y visibilidad),
     2 (patrones temporales) y 3 (infraestructura).
  2. ERRORES DEL MODELO. Por segmento (estado, hora, clima...): recall,
     precision, falsos positivos y falsos negativos, y cuanto del total de
     criticos que el modelo no detecta cae en cada segmento.

COMO LEER LAS TABLAS. Con millones de filas, hasta diferencias minimas salen
"estadisticamente distintas". Importa el TAMANO del efecto (columna
Veces_tasa_base), no solo si el intervalo excluye la tasa general. Una
asociacion debil es una respuesta valida, no un fallo.
=============================================================================
"""

import numpy as np
import pandas as pd

# Intervalos para variables numericas (limite inferior incluido, superior excluido)
BINS = {
    "Visibility(mi)":    ([-np.inf, 1, 3, 6, 10, np.inf], ["<1", "1-3", "3-6", "6-10", ">=10"]),
    "Precipitation(in)": ([-np.inf, 1e-9, 0.05, 0.2, np.inf], ["0", "0-0.05", "0.05-0.2", ">=0.2"]),
    "Wind_Speed(mph)":   ([-np.inf, 5, 10, 15, 20, np.inf], ["<5", "5-10", "10-15", "15-20", ">=20"]),
    "Temperature(F)":    ([-np.inf, 32, 50, 70, 90, np.inf], ["<32", "32-50", "50-70", "70-90", ">=90"]),
    "Humidity(%)":       ([-np.inf, 40, 60, 80, np.inf], ["<40", "40-60", "60-80", ">=80"]),
}
NOMBRES_MES = {1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
               7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic"}
NOMBRES_DIA = {0: "lun", 1: "mar", 2: "mie", 3: "jue", 4: "vie", 5: "sab", 6: "dom"}
MIN_CASOS_CONFIABLE = 300    # categorias con menos incidentes se marcan como poco confiables


def intervalo_wilson(exitos, total, z=1.96):
    """Intervalo de confianza del 95% para una proporcion (metodo de Wilson, en %)."""
    exitos = np.asarray(exitos, dtype="float64")
    total = np.asarray(total, dtype="float64")
    p = np.divide(exitos, total, out=np.zeros_like(exitos), where=total > 0)
    denom = 1 + z ** 2 / np.where(total > 0, total, 1)
    centro = (p + z ** 2 / (2 * np.where(total > 0, total, 1))) / denom
    mitad = z * np.sqrt(p * (1 - p) / np.where(total > 0, total, 1)
                        + z ** 2 / (4 * np.where(total > 0, total, 1) ** 2)) / denom
    return (centro - mitad) * 100, (centro + mitad) * 100


def _categorias(df, variable):
    serie = df[variable]
    if variable in BINS:
        bordes, etiquetas = BINS[variable]
        serie = pd.cut(serie, bordes, labels=etiquetas, right=False).astype(object)
    elif str(serie.dtype) in ("bool", "boolean"):
        serie = serie.map({True: "Si", False: "No"})
    else:
        serie = serie.astype(object)
    return serie.where(serie.notna(), "sin dato")


def _ordenar(tabla, variable):
    if variable in BINS:
        orden = {e: i for i, e in enumerate(BINS[variable][1])}
        return tabla.sort_values("Categoria", key=lambda s: s.map(lambda x: orden.get(x, 99)))
    if variable in ("Hora", "Dia_semana", "Mes"):
        return tabla.sort_values("Categoria", key=lambda s: pd.to_numeric(s, errors="coerce"))
    return tabla.sort_values("Incidentes", ascending=False)


def tabla_patrones(df, col_objetivo, preguntas):
    """preguntas: {numero: (nombre, [variables])}. Devuelve una tabla ordenada y 'ancha'."""
    y = df[col_objetivo].to_numpy()
    base = y.mean() * 100
    filas = []
    for numero, (nombre, variables) in preguntas.items():
        for variable in variables:
            cat = _categorias(df, variable)
            g = pd.DataFrame({"Categoria": cat.to_numpy(), "y": y}).groupby("Categoria", observed=True).agg(
                Incidentes=("y", "size"), Criticos=("y", "sum")).reset_index()
            g["Variable"], g["Pregunta"], g["Tema"] = variable, numero, nombre
            g = _ordenar(g, variable)
            filas.append(g)
    t = pd.concat(filas, ignore_index=True)
    t["Categoria"] = t["Categoria"].astype(str)
    t["Tasa_critico_%"] = t["Criticos"] / t["Incidentes"] * 100
    t["Veces_tasa_base"] = t["Tasa_critico_%"] / base
    t["IC95_inf_%"], t["IC95_sup_%"] = intervalo_wilson(t["Criticos"], t["Incidentes"])
    t["Distinta_de_la_tasa_base"] = (t["IC95_sup_%"] < base) | (t["IC95_inf_%"] > base)
    t["Pocos_casos"] = t["Incidentes"] < MIN_CASOS_CONFIABLE
    columnas = ["Pregunta", "Tema", "Variable", "Categoria", "Incidentes", "Criticos", "Tasa_critico_%",
                "Veces_tasa_base", "IC95_inf_%", "IC95_sup_%", "Distinta_de_la_tasa_base", "Pocos_casos"]
    return t[columnas], base


def tabla_segmentos(df, y, pred, variable, maximo=None):
    """Errores del modelo por categoria de `variable`."""
    cat = _categorias(df, variable).to_numpy()
    d = pd.DataFrame({"Categoria": cat, "y": y.astype(int), "pred": pred.astype(int)})
    d["tp"] = (d.y == 1) & (d.pred == 1)
    d["fp"] = (d.y == 0) & (d.pred == 1)
    d["fn"] = (d.y == 1) & (d.pred == 0)
    t = d.groupby("Categoria", observed=True).agg(
        Incidentes=("y", "size"), Criticos=("y", "sum"), Alertas=("pred", "sum"),
        TP=("tp", "sum"), FP=("fp", "sum"), FN=("fn", "sum")).reset_index()
    base = y.mean()
    t["Variable"] = variable
    t["Tasa_critico_%"] = t["Criticos"] / t["Incidentes"] * 100
    t["Alertas_%"] = t["Alertas"] / t["Incidentes"] * 100
    t["Recall"] = np.where(t["Criticos"] > 0, t["TP"] / t["Criticos"].replace(0, np.nan), np.nan)
    t["Precision"] = np.where(t["Alertas"] > 0, t["TP"] / t["Alertas"].replace(0, np.nan), np.nan)
    t["Pct_de_los_FN"] = t["FN"] / max(t["FN"].sum(), 1) * 100
    t["Pct_de_los_FP"] = t["FP"] / max(t["FP"].sum(), 1) * 100
    t["Veces_tasa_base"] = t["Tasa_critico_%"] / (base * 100)
    t = _ordenar(t, variable)
    if maximo:
        t = t.sort_values("Incidentes", ascending=False).head(maximo)
    t["Categoria"] = t["Categoria"].astype(str)
    return t


def tabla_estabilidad_mensual(fechas, y, pred):
    """Tasa de criticos, de alertas, recall y exactitud por mes: muestra la deriva en el tiempo."""
    d = pd.DataFrame({"Mes": pd.to_datetime(fechas).dt.to_period("M").astype(str),
                      "y": y.astype(int), "pred": pred.astype(int)})
    d["tp"] = (d.y == 1) & (d.pred == 1)
    d["ok"] = d.y == d.pred
    t = d.groupby("Mes").agg(Incidentes=("y", "size"), Criticos=("y", "sum"), Alertas=("pred", "sum"),
                             TP=("tp", "sum"), Aciertos=("ok", "sum")).reset_index()
    t["Tasa_critico_%"] = t["Criticos"] / t["Incidentes"] * 100
    t["Alertas_%"] = t["Alertas"] / t["Incidentes"] * 100
    t["Recall"] = t["TP"] / t["Criticos"].replace(0, np.nan)
    t["Precision"] = t["TP"] / t["Alertas"].replace(0, np.nan)
    t["Exactitud"] = t["Aciertos"] / t["Incidentes"]
    t["Baseline_exactitud"] = 1 - t["Criticos"] / t["Incidentes"]
    return t


# ---------------------------------------------------------------------------
# Graficos (matplotlib, sin pantalla)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def grafico_tasas(subtabla, base, titulo, ruta, linea=False):
    """Tasa de criticos por categoria con su intervalo de confianza y la tasa general."""
    plt = _plt()
    sub = subtabla.copy()
    x = np.arange(len(sub))
    y = sub["Tasa_critico_%"].to_numpy()
    err = np.vstack([y - sub["IC95_inf_%"].to_numpy(), sub["IC95_sup_%"].to_numpy() - y])
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sub) + 2), 4.2))
    if linea:
        ax.errorbar(x, y, yerr=err, fmt="-o", color="#2b6cb0", capsize=2, ms=4)
    else:
        ax.bar(x, y, yerr=err, color="#2b6cb0", capsize=3)
    ax.axhline(base, color="#c53030", ls="--", lw=1, label=f"tasa general {base:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Categoria"], rotation=45 if len(sub) > 8 else 0, ha="right" if len(sub) > 8 else "center")
    ax.set_ylabel("% de incidentes criticos")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafico_recall_precision(tabla, titulo, ruta, recall_global, precision_global):
    plt = _plt()
    t = tabla.sort_values("Incidentes", ascending=False)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(t) + 2), 4.2))
    ax.bar(x - 0.2, t["Recall"].fillna(0) * 100, 0.4, label="Recall (criticos detectados)", color="#2b6cb0")
    ax.bar(x + 0.2, t["Precision"].fillna(0) * 100, 0.4, label="Precision (alertas acertadas)", color="#c05621")
    ax.axhline(recall_global * 100, color="#2b6cb0", ls="--", lw=1)
    ax.axhline(precision_global * 100, color="#c05621", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(t["Categoria"], rotation=45, ha="right")
    ax.set_ylabel("%")
    ax.set_title(titulo + "\n(lineas punteadas: valor global)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafico_estabilidad(tabla, titulo, ruta):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tabla["Mes"], tabla["Tasa_critico_%"], "-o", label="% de incidentes criticos", color="#2b6cb0")
    ax.plot(tabla["Mes"], tabla["Alertas_%"], "-s", label="% de incidentes marcados como criticos", color="#c05621")
    ax.set_ylabel("%")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
