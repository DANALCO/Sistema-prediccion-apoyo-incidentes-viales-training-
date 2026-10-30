"""
=============================================================================
EDA  -  PATRONES DE RIESGO  (preguntas de investigacion 1, 2 y 3)
=============================================================================
Entrada : outputs/limpio/us_accidents_limpio.parquet   (script 01)
Salida  : eda/figures/patrones_de_riesgo.csv           tabla completa
          eda/figures/patron_*.png                      graficos

NO usa ningun modelo: describe, sobre los datos del alcance, que porcentaje de
los incidentes fue critico en cada condicion.

    Pregunta 1  clima y visibilidad  (tipo de clima, visibilidad, lluvia, viento...)
    Pregunta 2  patrones temporales  (hora, dia, mes, estacion, hora pico...)
    Pregunta 3  infraestructura      (semaforos, cruces, rotondas...)

COMO LEER EL RESULTADO
  - 'Veces_tasa_base' es el TAMANO del efecto: 1.0 = igual que el promedio,
    2.0 = el doble de probable. Es lo que importa.
  - Con millones de filas, casi toda diferencia sale "estadisticamente distinta"
    (columna Distinta_de_la_tasa_base). Eso no significa que sea importante.
  - Una asociacion debil es una respuesta VALIDA: no es un fallo del analisis.
  - Es asociacion, no causa.

Uso (desde la raiz del repositorio):   python eda/patrones_de_riesgo.py
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import pandas as pd

from accidentes import config as cfg
from accidentes.analysis import MIN_CASOS_CONFIABLE, grafico_tasas, tabla_patrones
from accidentes.storage import leer_tabla
from accidentes.utils import titulo

MIN_PARA_RESUMEN = 1000     # categorias con menos incidentes no entran en el resumen por variable


def main():
    titulo("PATRONES DE RIESGO - carga")
    variables = sorted({v for _, (_, vs) in cfg.PREGUNTAS_INVESTIGACION.items() for v in vs})
    df = leer_tabla(str(cfg.DIR_LIMPIO / "us_accidents_limpio"))[variables + [cfg.COL_OBJETIVO]]
    print(f"Incidentes: {len(df):,} | criticos: {int(df[cfg.COL_OBJETIVO].sum()):,}")

    tabla, base = tabla_patrones(df, cfg.COL_OBJETIVO, cfg.PREGUNTAS_INVESTIGACION)
    cfg.DIR_EDA.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(cfg.DIR_EDA / "patrones_de_riesgo.csv", index=False, encoding=cfg.ENCODING)
    print(f"Tasa general de criticos: {base:.3f}%")

    # --- Resumen: cuanto varia el riesgo dentro de cada variable (tamano del efecto) ---
    titulo("TAMANO DEL EFECTO POR VARIABLE (categorias con >= %d incidentes)" % MIN_PARA_RESUMEN)
    suficiente = tabla[tabla["Incidentes"] >= MIN_PARA_RESUMEN]
    observadas = suficiente[suficiente["Categoria"] != "sin dato"]     # condiciones realmente medidas
    faltantes = suficiente[suficiente["Categoria"] == "sin dato"]      # el dato no estaba registrado
    resumen = observadas.groupby(["Pregunta", "Variable"]).agg(
        Min_veces_tasa_base=("Veces_tasa_base", "min"),
        Max_veces_tasa_base=("Veces_tasa_base", "max")).reset_index()
    resumen["Rango"] = resumen["Max_veces_tasa_base"] - resumen["Min_veces_tasa_base"]
    resumen = resumen.sort_values("Rango", ascending=False)
    print("Solo cuentan condiciones realmente medidas: 'sin dato' se analiza aparte, abajo.")
    print("Rango = distancia entre la categoria mas riesgosa y la menos riesgosa (en veces la tasa base).")
    print("Un rango cercano a 0 significa que la variable casi no distingue.\n")
    print(resumen.round(2).to_string(index=False))

    # --- Datos faltantes ---
    if len(faltantes):
        titulo("CUANDO FALTA EL DATO (categoria 'sin dato')")
        print(faltantes[["Variable", "Incidentes", "Tasa_critico_%", "Veces_tasa_base", "IC95_inf_%",
                         "IC95_sup_%"]].round(2).sort_values("Veces_tasa_base", ascending=False)
              .to_string(index=False))
        print("\nSi la tasa de criticos es MUY distinta cuando falta el dato, el hecho de que falte")
        print("es una pista sobre COMO o DONDE se registro el incidente, no un efecto fisico del clima.")
        print("Es un artefacto de los datos: conviene declararlo como limitacion y NO explotarlo.")

    # --- Por pregunta: lo mas y lo menos riesgoso ---
    for numero, (nombre, _) in cfg.PREGUNTAS_INVESTIGACION.items():
        titulo(f"PREGUNTA {numero} - {nombre.upper()}")
        sub = observadas[observadas["Pregunta"] == numero].sort_values("Veces_tasa_base", ascending=False)
        columnas = ["Variable", "Categoria", "Incidentes", "Tasa_critico_%", "Veces_tasa_base",
                    "IC95_inf_%", "IC95_sup_%"]
        print("Condiciones MAS riesgosas:")
        print(sub.head(6)[columnas].round(2).to_string(index=False))
        print("\nCondiciones MENOS riesgosas:")
        print(sub.tail(4)[columnas].round(2).to_string(index=False))

    # --- Graficos ---
    def sub(variable):
        return tabla[(tabla["Variable"] == variable) & (~tabla["Pocos_casos"]) & (tabla["Categoria"] != "sin dato")]
    grafico_tasas(sub("Hora"), base, "% de incidentes criticos por hora del dia (con IC 95%)",
                  cfg.DIR_EDA / "patron_hora.png", linea=True)
    grafico_tasas(sub("Dia_semana"), base, "% de incidentes criticos por dia de la semana (0 = lunes)",
                  cfg.DIR_EDA / "patron_dia_semana.png")
    grafico_tasas(sub("Weather_Condition"), base, "% de incidentes criticos por condicion climatica",
                  cfg.DIR_EDA / "patron_clima.png")
    grafico_tasas(sub("Visibility(mi)"), base, "% de incidentes criticos por visibilidad (millas)",
                  cfg.DIR_EDA / "patron_visibilidad.png")
    infra = tabla[(tabla["Variable"].isin(cfg.INFRAESTRUCTURA)) & (tabla["Categoria"] == "Si")
                  & (~tabla["Pocos_casos"])].copy()
    infra["Categoria"] = infra["Variable"]
    grafico_tasas(infra, base, "% de incidentes criticos cuando HAY cada elemento de infraestructura",
                  cfg.DIR_EDA / "patron_infraestructura.png")

    pocos = int(tabla["Pocos_casos"].sum())
    print("\n" + "=" * 70)
    print("PATRONES COMPLETADOS")
    print(f"Tabla    -> {cfg.DIR_EDA / 'patrones_de_riesgo.csv'}")
    print(f"Graficos -> {cfg.DIR_EDA}")
    if pocos:
        print(f"Aviso: {pocos} categorias tienen menos de {MIN_CASOS_CONFIABLE} incidentes y no se grafican.")
    print("=" * 70)


if __name__ == "__main__":
    main()
