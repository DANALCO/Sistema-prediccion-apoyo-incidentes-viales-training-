"""
INGESTA DE DATOS
=============================================================================
Responsabilidad unica (SRP): LEER. Esta clase solo conecta con el CSV,
lee por trozos y recorta al alcance (fuente y fecha de inicio). No limpia,
no imputa y no entrena.

Mientras lee, acumula (sin costo extra) cuantos incidentes hay por
FUENTE, ANIO y SEVERITY en TODO el archivo. Esa tabla es la evidencia que
justifica el alcance: muestra que "Severity" no significa lo mismo en
todas las fuentes ni en todos los anios.
=============================================================================
"""

import pandas as pd


class DataIngester:

    def __init__(self, ruta_csv, columnas, encoding="utf-8", tam_trozo=500_000):
        self.ruta_csv = ruta_csv
        self.columnas = list(columnas)
        self.encoding = encoding
        self.tam_trozo = tam_trozo

    def _verificar_columnas(self):
        cabecera = set(pd.read_csv(self.ruta_csv, nrows=0, encoding=self.encoding).columns)
        faltan = [c for c in self.columnas if c not in cabecera]
        if faltan:
            raise ValueError(f"Columnas no encontradas en el CSV: {faltan}")

    def leer_alcance(self, fuente, fecha_inicio):
        """Devuelve (df, evidencia, filas_csv).

        df         : filas de `fuente` con Start_Time >= fecha_inicio (sin la
                     columna Source).
        evidencia  : DataFrame con filas y % de cada Severity por fuente y anio
                     (sobre TODO el CSV).
        filas_csv  : total de filas del archivo.
        """
        self._verificar_columnas()
        trozos, conteos, filas_csv = [], None, 0

        for ch in pd.read_csv(self.ruta_csv, usecols=self.columnas,
                              chunksize=self.tam_trozo, encoding=self.encoding):
            filas_csv += len(ch)

            texto_fecha = ch["Start_Time"].astype(str)
            anio = pd.to_numeric(texto_fecha.str[:4], errors="coerce").rename("Anio")
            parcial = ch.groupby(["Source", anio, "Severity"]).size()
            conteos = parcial if conteos is None else conteos.add(parcial, fill_value=0)

            # Las fechas ISO ("2021-03-01 ...") se comparan bien como texto.
            en_alcance = (ch["Source"] == fuente) & (texto_fecha.str[:10] >= fecha_inicio)
            trozos.append(ch[en_alcance].drop(columns="Source"))

        df = pd.concat(trozos, ignore_index=True)
        return df, self.resumir_evidencia(conteos), filas_csv

    @staticmethod
    def resumir_evidencia(conteos):
        tabla = conteos.unstack("Severity", fill_value=0)
        filas = tabla.sum(axis=1)
        pct = tabla.div(filas, axis=0) * 100
        pct.columns = [f"Sev_{int(c)}_%" for c in pct.columns]
        pct.insert(0, "Filas", filas.astype("int64"))
        pct["Baseline_exactitud_%"] = pct.filter(like="Sev_").max(axis=1)
        pct = pct.round(2).reset_index()
        pct["Anio"] = pct["Anio"].astype(int)
        return pct
