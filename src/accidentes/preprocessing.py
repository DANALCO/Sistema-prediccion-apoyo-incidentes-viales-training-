"""
PREPROCESAMIENTO (todo lo que "aprende" de los datos)
=============================================================================
Responsabilidad unica: transformar tablas limpias en matrices numericas.

REGLA DE ORO de la clase: TRAIN aprende parametros; VAL/TEST solo transforman.
Por eso hay un metodo fit() (se llama UNA vez, con train) y metodos de
transformacion que solo aplican lo aprendido. Lo que se aprende:
    - medianas por estado y medianas globales  (imputacion)
    - limites IQR                               (recorte de outliers)
    - escaladores RobustScaler                  (escalado)
    - categorias de cada variable               (codificacion)

El mismo objeto se guarda en disco y lo reutiliza la aplicacion, de modo que
el formulario del tablero aplica exactamente las mismas transformaciones.

DOS REPRESENTACIONES (necesarias porque se usa SMOTENC):
    mixto  : continuas escaladas | discretas | categoricas (como codigos) |
             booleanas. SMOTENC necesita las categoricas SIN one-hot.
    matriz : continuas | discretas escaladas | one-hot | booleanas.
             Es lo que reciben los modelos.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class Preprocesador:

    def __init__(self, continuas, a_imputar, a_recortar, discretas, categoricas,
                 booleanas, columna_grupo="State", valor_desconocido="Desconocido",
                 factor_iqr=1.5):
        self.continuas = list(continuas)
        self.a_imputar = list(a_imputar)
        self.a_recortar = list(a_recortar)
        self.discretas = list(discretas)
        self.categoricas = list(categoricas)
        self.booleanas = list(booleanas)
        self.columna_grupo = columna_grupo
        self.valor_desconocido = valor_desconocido
        self.factor_iqr = factor_iqr
        self.ajustado_ = False

    # ------------------------------------------------------------------
    # Columnas necesarias en la tabla de entrada
    # ------------------------------------------------------------------
    @property
    def columnas_entrada(self):
        cols = self.continuas + self.discretas + self.categoricas + self.booleanas
        if self.columna_grupo not in cols:
            cols = cols + [self.columna_grupo]
        return cols

    # ------------------------------------------------------------------
    # APRENDIZAJE (solo con train)
    # ------------------------------------------------------------------
    def fit(self, df_train):
        d = df_train[self.columnas_entrada]
        grupo = d[self.columna_grupo].astype(object).fillna(self.valor_desconocido)

        self.medianas_grupo_ = {
            c: d[c].groupby(grupo).median() for c in self.a_imputar
        }
        self.medianas_globales_ = {c: d[c].median() for c in self.a_imputar}
        self.nulos_train_ = {c: int(d[c].isna().sum()) for c in self.a_imputar}

        d = self._imputar(d)

        # Limites IQR. Si IQR = 0 (mas de la mitad de los valores son iguales,
        # tipico de Precipitation en 0) NO se recorta: la formula convertiria
        # la columna en una constante y borraria la informacion util.
        self.limites_iqr_ = {}
        self.columnas_sin_recorte_ = []
        for c in self.a_recortar:
            q1, q3 = d[c].quantile(0.25), d[c].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                self.columnas_sin_recorte_.append(c)
                continue
            self.limites_iqr_[c] = (q1 - self.factor_iqr * iqr, q3 + self.factor_iqr * iqr)
        d = self._recortar(d)

        self.escalador_continuas_ = RobustScaler().fit(d[self.continuas].to_numpy("float64"))
        self.escalador_discretas_ = RobustScaler().fit(d[self.discretas].to_numpy("float64"))

        # Categorias aprendidas en train; 'Desconocido' siempre existe para que
        # la aplicacion pueda usarla cuando falte un dato.
        self.categorias_ = {}
        for c in self.categoricas:
            vistas = set(d[c].astype(object).dropna().unique()) | {self.valor_desconocido}
            self.categorias_[c] = sorted(vistas)

        self.ajustado_ = True
        return self

    # ------------------------------------------------------------------
    # TRANSFORMACION (solo aplica lo aprendido)
    # ------------------------------------------------------------------
    def _imputar(self, d):
        d = d.copy()
        grupo = d[self.columna_grupo].astype(object).fillna(self.valor_desconocido)
        for c in self.a_imputar:
            if d[c].isna().any():
                por_grupo = grupo.map(self.medianas_grupo_[c])
                d[c] = d[c].fillna(por_grupo).fillna(self.medianas_globales_[c])
        for c in self.categoricas:
            d[c] = d[c].astype(object).where(d[c].notna(), self.valor_desconocido)
        for c in self.booleanas:
            d[c] = d[c].fillna(False).astype("int8")
        return d

    def _recortar(self, d):
        for c, (inferior, superior) in self.limites_iqr_.items():
            d[c] = d[c].clip(lower=inferior, upper=superior)
        return d

    def transformar_mixto(self, df):
        """Continuas escaladas | discretas | categoricas como codigos | booleanas."""
        self._exigir_ajuste()
        d = self._recortar(self._imputar(df[self.columnas_entrada]))

        faltantes = [c for c in self.continuas + self.discretas if d[c].isna().any()]
        if faltantes:
            raise ValueError(f"Hay valores nulos en columnas sin imputacion: {faltantes}")

        partes = {}
        cont = self.escalador_continuas_.transform(d[self.continuas].to_numpy("float64"))
        for i, c in enumerate(self.continuas):
            partes[c] = cont[:, i]
        for c in self.discretas:
            partes[c] = d[c].to_numpy("float64")
        for c in self.categoricas:
            codigos = pd.Categorical(d[c], categories=self.categorias_[c]).codes
            partes[c] = codigos.astype("float64")          # -1 = categoria no vista
        for c in self.booleanas:
            partes[c] = d[c].to_numpy("float64")
        return pd.DataFrame(partes, index=d.index)

    @property
    def posiciones_categoricas_mixto(self):
        """Posiciones (en la tabla 'mixta') de lo que SMOTENC trata como categorico."""
        return list(range(len(self.continuas),
                          len(self.continuas) + len(self.discretas)
                          + len(self.categoricas) + len(self.booleanas)))

    def a_matriz(self, mixto):
        """Convierte la tabla 'mixta' en la matriz final float32 para los modelos."""
        self._exigir_ajuste()
        n = len(mixto)
        nc, nd = len(self.continuas), len(self.discretas)
        tam_ohe = sum(len(self.categorias_[c]) for c in self.categoricas)
        X = np.zeros((n, nc + nd + tam_ohe + len(self.booleanas)), dtype="float32")

        X[:, :nc] = mixto[self.continuas].to_numpy("float64")
        X[:, nc:nc + nd] = self.escalador_discretas_.transform(
            mixto[self.discretas].to_numpy("float64"))

        desplazamiento = nc + nd
        filas = np.arange(n)
        for c in self.categoricas:
            codigos = np.rint(mixto[c].to_numpy()).astype("int64")
            validas = codigos >= 0                        # -1 -> todo ceros (no vista)
            X[filas[validas], desplazamiento + codigos[validas]] = 1.0
            desplazamiento += len(self.categorias_[c])

        X[:, desplazamiento:] = mixto[self.booleanas].to_numpy("float64")
        return X

    def transformar(self, df):
        """Atajo: tabla limpia -> matriz final (lo que usa la aplicacion)."""
        return self.a_matriz(self.transformar_mixto(df))

    @property
    def nombres_features(self):
        self._exigir_ajuste()
        nombres = list(self.continuas) + list(self.discretas)
        for c in self.categoricas:
            nombres += [f"{c}_{cat}" for cat in self.categorias_[c]]
        return nombres + list(self.booleanas)

    def _exigir_ajuste(self):
        if not self.ajustado_:
            raise RuntimeError("Llame primero a fit() con los datos de entrenamiento")
