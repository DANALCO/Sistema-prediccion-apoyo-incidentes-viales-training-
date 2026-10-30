"""
UTILIDADES COMPARTIDAS
=============================================================================
Responsabilidad unica: pequenas herramientas de apoyo que no pertenecen a
ninguna etapa del pipeline (titulos de consola, memoria, bitacora).

La Bitacora responde al criterio "Simulacion de Resultados" de la rubrica
(trazabilidad): documenta cuantas filas se pierden en cada paso y por que.
=============================================================================
"""

import pandas as pd


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def memoria_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


class Bitacora:
    """Registro de filas eliminadas por paso (para anexarlo al informe)."""

    def __init__(self):
        self._filas = []

    def registrar(self, paso, filas_antes, filas_despues, motivo):
        eliminadas = filas_antes - filas_despues
        pct = (eliminadas / filas_antes * 100) if filas_antes else 0.0
        self._filas.append({
            "Paso": paso,
            "Filas antes": filas_antes,
            "Filas despues": filas_despues,
            "Eliminadas": eliminadas,
            "% eliminado": round(pct, 3),
            "Motivo": motivo,
        })
        print(f"  [{paso}] {filas_antes:,} -> {filas_despues:,} "
              f"({eliminadas:,} eliminadas, {pct:.3f}%)")

    def a_dataframe(self):
        return pd.DataFrame(self._filas)
