"""
=============================================================================
SCRIPT 03  -  ENTRENAMIENTO E INTEGRACION DE LOS 4 MODELOS  (orquestador)
=============================================================================
Entrada : outputs/preparado/datos_<escenario>.npz   (lo genera el script 02)
Salida  : outputs/modelos/<escenario>/
              modelo_<nombre>.joblib        los 4 modelos entrenados
              calibradores.joblib           calibracion de Platt por modelo
              apilador.joblib               stacking (regresion logistica)
              decision.json                 umbral, franja de vigilancia, pesos
              tabla_comparativa.csv         4 individuales + 3 integraciones
              zonas_vigilancia.csv          % de criticos reales por zona
              comparacion_balanceo.csv      con y sin SMOTENC
              puntajes.npz                  puntajes por incidente (para SHAP/errores)
          outputs/modelos/perdida_tiempo_real.csv   (si se corrieron los 2 escenarios)

Este script solo ORQUESTA. Cada cosa vive en su modulo (SRP):
    models.py      los 4 modelos base
    ensemble.py    calibracion, promedio, stacking, umbral, vigilancia
    evaluation.py  metricas
    training.py    el flujo de un escenario

SOLUCION FINAL: STACKING de Random Forest + XGBoost + LightGBM + Regresion
Logistica. El promedio simple y el ponderado se reportan como comparacion.

Uso (desde la raiz del repositorio):
    python pipeline/03_entrenar_modelos.py                    # ambos escenarios
    python pipeline/03_entrenar_modelos.py --escenario tiempo_real
    python pipeline/03_entrenar_modelos.py --escenario tiempo_real --solo-balanceo
    python pipeline/03_entrenar_modelos.py --escenario tiempo_real --estrategia Sin_balancear

Tiempo: decenas de minutos por escenario (estimado). Solo el escenario
'tiempo_real' opera en la aplicacion; 'informacion_completa' es la referencia
para medir cuanto se pierde al operar en tiempo real (pregunta de
investigacion 4).
=============================================================================
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # permite importar `accidentes`

import pandas as pd

from accidentes import config as cfg
from accidentes.evaluation import comparar_escenarios
from accidentes.training import comparar_balanceo_escenario, entrenar_escenario
from accidentes.utils import titulo


def main():
    parser = argparse.ArgumentParser(description="Entrena e integra los 4 modelos.")
    parser.add_argument("--escenario", choices=[*cfg.ESCENARIOS, "ambos"], default="ambos")
    parser.add_argument("--estrategia", choices=cfg.ESTRATEGIAS_BALANCEO, default=cfg.ESTRATEGIA_FINAL,
                        help="como se maneja el desbalance al entrenar (por defecto, ESTRATEGIA_FINAL de config.py)")
    parser.add_argument("--solo-balanceo", action="store_true",
                        help="solo compara las estrategias de balanceo con LightGBM (unos minutos)")
    args = parser.parse_args()
    cfg.ESTRATEGIA_FINAL = args.estrategia
    escenarios = list(cfg.ESCENARIOS) if args.escenario == "ambos" else [args.escenario]

    if args.solo_balanceo:
        for nombre in escenarios:
            comparar_balanceo_escenario(nombre)
        return

    cfg.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    tablas = {nombre: entrenar_escenario(nombre) for nombre in escenarios}

    # Pregunta de investigacion 4: usa lo recien entrenado o lo guardado antes.
    for nombre in cfg.ESCENARIOS:
        if nombre not in tablas:
            ruta = cfg.DIR_MODELOS / nombre / "tabla_comparativa.csv"
            if ruta.exists():
                tablas[nombre] = pd.read_csv(ruta, encoding=cfg.ENCODING)

    if len(tablas) == len(cfg.ESCENARIOS):
        titulo("PREGUNTA DE INVESTIGACION 4 - PERDIDA POR OPERAR EN TIEMPO REAL (test)")
        perdida = comparar_escenarios(tablas["tiempo_real"], tablas["informacion_completa"])
        columnas = ["Metodo", "ROC_AUC_tiempo_real", "ROC_AUC_completa", "ROC_AUC_perdida",
                    "PR_AUC_tiempo_real", "PR_AUC_completa", "PR_AUC_perdida",
                    "Recall_perdida", "Exactitud_perdida"]
        print(perdida[columnas].round(4).to_string(index=False))
        print("\n'perdida' = completa - tiempo_real (positivo: la informacion posterior ayuda).")
        ruta = cfg.DIR_MODELOS / "perdida_tiempo_real.csv"
        perdida.to_csv(ruta, index=False, encoding=cfg.ENCODING)
        print(f"  -> {ruta}")

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("Recuerde: no ajuste hiperparametros mirando la columna 'test'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
