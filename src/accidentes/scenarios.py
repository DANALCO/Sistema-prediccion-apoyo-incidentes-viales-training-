"""
ESCENARIOS DE MODELADO
=============================================================================
Responsabilidad unica: armar las matrices de un escenario y guardarlas.

    tiempo_real            solo informacion disponible AL REPORTAR el incidente.
                           Es el escenario que opera en la aplicacion.
    informacion_completa   anade Distance(mi), que solo se conoce DESPUES.
                           Es la referencia contra la que se mide cuanto se
                           pierde por operar en tiempo real (pregunta de
                           investigacion 4). NO opera en la aplicacion.

Ambos escenarios usan las MISMAS filas y la misma particion temporal; solo
cambian las columnas. Si cambiaran las filas, no se sabria si la diferencia
de desempeno se debe a la informacion extra o a que a un modelo le tocaron
casos mas faciles.
=============================================================================
"""

import os
import time

import joblib
import numpy as np

from accidentes import config as cfg
from accidentes.balancing import aplicar_smotenc, composicion, recortar_normales
from accidentes.preprocessing import Preprocesador
from accidentes.utils import titulo

PARTICIONES_EVALUACION = ("val_ajuste", "val_umbral", "test")


def construir_preprocesador(cfg_escenario):
    return Preprocesador(
        continuas=cfg_escenario["continuas"],
        a_imputar=cfg_escenario["a_imputar"],
        a_recortar=cfg_escenario["a_recortar"],
        discretas=cfg.DISCRETAS,
        categoricas=cfg.CATEGORICAS,
        booleanas=cfg.BOOLEANAS,
        columna_grupo=cfg.COLUMNA_GRUPO_IMPUTACION,
        valor_desconocido=cfg.VALOR_DESCONOCIDO,
        factor_iqr=cfg.FACTOR_IQR,
    )


def preparar_escenario(nombre, cfg_escenario, df, posiciones):
    """Devuelve un diccionario con matrices, preprocesador y composicion de clases."""
    titulo(f"ESCENARIO: {nombre}")
    t0 = time.time()
    pre = construir_preprocesador(cfg_escenario)
    train = df.iloc[posiciones["train"]]
    y_train = train[cfg.COL_OBJETIVO].to_numpy()

    print("\n[1] Aprendizaje de parametros (SOLO con train)")
    pre.fit(train)
    con_nulos = {c: n for c, n in pre.nulos_train_.items() if n}
    print(f"    Nulos imputados con mediana por {cfg.COLUMNA_GRUPO_IMPUTACION}: "
          f"{con_nulos if con_nulos else 'ninguno'}")
    for c, (lo, hi) in pre.limites_iqr_.items():
        print(f"    IQR {c:<20} recorte a [{lo:,.2f}, {hi:,.2f}]")
    for c in pre.columnas_sin_recorte_:
        print(f"    AVISO: {c} tiene IQR = 0; se omite el recorte para no volverla constante")

    print("\n[2] Transformacion de train")
    mixto = pre.transformar_mixto(train)
    X_train_real = pre.a_matriz(mixto)
    comps = [composicion("train (real)", y_train)]

    print("\n[3] Balanceo con SMOTENC (SOLO sobre train)")
    X_cap, y_cap = recortar_normales(mixto, y_train, cfg.MAX_FILAS_NORMALES_ENTRENAMIENTO, cfg.SEMILLA)
    if len(y_cap) != len(y_train):
        comps.append(composicion("train (con tope de normales)", y_cap))
    mixto_bal, y_bal = aplicar_smotenc(
        X_cap, y_cap, pre.posiciones_categoricas_mixto,
        ratio=cfg.RATIO_SMOTE, k_vecinos=cfg.K_VECINOS_SMOTE,
        lote=cfg.LOTE_SMOTE, semilla=cfg.SEMILLA,
    )
    X_train_bal = pre.a_matriz(mixto_bal)
    comps.append(composicion("train (SMOTENC)", y_bal))
    del mixto, mixto_bal, X_cap

    print("\n[4] Validacion y test: SOLO se transforman (nunca se balancean)")
    evaluacion = {}
    for particion in PARTICIONES_EVALUACION:
        sub = df.iloc[posiciones[particion]]
        y_sub = sub[cfg.COL_OBJETIVO].to_numpy()
        evaluacion[particion] = (pre.transformar(sub), y_sub)
        comps.append(composicion(f"{particion} (real)", y_sub))

    for fila in comps:
        fila["Escenario"] = nombre
        print(f"    {fila['Etapa']:<30} {fila['Filas']:>10,} filas | criticos "
              f"{fila['Criticos']:>9,} ({fila['Pct_critico']:6.2f}%)")
    print(f"\n    Columnas finales: {X_train_bal.shape[1]} | tiempo del escenario: "
          f"{time.time() - t0:.0f} s")

    return {
        "preprocesador": pre,
        "X_train_bal": X_train_bal, "y_train_bal": y_bal.astype("int8"),
        "X_train_real": X_train_real, "y_train_real": y_train.astype("int8"),
        "evaluacion": evaluacion,
        "composicion": comps,
    }


def guardar_escenario(nombre, resultado, posiciones, directorio):
    os.makedirs(directorio, exist_ok=True)
    datos = {
        "X_train_bal": resultado["X_train_bal"], "y_train_bal": resultado["y_train_bal"],
    }
    if cfg.GUARDAR_TRAIN_REAL:
        datos["X_train_real"] = resultado["X_train_real"]
        datos["y_train_real"] = resultado["y_train_real"]
    for particion, (X, y) in resultado["evaluacion"].items():
        datos[f"X_{particion}"] = X
        datos[f"y_{particion}"] = y.astype("int8")
        datos[f"pos_{particion}"] = posiciones[particion].astype("int64")   # alinea con la tabla limpia

    ruta_npz = os.path.join(directorio, f"datos_{nombre}.npz")
    np.savez_compressed(ruta_npz, **datos)

    ruta_pre = os.path.join(directorio, f"preprocesador_{nombre}.joblib")
    joblib.dump(resultado["preprocesador"], ruta_pre)

    ruta_txt = os.path.join(directorio, f"features_{nombre}.txt")
    with open(ruta_txt, "w", encoding=cfg.ENCODING) as f:
        f.write("\n".join(resultado["preprocesador"].nombres_features))

    print(f"\n  {ruta_npz}")
    print(f"  {ruta_pre}   (lo reutiliza la aplicacion)")
    print(f"  {ruta_txt}   (nombres de columnas, para SHAP)")
