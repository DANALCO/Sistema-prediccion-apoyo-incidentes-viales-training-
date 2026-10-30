"""
VERIFICACION DE LA PREPARACION (opcional, se ejecuta despues del script 02)
=============================================================================
Comprueba los invariantes que sostienen el criterio "Dataset" y la regla de
oro de la clase, y deja evidencia para el informe:

  - sin NaN/inf y con los nombres de columnas alineados
  - train balanceado; validacion y test con la distribucion real
  - train termina antes de que empiece test (orden temporal)
  - el preprocesador aprendio SOLO con train (sin fuga)
  - las filas sinteticas de SMOTENC son validas (one-hot, 0/1, horas enteras)
  - las posiciones guardadas se alinean con la tabla limpia
  - el preprocesador sirve para UNA fila suelta, como la del formulario de la app

Uso (desde la raiz del repositorio):   python pipeline/verificar_preparacion.py   (debe terminar en "TODO OK")
=============================================================================
"""
import os, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from accidentes import config as cfg, features
from accidentes.storage import leer_tabla

df = leer_tabla(os.path.join(cfg.DIR_LIMPIO, "us_accidents_limpio"))
ok = lambda c, m: print(("  OK   " if c else "  FALLA ") + m) or c
todo = []
for esc in cfg.ESCENARIOS:
    print(f"\n=== {esc} ===")
    z = np.load(os.path.join(cfg.DIR_PREPARADO, f"datos_{esc}.npz"))
    pre = joblib.load(os.path.join(cfg.DIR_PREPARADO, f"preprocesador_{esc}.joblib"))
    nombres = open(os.path.join(cfg.DIR_PREPARADO, f"features_{esc}.txt"), encoding="utf-8").read().split("\n")
    n_col = z["X_train_bal"].shape[1]
    todo.append(ok(len(nombres) == n_col == len(pre.nombres_features), f"nombres de features ({len(nombres)}) = columnas ({n_col})"))
    for k in z.files:
        if k.startswith("X_"):
            todo.append(ok(z[k].dtype == np.float32 and np.isfinite(z[k]).all(), f"{k}: float32, sin NaN/inf, forma {z[k].shape}"))
    # balance solo en train
    todo.append(ok(abs(z["y_train_bal"].mean() - 0.5) < 0.01, f"train balanceado: {z['y_train_bal'].mean():.3f} criticos"))
    for p in ("val_ajuste", "val_umbral", "test"):
        tasa = z[f"y_{p}"].mean()
        # alineacion con la tabla limpia
        alin = np.array_equal(df.iloc[z[f"pos_{p}"]][cfg.COL_OBJETIVO].to_numpy(), z[f"y_{p}"])
        todo.append(ok(alin and tasa < 0.2, f"{p}: real (criticos {tasa:.3f}), posiciones alineadas con la tabla limpia"))
    # orden temporal sin solapes
    f_tr = df.iloc[np.flatnonzero(df.Start_Time < cfg.CORTE_TRAIN)].Start_Time.max()
    f_te = df.iloc[z["pos_test"]].Start_Time.min()
    todo.append(ok(f_tr < pd.Timestamp(cfg.CORTE_TRAIN) <= f_te, f"train termina {f_tr.date()} < test empieza {f_te.date()}"))

    # --- FUGA: el preprocesador aprendio SOLO con train ---
    train = df[df.Start_Time < cfg.CORTE_TRAIN]
    med_train = train.groupby("State", observed=True)["Temperature(F)"].median()
    med_todo = df.groupby("State", observed=True)["Temperature(F)"].median()
    aprendida = pre.medianas_grupo_["Temperature(F)"]
    igual_train = all(np.isclose(aprendida[s], med_train[s]) for s in med_train.index)
    difiere_todo = any(not np.isclose(aprendida[s], med_todo[s]) for s in med_train.index)
    todo.append(ok(igual_train and difiere_todo, "medianas por estado = calculadas SOLO con train (y distintas de usar todo)"))

    # --- SMOTENC: categoricas/discretas validas en el train balanceado ---
    X = z["X_train_bal"]; nc, nd = len(pre.continuas), len(pre.discretas)
    off = nc + nd; bloques_ok = True
    for c in pre.categoricas:
        k = len(pre.categorias_[c]); suma = X[:, off:off+k].sum(axis=1)
        bloques_ok &= bool(np.allclose(suma, 1.0)); off += k
    todo.append(ok(bloques_ok, "cada variable categorica tiene exactamente un 1 (one-hot valido) tambien en las filas sinteticas"))
    bools = X[:, off:]; todo.append(ok(set(np.unique(bools)) <= {0.0, 1.0}, "booleanas solo 0/1 (SMOTE puro generaba 0.4)"))
    disc = pre.escalador_discretas_.inverse_transform(X[:, nc:nc+nd].astype("float64"))
    todo.append(ok(np.allclose(disc, np.rint(disc), atol=1e-3) and disc[:,0].min()>=-1e-3 and disc[:,0].max()<=23+1e-3, "Hora/Dia/Mes sinteticos son enteros validos (Hora en 0-23)"))

    # --- reproducibilidad: transformar() == matriz guardada ---
    sub = df.iloc[z["pos_test"][:200]]
    todo.append(ok(np.array_equal(pre.transformar(sub), z["X_test"][:200]), "preprocesador.transformar() reproduce exactamente X_test"))

    # --- fila suelta como la enviaria el formulario de la app ---
    fila = pd.DataFrame([{
        "Start_Time": pd.Timestamp("2023-02-14 08:30:00"), "State": "VA", "Start_Lat": 37.5, "Start_Lng": -77.4,
        "Temperature(F)": 41.0, "Humidity(%)": 80.0, "Pressure(in)": 30.1, "Visibility(mi)": 3.0,
        "Wind_Speed(mph)": 9.0, "Precipitation(in)": 0.02, "Wind_Direction": "NE", "Weather_Condition": "Lluvia",
        "Sunrise_Sunset": "Day", "Tipo_via": "Interestatal", "Distance(mi)": 0.4, **{c: (c in ("Traffic_Signal", "Crossing")) for c in cfg.INFRAESTRUCTURA}}])
    fila = features.agregar_variables_temporales(fila)
    xv = pre.transformar(fila)
    todo.append(ok(xv.shape == (1, n_col) and np.isfinite(xv).all(), f"fila del formulario -> matriz {xv.shape}"))
    # campos opcionales vacios + estado nunca visto
    vacia = fila.copy(); vacia[["Temperature(F)", "Humidity(%)", "Wind_Direction", "Weather_Condition", "Tipo_via"]] = np.nan; vacia["State"] = "ZZ"
    xs = pre.transformar(vacia); i0 = nc + nd; k = len(pre.categorias_["State"])
    todo.append(ok(np.isfinite(xs).all() and xs[0, i0:i0+k].sum() == 0, "campos vacios se imputan; estado desconocido no rompe (queda en ceros)"))
    # Tipo_via vacio -> categoria 'Desconocido' (igual que el clima); uno dado -> su propia columna
    if "Tipo_via" in pre.categoricas:
        nombres = pre.nombres_features
        col_desc, col_int = nombres.index("Tipo_via_Desconocido"), nombres.index("Tipo_via_Interestatal")
        todo.append(ok(xs[0, col_desc] == 1.0 and xv[0, col_int] == 1.0 and xv[0, col_desc] == 0.0,
                       "Tipo_via vacio -> 'Desconocido'; Tipo_via dado -> su categoria"))
    # campo obligatorio vacio -> error explicito
    mala = fila.copy(); mala["Start_Lat"] = np.nan
    try: pre.transformar(mala); todo.append(ok(False, "latitud vacia debia dar error"))
    except ValueError: todo.append(ok(True, "latitud vacia -> ValueError claro (la app debe exigirla)"))

print("\nRESULTADO:", "TODO OK" if all(todo) else f"{todo.count(False)} FALLAS")
