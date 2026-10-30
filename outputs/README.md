# outputs/

Salidas **pesadas y regenerables** de los scripts (matrices `.npz`, tablas `.parquet`, modelos completos).
No se versionan: se recrean ejecutando `pipeline/`.

- `limpio/`     -> tabla limpia, evidencia por fuente y anio, bitacora
- `preparado/`  -> matrices por escenario, preprocesadores, tablas de particion y composicion
- `modelos/`    -> modelos entrenados y tabla comparativa
