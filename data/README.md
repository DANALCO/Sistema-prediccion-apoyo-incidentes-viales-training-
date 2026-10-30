# Datos

Esta carpeta **no se versiona** (el CSV pesa unos 3 GB y GitHub rechaza archivos de mas de 100 MB).

## Como obtener el dataset

1. Descargue **US Accidents (2016 - 2023)**, de Sobhan Moosavi, desde Kaggle:
   <https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>
2. Copie el archivo `US_Accidents_March23.csv` en esta carpeta (`data/`).

Si prefiere dejarlo en otro lugar, defina la variable de entorno `ACCIDENTES_CSV`:

```
# Windows (PowerShell)
$env:ACCIDENTES_CSV = "C:/Users/usuario/Downloads/archive/US_Accidents_March23.csv"
# Linux / macOS
export ACCIDENTES_CSV=/ruta/al/US_Accidents_March23.csv
```

## Referencia

*US Accidents (2016-2023): A Countrywide Traffic Accident Dataset*, de Sobhan Moosavi et al.
Es la referencia [23] de `docs/entregable_1_propuesta_y_estado_del_arte.md`; copie de ahi la cita completa.
