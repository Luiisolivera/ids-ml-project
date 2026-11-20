# IDS-ML project

Prototipo de detección de intrusiones en dos etapas con UNSW-NB15:

1. Random Forest clasifica cada registro como tráfico normal o ataque.
2. Si Random Forest predice ataque, XGBoost predice su categoría.

## Estructura

```text
data/
  raw/          CSV originales de UNSW-NB15
  processed/    matrices transformadas, etiquetas y preprocesador
src/            scripts de preparación y entrenamiento
models/         modelos entrenados y codificador de categorías
reports/        métricas, matrices de confusión y predicciones de validación
docs/           documentos de referencia del TSP
```

## Preparar el entorno y ejecutar

Desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/preparar_datos_ids.py \
  --train "data/raw/UNSW_NB15_training-set.csv" \
  --test "data/raw/UNSW_NB15_testing-set.csv" \
  --out src/ids_preparado

python src/entrenar_ids.py \
  --data src/ids_preparado \
  --out src/ids_modelos \
  --trees 300 \
  --threshold 0.55

python src/analizar_umbral_rf.py \
  --data src/ids_preparado \
  --model src/ids_modelos/random_forest.joblib \
  --out src/ids_modelos/umbrales_rf_validacion.csv
```

En Windows PowerShell, activa el entorno con `.venv\\Scripts\\Activate.ps1`.

La preparación excluye `id` de las entradas y codifica `proto`, `service` y
`state`. Antes de dividir el conjunto oficial de entrenamiento, retira las filas
cuyas características aparecen en la prueba oficial. Luego crea entrenamiento y
validación agrupando las filas con características idénticas, para que un grupo
no cruce ambas particiones. Conserva todas las filas de prueba y ajusta el
preprocesador solo con entrenamiento. El detalle de duplicados, conflictos de
etiquetas y tamaños de partición queda en `src/ids_preparado/resumen.json`.

Los artefactos antiguos de `data/processed/`, `models/` y `reports/` se generaron
cuando los nombres de los archivos estaban intercambiados; usa los artefactos de
`src/ids_preparado/` y `src/ids_modelos/` para esta corrida.

El umbral 0.55 se aplica a la probabilidad de ataque de Random Forest antes de
enrutar los registros a XGBoost. El entrenamiento exporta las predicciones y las
falsas alarmas de validación para facilitar su análisis, y guarda el umbral en
`src/ids_modelos/configuracion_inferencia.json`. El script de umbrales permite
comparar otros candidatos con validación; no cambia el umbral del modelo por sí
solo.

## Estado actual

La preparación corregida está en `src/ids_preparado/`. Antes de interpretar los
resultados, revisa los grupos de características idénticas que tienen etiquetas
distintas; sus cantidades están en `src/ids_preparado/resumen.json`.

El test oficial ya se evaluó una vez con umbral 0.50 durante la auditoría. Como
ese resultado motivó el ajuste del umbral, usa un conjunto nuevo o tráfico
capturado aparte para una estimación final independiente. No ajustes el umbral
con el test oficial ni presentes una nueva puntuación en ese mismo test como
estimación independiente para el umbral 0.55.

Los CSV de entrenamiento y prueba son copias de los archivos proporcionados.
Los documentos PDF de `docs/` son referencias del informe, no datos de entrenamiento.
