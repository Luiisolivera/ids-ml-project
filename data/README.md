# Datos y reproducción

Se conservan localmente los dos CSV originales recibidos. No se modificaron.
La [página oficial UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
contiene el enlace de descarga y condiciones de uso. Obtener los archivos de la
partición train/test, sin intercambiar nombres, y colocarlos en `data/raw/`:

| Archivo | Filas | SHA-256 local |
|---|---:|---|
| UNSW_NB15_training-set.csv | 175341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| UNSW_NB15_testing-set.csv | 82332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

Estos hashes identifican los bytes locales; no son una certificación de origen.
Cambios de finales de línea pueden cambiar el hash sin cambiar las filas.

Los originales grandes y `data/processed/` se excluyen de Git, pero no se borran
localmente. Una clonación requiere obtener los CSV y ejecutar la preparación.
Los reportes, configuración y contratos pequeños sí se versionan. Los pesos finales
se generan entrenando o se distribuyen como la carpeta completa `models/final/`.
No existe descarga automática ni almacenamiento remoto de artefactos configurado.

`python src/preparar_datos_ids.py` crea matrices sparse, labels, preprocesador,
entradas crudas por partición, manifiesto de split, categorías y hashes. Estos CSV
preparados son salidas de evaluación reproducibles, no copias originales adicionales.
El solapamiento se purga solo del train oficial, conserva íntegro el test y no
consulta sus etiquetas para elegir las filas retiradas. Esta política consulta
las **features** del test para descontaminar: no es un holdout totalmente ciego.
No se elimina ninguna etiqueta contradictoria para mejorar artificialmente métricas.
