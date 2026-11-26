# Project Overview

IDS offline con UNSW-NB15: RF detecta ataque y XGBoost clasifica su categoría.
El repositorio contiene ML; backend, captura, alertas y dashboard son futuros.
Leer README.md, docs/AUDITORIA_ML.md y docs/CONTRATO_INFERENCIA.md antes de editar.

# Current Status

Pipeline, entrenamiento final, artefactos, métricas, backend MVP (`backend/`), frontend MVP (`frontend/`) y 19 pruebas completados.
**Fase ML global parcial**: falta evaluación independiente. El test oficial fue
consultado históricamente para una decisión de umbral; no se puede recuperar su
independencia cambiando el split. El resultado actual del test es diagnóstico.
FPR diagnóstico 0.236054; no declarar el sistema listo para producción.
No hay extractor de tráfico en vivo ni despliegue de producción.

# Architecture

Actual: features preparadas → Dashboard (`frontend/`) / API REST (`backend/`) → validación →
one-hot/passthrough → RF → XGB si ataque → salida estructurada + registro SQLite de alertas → visualización web.
Todo el estado ajustado de ML vive en IDSPipeline.
Objetivo: tráfico → captura/flujos → extractor compatible → ML → API → alertas,
almacenamiento/eventos → dashboard.

# Repository Structure

- `backend/`: adaptador de inferencia (`adapter.py`), almacén SQLite de alertas (`alerts.py`) y servidor HTTP REST + estáticos (`server.py`).
- `frontend/`: dashboard web sin dependencias externas (`index.html`, `styles.css`, `app.js`) que consume la API REST local.
- `src/ids_ml/`: contrato, clase serializable, loader y métricas compartidas.
- `src/preparar_datos_ids.py`: split, preparación, auditoría y manifiestos.
- `src/entrenar_ids.py`: regularización RF y cinco variantes de pesos XGB; selección en validación.
- `src/evaluar_ids.py`, `src/predecir_ids.py`: consumo del modelo desde disco.
- `src/analizar_umbral_rf.py`: análisis de validación; no muta el modelo.
- `src/simular_trafico.py`: demo; reenvía flujos etiquetados del test a `/predict` (diagnóstico, no evaluación).
- `config/training.json`: semilla de modelos, recursos y criterios de selección.
- `data/raw/`: originales conservados e ignorados por Git; ver data/README.md.
- `data/processed/`: derivado, regenerable e ignorado por Git.
- `models/final/`: modelo de esta entrega. Pesos locales, excluidos de Git.
- `reports/final/`: resultados actuales, verificaciones y registro de limpieza.
- `reports/historico/`: evidencias originales; rutas de esos documentos son históricas.
- `docs/referencias/`: PDF del TSP; no se usan como datos de ML.
- `tests/`: contrato, artefactos y pruebas de integración del backend y frontend MVP.

# Machine Learning Pipeline

Original train 175341, test 82332. Se retiran del train 10279 filas cuyas features
están en test; el test no se altera. Se usan features del test para esta purga,
nunca targets. Primer fold de StratifiedGroupKFold de cinco folds con shuffle y
seed 42: train 132049, validación 33013. Se agrupan firmas de las 42 features sin
id ni targets. No hay solapamiento exacto entre particiones. Esto no garantiza
independencia temporal/de hosts y no elimina la exposición histórica del test.

Validador compartido, tres categorías one-hot aprendidas solo con train y 39
numéricas sin escalado. 194 columnas transformadas. No imputación ni clipping.
No remover conflictos de etiquetas para mejorar resultados. Pesos calculados solo
en train; XGB aprende únicamente ataques. No se hace refit con validación.

# Final Model

`models/final/pipeline.joblib`: RF 300 árboles, `min_samples_leaf=2`, balanced_subsample,
max_features=sqrt; XGB 300 rondas, profundidad 6, learning_rate .1, subsample/colsample .8,
lambda 1, hist y pesos suavizados `power_0.75`. Cuatro workers, seed 42. RF usa `score > 0.50`.

Regularización RF (`min_samples_leaf` en `[1, 2, 5]`) y umbral elegidos por F1 de ataque
en validación, desempate recall. Pesos XGB elegidos entre `none`, `power_0.25`, `power_0.5`,
`power_0.75` y `balanced` por F1 macro de cascada: `power_0.75` (.632895) frente a
`balanced` (.594703) y `none` (.590686). `power_0.75` supera tanto a `balanced` como a `none`
en F1 ponderado (.817769), aunque `power_0.25` logró mayor F1 ponderado (.824825); no ocultar
esa contrapartida. Configuración completa, hashes, versiones y tiempos están en metadata.json.
No seleccionar otro modelo usando el diagnóstico de test. No alegar superioridad sobre
algoritmos no evaluados.

# Model Artifacts

El loader requiere juntos `pipeline.joblib`, `metadata.json`, `feature_contract.json`
y `artifact_manifest.json`. El preprocesador, dos clasificadores, LabelEncoder,
umbral y contrato están encapsulados; no recomponerlos desde carpetas históricas.
Ejemplos reales: `example_input.json` y `example_expected.json`.
`load_pipeline` verifica hashes y versiones instaladas. Joblib usa pickle:
solo cargar artefactos propios/confiables; hashes no autentican un emisor ajeno.
La inferencia no necesita datos de entrenamiento ni data/processed.

# Feature Contract

Fuente canónica: `src/ids_ml/contract.py`, contrato generado en models/final y
especificación humana en docs/CONTRATO_INFERENCIA.md. Exactamente 42 nombres,
todos obligatorios; dict, list[dict] o DataFrame no vacío. Orden alineado por nombre.
Rechazar extras (incluso id/targets), nulos, texto numérico, booleanos numéricos,
infinitos, negativos y fracciones en enteros. `is_sm_ips_ports` acepta 0/1.
Categorías sensibles a mayúsculas; desconocidas → bloque one-hot cero y advertencia
en `unknown_categories`. No convertir `is_ftp_login` a booleano.
Salida list[dict]: label (0/1), attack_cat, probability_attack (score RF no calibrado),
unknown_categories. label=0 implica Normal. No confundir score con confianza XGB.

# Development Rules

- Justificar cambios del pipeline y reflejarlos en documentación/artefactos.
- No cambiar nombres, orden ni tipos sin actualizar entrenamiento e inferencia.
- Evitar leakage: fit, pesos, encoding y decisiones solo con train/validación.
- Nunca usar test para tuning. Para evaluación independiente obtener datos nuevos.
- No entrenar desde backend: cargar el artefacto ya ajustado una vez al iniciar.
- Evitar preprocessing duplicado y mantener separación ML/backend/dashboard.
- Ejecutar pruebas aplicables antes de declarar terminado un cambio.
- Actualizar README y AGENTS cuando cambien contrato, modelo o arquitectura.
- Conservar originales y evidencia necesaria; justificar borrados con dependencias.
- No inventar resultados ni ocultar clases minoritarias con malos resultados.
- No publicar/commitear automáticamente datos o pesos grandes. Documentar distribución.

# Commands

Desde la raíz, Python >=3.12 (entorno validado: 3.14.6):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pip check
python src/preparar_datos_ids.py
python src/entrenar_ids.py
python src/evaluar_ids.py --split val
python src/evaluar_ids.py --split test
python src/predecir_ids.py
python src/analizar_umbral_rf.py --out reports/final/thresholds_reload.csv
python -m backend.server --host 127.0.0.1 --port 8000 --model models/final
python -m unittest discover -s tests -v
python -m compileall -q src backend tests
```

Las pruebas de integración requieren preparación y entrenamiento previos. No
presentar skips o ausencia de artefactos como validación completa. Los scripts
resuelven paths CLI desde cwd; utilizar la raíz o pasar rutas explícitas.

# Backend Handoff

MVP implementado en `backend/`: `InferenceAdapter` carga una vez la carpeta final
con ruta absoluta, controla `max_batch_size`, delega íntegramente en `pipeline.predict`
(sin `fit`), mapea errores de schema/lote/readiness y `AlertStore` registra ataques
en SQLite. Probado en `tests/test_backend.py` con el ejemplo real, errores de schema y HTTP.

El extractor se implementará después y requiere unidades/dirección/ventanas/timeout
compatibles con UNSW-NB15; no está resuelta esa equivalencia. No rellenar campos
faltantes con ceros. Los contadores históricos ct_* necesitan contexto, no solo
un paquete. Obtener un conjunto independiente antes de afirmar calidad operativa.

# Do Not Do

No reusar models antiguos, mezclar preprocesadores, recalcular categorías en
producción, usar id como feature, reordenar vectores manualmente, ajustar al test,
cambiar automáticamente el umbral en inferencia o entrenar al cargar el servicio.
No marcar la fase ML global como completada mientras falte evaluación independiente.
