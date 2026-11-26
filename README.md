# IDS basado en Machine Learning — UNSW-NB15

Prototipo offline de detección de intrusiones en dos etapas: Random Forest detecta
normal/ataque y XGBoost asigna categoría a las filas detectadas como ataque.
El objetivo es entregar un modelo reproducible y una interfaz de features para
una futura integración de tráfico, alertas y visualización.

## Estado actual

- **FASE ML: parcial.** Implementación, entrenamiento, serialización, evaluación e
  inferencia offline completados y probados. Falta una evaluación independiente
  sobre datos nuevos: el test oficial ya había influido en decisiones históricas.
- **BACKEND Y DASHBOARD: MVP implementados (`backend/`, `frontend/`).** Adaptador de inferencia
  con carga única del artefacto, servidor HTTP REST (`/health`, `/ready`, `/contract`,
  `/metrics`, `/predict`, `/alerts`), persistencia ligera SQLite para alertas detectadas
  y consola web interactiva servida en `/`.
- La entrega permite integrar features ya preparadas, pero no acredita un IDS
  apto para producción. El diagnóstico oficial muestra **23,61 % de falsas alarmas**.

## Arquitectura

Implementado: `features de flujo → Dashboard (frontend/) / API REST (backend/) → validación de contrato → ColumnTransformer → RF → (si ataque) XGBoost → predicción estructurada + registro de alerta SQLite → visualización en consola web`.

Arquitectura general:

```text
Network Traffic
  ↓ Packet / Flow Capture             [pendiente]
  ↓ Feature Extraction                [pendiente, requiere paridad UNSW-NB15]
  ↓ ML Preprocessing                  [implementado]
  ↓ ML Model / Prediction             [implementado]
  ↓ Backend/API + Alert Store (MVP)   [implementado en backend/]
  ↓ Dashboard Web (MVP)               [implementado en frontend/]
```

## Estructura del proyecto

```text
config/training.json             Semilla, recursos, candidatos y criterios
backend/
  adapter.py                     Adaptador de inferencia (carga única, lotes, readiness)
  alerts.py                      Almacenamiento ligero SQLite de alertas detectadas
  server.py                      Servidor HTTP REST + servidor estático del Dashboard
frontend/
  index.html                     Dashboard operativo para TI (estado, alertas, análisis de flujos)
  styles.css                     Estilos del panel de operaciones (sin dependencias externas)
  app.js                         Cliente JS que consume exclusivamente la API REST local
src/
  preparar_datos_ids.py           Auditoría, split por grupos y preprocessing
  entrenar_ids.py                 Entrenamiento y selección solo con validación
  evaluar_ids.py                  Evaluación desde artefacto, sin entrenamiento
  predecir_ids.py                 CLI JSON de inferencia offline
  analizar_umbral_rf.py           Análisis auxiliar sobre validación
  ids_ml/                        Contrato, pipeline, métricas y carga verificada
data/raw/                        Originales locales, excluidos de Git
data/processed/                  Salidas reproducibles, excluidas de Git
models/final/                    Artefacto único + contrato + metadatos + ejemplos
reports/final/                   Auditoría, candidatos, métricas y validaciones
reports/historico/               Evidencias anteriores, NO métricas finales
docs/CONTRATO_INFERENCIA.md       Entrada, salida y handoff
docs/AUDITORIA_ML.md             Hallazgos y limitaciones metodológicas
docs/referencias/                PDF académicos originales conservados
tests/                          Pruebas de contrato, artefactos, backend y frontend MVP
requirements.txt                Dependencias directas fijadas
requirements-lock.txt           Snapshot de dependencias transitivas
pyproject.toml                  Instalación del paquete ids_ml
AGENTS.md                       Contexto y reglas para futuros agentes
```

## Dataset

Se usan los dos CSV proporcionados de UNSW-NB15: 175.341 filas oficiales de train y
82.332 de test; 45 columnas, de las que se excluyen `id`, `label`, `attack_cat`.
Quedan 42 features (39 numéricas y `proto`, `service`, `state` categóricas).
La [fuente primaria de UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
describe la partición y enlaza su descarga. [data/README.md](data/README.md) detalla
ubicaciones y SHA-256 de las copias locales.

Objetivos: `label=0` normal, `label=1` ataque; `attack_cat` tiene Normal y nueve
tipos de ataque. No hay nulos ni infinitos en los originales. Hay 74.301 filas de
train con features repetidas y 229 grupos con etiquetas binarias contradictorias.
No se eliminaron contradicciones ni clases difíciles para favorecer resultados.

Se retiraron del train 10.279 filas cuyas features también están en test; no se
usaron sus etiquetas para esa operación. El test permanece íntegro. El train
restante se divide mediante el primer fold de StratifiedGroupKFold, cinco folds,
shuffle y semilla 42, agrupando features idénticas: 132.049 train / 33.013 validación.
La firma excluye id y targets. No hay firmas compartidas entre las tres particiones.
La deduplicación utiliza features del test y debe describirse como descontaminación,
no como aislamiento ciego absoluto.

El balanceo se realiza solo mediante pesos de entrenamiento. Worms tiene 103 filas
en train y 26 en validación: sus métricas son especialmente inestables. Distribuciones,
conflictos, tipos, cuantiles y categorías desconocidas: [auditoría real](reports/final/data_audit.json).

## Pipeline de Machine Learning

```text
CSV → validación de schema → firmas y purga de solapamiento → split por grupos
    → fit del one-hot solo en train → transform de validación/test
    → entrenamiento RF + XGB solo en train
    → selección de regularización RF, umbral y pesos XGB con validación
    → serialización del pipeline → recarga → evaluación/predicción
```

One-hot desconoce categorías futuras (`handle_unknown=ignore`), numéricas pasan
sin escalado. No hay imputación, selección estadística de features, SMOTE ni recorte
de outliers. Los extremos pueden contener ataques; se describen con cuantiles de
train. Se conservan 194 columnas transformadas. Las operaciones de inferencia usan
el mismo validador y preprocesador guardado que la preparación.

No se hace refit sobre validación. No se usa test para entrenar, seleccionar pesos,
ajustar el umbral ni escoger hiperparámetros en esta entrega. Sus resultados se
obtuvieron después de congelar el artefacto y no se usaron para modificarlo.

## Modelo final y selección

RandomForestClassifier: 300 árboles, `max_features=sqrt`, `min_samples_leaf=2`
(seleccionado en validación entre `[1, 2, 5]`), `class_weight=balanced_subsample`,
semilla 42 y cuatro workers. Umbral **0,50** con comparación estricta `>`; elegido
entre 12 candidatos por máximo F1 de ataque en validación, desempate por recall.
La regularización `min_samples_leaf=2` elevó en validación el F1 de ataque (`0.965234 → 0.965447`),
ROC-AUC (`0.992343 → 0.992497`) y Average Precision (`0.995997 → 0.996281`), reduciendo
además el FPR en validación (`0.080807 → 0.078319`). No representa un óptimo de coste
operativo: esos costes aún no se han definido.

XGBClassifier: 300 rondas de boosting (`n_estimators=300`, multiclass
con nueve clases), `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_lambda=1`, `tree_method=hist`,
`objective=multi:softprob`, cuatro workers, semilla 42. Solo aprende ataques de train.

Se compararon en validación cinco esquemas de pesos de clase ($w_c \propto (1/N_c)^\alpha$,
normalizados a media 1):

| Pesos XGB | F1 macro cascada validación | F1 ponderado cascada |
|---|---:|---:|
| `power_0.75` | **0.632895** | 0.817769 |
| `power_0.5` | 0.632340 | 0.822549 |
| `power_0.25` | 0.622521 | **0.824825** |
| `balanced` ($\alpha=1$) | 0.594703 | 0.803389 |
| `none` ($\alpha=0$) | 0.590686 | 0.811896 |

Se seleccionó `power_0.75` por máximo F1 macro en validación (`0.632895`, frente al
anterior `balanced` de `0.594651`), superando además tanto a `balanced` como a `none`
en F1 ponderado (`0.817769`). Los esquemas suavizados `power_0.5` y `power_0.25`
obtuvieron un F1 ponderado aún mayor (`0.822549` y `0.824825`); esa contrapartida se
conserva en el reporte. No se afirma que esta cascada supere algoritmos no evaluados.

Duración real de entrenamiento y selección completa (3 candidatos RF + 5 candidatos XGB): **330.5 s** (RF ganador 69.9 s).

## Métricas finales reales

**Validación participó en selección. El test oficial está históricamente expuesto**:
el README original reconoce que una evaluación a umbral 0,50 motivó un ajuste previo.
Un nuevo split del mismo material no recuperaría independencia. Se necesita un
conjunto nuevo, con criterios congelados antes de observar sus etiquetas.

| Métrica | Validación (selección) | Test oficial (diagnóstico, NO independiente) |
|---|---:|---:|
| Precision ataque | 0.961923 | 0.836117 |
| Recall ataque | 0.968998 | 0.982970 |
| F1 ataque | 0.965447 | 0.903616 |
| FPR (normal → ataque) | 0.078319 | 0.236054 |
| FNR (ataque → normal) | 0.031002 | 0.017030 |
| ROC-AUC binaria | 0.992497 | 0.983830 |
| PR-AUC trapezoidal | 0.996281 | 0.987888 |
| Average precision | 0.996281 | 0.987888 |
| F1 macro cascada | 0.632895 | 0.557069 |
| F1 ponderado cascada | 0.817769 | 0.799932 |
| Accuracy cascada (secundaria) | 0.810257 | 0.773563 |

Validación: TN=10003, FP=850, FN=687, TP=21473.
Test diagnóstico: TN=28266, FP=8734, FN=772, TP=44560.

Las matrices usan filas reales y columnas predichas. La binaria sigue `[0,1]`;
la multiclass declara explícitamente `cascade_class_order`. PR-AUC trapezoidal y
average precision se registran separadas, pues no son la misma definición.

Métricas por clase de la **cascada completa**, que incluyen los ataques detenidos
incorrectamente por RF como Normal:

| Clase | Soporte val | Precision val | Recall val | F1 val | F1 test diagnóstico |
|---|---:|---:|---:|---:|---:|
| Normal | 10853 | 0.9357 | 0.9217 | 0.9287 | 0.8561 |
| Analysis | 364 | 0.4427 | 0.1593 | 0.2343 | 0.0567 |
| Backdoor | 292 | 0.2897 | 0.1438 | 0.1922 | 0.1568 |
| DoS | 2373 | 0.3565 | 0.8382 | 0.5002 | 0.4900 |
| Exploits | 6523 | 0.8677 | 0.5511 | 0.6741 | 0.6980 |
| Fuzzers | 3499 | 0.7576 | 0.7262 | 0.7416 | 0.4180 |
| Generic | 7245 | 0.9980 | 0.9779 | 0.9879 | 0.9840 |
| Reconnaissance | 1616 | 0.8270 | 0.7512 | 0.7873 | 0.8579 |
| Shellcode | 222 | 0.5493 | 0.9279 | 0.6901 | 0.4016 |
| Worms | 26 | 0.5714 | 0.6154 | 0.5926 | 0.6517 |

No confundir el reporte XGB condicionado a ataques reales con la calidad extremo a
extremo. Los JSON contienen ambos y los ataques omitidos como Normal por cada clase.
Fuentes ejecutadas: [selección](reports/final/model_selection.json),
[validación recargada](reports/final/evaluation_val.json),
[test diagnóstico](reports/final/evaluation_test.json),
[umbrales](reports/final/thresholds_validation.csv).

## Instalación

Ejecutar desde la raíz. Se verificó en Linux con Python 3.14.6. Las dependencias
fijadas requieren Python >=3.12; otras versiones/plataformas no fueron probadas.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pip check
```

No hace falta GPU. El paquete Linux de XGBoost declara NCCL como dependencia
transitiva aunque esta configuración entrene en CPU. El lock refleja el entorno
real; no introduce dependencias de API, dashboard o notebooks. La instalación
editable y `pip check` se ejecutaron correctamente en el entorno existente;
no se creó un segundo entorno vacío para comprobar descargas desde cero.

## Cómo entrenar

Obtener los CSV según `data/README.md`. Los comandos sobrescriben salidas derivadas;
conservar otra carpeta usando `--out`/`--reports` si se quiere archivar una corrida.

```bash
python src/preparar_datos_ids.py
python src/entrenar_ids.py
```

Configuración en `config/training.json`. La preparación fija seed 42 para el split;
`seed` de la configuración controla los modelos. Los hashes del input, el código,
las versiones y parámetros efectivos quedan en `models/final/metadata.json`.
El entrenamiento verifica integridad de sus entradas preparadas antes de usarlas.

## Cómo evaluar y verificar

```bash
python src/evaluar_ids.py --split val
python src/evaluar_ids.py --split test
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

`--split test` solo produce diagnóstico del conjunto expuesto; no autoriza tuning.
El registro [verification.json](reports/final/verification.json) resume las comprobaciones.
Las 19 pruebas ejecutadas validan schema, umbral exacto, routing, pesos suavizados,
grupos separados, categorías aprendidas solo de train, integridad, salida, recarga
en otro proceso con métodos `fit` bloqueados, el servidor/adaptador del backend MVP
y la entrega del frontend/métricas.
Requieren preparación y artefactos finales existentes.

Análisis auxiliar (no modifica el umbral del artefacto):

```bash
python src/analizar_umbral_rf.py --out reports/final/thresholds_reload.csv
```

## Cómo ejecutar una predicción, el Backend MVP y el Dashboard Frontend

Inferencia CLI offline:

```bash
python src/predecir_ids.py
python src/predecir_ids.py --input models/final/example_input.json --out reports/final/smoke_predictions.json
```

Servidor HTTP REST + Dashboard Frontend (`http://127.0.0.1:8000/`):

```bash
python -m backend.server --host 127.0.0.1 --port 8000 --model models/final --max-batch-size 256 --alerts-db :memory:
```

Rutas del Dashboard y Endpoints disponibles:
- `GET /` (`frontend/index.html`, `/styles.css`, `/app.js`): Dashboard operativo orientado al personal de TI: estado del sistema, resumen y tabla de alertas (auto-actualización cada 5 s, filtro por tipo) y pestaña para analizar flujos en JSON (pegados o desde archivo). Las métricas de auditoría ML y el contrato de features no se muestran en la interfaz; siguen disponibles en `GET /metrics`, `GET /contract`, `reports/final/` y `docs/`.
- `GET /health`: estado del proceso (`200 OK`).
- `GET /ready`: verificación de disponibilidad e integridad del artefacto (`200 OK` / `503 Service Unavailable`).
- `GET /contract`: especificación JSON de las 42 features (`200 OK`).
- `GET /metrics`: métricas auditadas de validación, test diagnóstico, selección de pesos XGB, barrido de umbrales RF y ejemplos reales (`200 OK`).
- `POST /predict`: recibe un objeto de 42 features, una lista de objetos o `{"rows": [...]}`, delega en `IDSPipeline.predict` y registra alertas cuando `label == 1` (`200 OK` / `400` / `413` / `422` / `503`).
- `GET /alerts?limit=50&attack_cat=Reconnaissance`: consulta las alertas detectadas y el resumen por categoría.

Simulación de tráfico para demostraciones (con el servidor ya en ejecución, en otra terminal):

```bash
python src/simular_trafico.py --per-class 3 --delay 1.5
```

Envía flujos reales y etiquetados del test oficial (sin `id`, `label` ni `attack_cat`) uno a uno a
`POST /predict`, imprime etiqueta real frente a predicción (incluidos los errores) y las alertas
aparecen en el dashboard. Es una demostración: el test fue expuesto históricamente y la muestra es
pequeña; no sustituye una evaluación independiente.

Uso directo desde Python:

```python
from ids_ml.artifacts import load_pipeline
pipeline = load_pipeline("models/final")
predictions = pipeline.predict(features)  # dict, lista de dicts o DataFrame
```

## Artefactos generados y contrato

- `models/final/pipeline.joblib`: objeto único comprimido con validador (código del
  paquete), preprocesador ajustado, RF, XGB, codificador de clases y umbral.
- `metadata.json`: configuración, selección, tiempos, versiones y procedencia.
- `feature_contract.json`: 42 nombres, tipos, orden, categorías aprendidas y output.
- `artifact_manifest.json`: SHA-256 de los tres anteriores; requerido por el loader.
- `example_input.json` / `example_expected.json`: entrada válida y predicciones
  reales antes de serializar, cotejadas al recargar sin entrenar.
- `reports/final/`: evidencia auditable de preparación, selección, evaluación y pruebas.

Los pesos se conservan localmente, fuera de Git por su tamaño. Para distribuir el
modelo, enviar la carpeta completa junto a esta versión del paquete o regenerar
con los comandos anteriores. Los hashes no autentican modelos de fuentes ajenas.

El backend debe entregar exactamente 42 features crudas, sin targets/id ni one-hot,
y cargar una vez el artefacto de confianza. Rechaza nulos, campos extra y números
no finitos. Las categorías nuevas se informan en salida. Ver
[contrato completo](docs/CONTRATO_INFERENCIA.md). La inferencia no depende de los
CSV de entrenamiento ni de `data/processed/`.

## Limitaciones conocidas

- Falta evaluación independiente; exposición histórica al test no reversible.
- Falsas alarmas elevadas en test diagnóstico; no hay aceptación operativa definida.
- F1 bajo en Analysis, Backdoor y DoS. Los pesos mejoran macro F1, pero introducen
  falsas asignaciones a clases raras. No ocultar este resultado.
- Features idénticas con etiquetas contradictorias limitan la separabilidad.
- El split agrupa duplicados, pero no garantiza independencia temporal, de host o
  sesión: esos identificadores no están disponibles en estos CSV preparados.
- No hay calibración de scores ni intervalos de confianza, evaluación temporal,
  medición de drift o benchmark de latencia/concurrencia de producción.
- El extractor de tráfico real no existe. No se ha demostrado equivalencia de
  unidades, ventanas `ct_*`, dirección o timeouts con la extracción UNSW-NB15.
- Joblib exige compatibilidad de versiones y artefactos de confianza.

## Próximas etapas y handoff

1. Backend: crear primero un adaptador de inferencia que cargue la carpeta final
   al iniciar y valide solicitudes de features mediante `pipeline.predict`.
   Su primera prueba de integración debe usar `example_input.json`, comprobar
   `example_expected.json` y garantizar que nunca se llama `fit`.
2. Captura/procesamiento: implementar extractor con pruebas de paridad y definiciones
   originales; no sustituir features ausentes por ceros.
3. Integrar modelo y obtener datos nuevos etiquetados para evaluación independiente
   con modelo/umbral congelados antes de observar resultados.
4. Sistema de alertas y almacenamiento, criterios de coste/aceptación y eventos realtime.
5. Dashboard.
6. Testing end-to-end, rendimiento y monitorización.
7. Deployment tras validar generalización y límites operativos.

Estas etapas no se implementaron en esta entrega.
