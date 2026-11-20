# Contrato de inferencia IDS, versión 1

El consumidor entrega **features de flujos ya calculadas**, una fila por observación
compatible con UNSW-NB15. El modelo no recibe paquetes, PCAP, direcciones IP,
puertos aislados ni vectores one-hot. `ids_ml.artifacts.load_pipeline()` carga el
artefacto una vez y `pipeline.predict(datos)` devuelve una lista de resultados.

La especificación verificable está en `models/final/feature_contract.json`;
se genera con las categorías observadas exclusivamente en entrenamiento y queda
encapsulada también dentro de `pipeline.joblib`. No hay que reconstruir encoders.

## Entrada

Se acepta un diccionario, una lista no vacía de diccionarios o un DataFrame.
Son obligatorias exactamente estas 42 columnas. El orden tabular se corrige por
nombre; se rechazan vectores anónimos, nombres duplicados y campos extra, incluidos
`id`, `label`, `attack_cat`. El orden canónico (base 0 en JSON) es el de esta tabla.

| Orden | Feature | Tipo | Interpretación orientativa |
|---:|---|---|---|
| 0 | dur | number | Duración del flujo |
| 1 | proto | string | Protocolo |
| 2 | service | string | Servicio; `-` es una categoría válida |
| 3 | state | string | Estado de conexión |
| 4 | spkts | integer | Paquetes de origen |
| 5 | dpkts | integer | Paquetes de destino |
| 6 | sbytes | integer | Bytes de origen |
| 7 | dbytes | integer | Bytes de destino |
| 8 | rate | number | Tasa del flujo |
| 9 | sttl | integer | TTL de origen |
| 10 | dttl | integer | TTL de destino |
| 11 | sload | number | Carga de origen |
| 12 | dload | number | Carga de destino |
| 13 | sloss | integer | Pérdidas de origen |
| 14 | dloss | integer | Pérdidas de destino |
| 15 | sinpkt | number | Tiempo entre paquetes de origen |
| 16 | dinpkt | number | Tiempo entre paquetes de destino |
| 17 | sjit | number | Jitter de origen |
| 18 | djit | number | Jitter de destino |
| 19 | swin | integer | Ventana TCP de origen |
| 20 | stcpb | integer | Secuencia TCP base de origen |
| 21 | dtcpb | integer | Secuencia TCP base de destino |
| 22 | dwin | integer | Ventana TCP de destino |
| 23 | tcprtt | number | Tiempo de ida/vuelta TCP |
| 24 | synack | number | Intervalo SYN/ACK |
| 25 | ackdat | number | Intervalo ACK/datos |
| 26 | smean | integer | Tamaño medio de paquete de origen |
| 27 | dmean | integer | Tamaño medio de paquete de destino |
| 28 | trans_depth | integer | Profundidad de transacciones |
| 29 | response_body_len | integer | Longitud del cuerpo de respuesta |
| 30 | ct_srv_src | integer | Conteo por servicio y origen |
| 31 | ct_state_ttl | integer | Conteo por estado y TTL |
| 32 | ct_dst_ltm | integer | Conteo reciente por destino |
| 33 | ct_src_dport_ltm | integer | Conteo por origen y puerto destino |
| 34 | ct_dst_sport_ltm | integer | Conteo por destino y puerto origen |
| 35 | ct_dst_src_ltm | integer | Conteo por par origen/destino |
| 36 | is_ftp_login | integer | Valor de login FTP del dataset; no convertir a booleano |
| 37 | ct_ftp_cmd | integer | Conteo de comandos FTP |
| 38 | ct_flw_http_mthd | integer | Conteo de métodos HTTP |
| 39 | ct_src_ltm | integer | Conteo reciente por origen |
| 40 | ct_srv_dst | integer | Conteo por servicio y destino |
| 41 | is_sm_ips_ports | integer | Indicador de igualdad IP/puerto; solo 0 o 1 |

Los valores numéricos deben ser finitos y no negativos. Los enteros pueden viajar
como `1.0`, pero no como `1.5`, texto o booleanos. No se imputan nulos ni se
recortan extremos. Los mínimos/máximos observados del JSON describen train;
no son límites máximos de aceptación. Los árboles no necesitan escalado.

Las categorías son sensibles a mayúsculas; no se convierten ni se recortan espacios.
Texto vacío, nulos y espacios exteriores se rechazan. Valores desconocidos de
`proto`, `service`, `state` se codifican como todos ceros en su bloque y se anuncian
en `unknown_categories`. No añadir categorías al encoder en producción.

## Transformación y salida

1. Validar y ordenar por nombre.
2. Aplicar el ColumnTransformer guardado: one-hot (`handle_unknown='ignore'`) para
   las tres categorías y passthrough para los 39 números. Resultado: 194 columnas.
3. RF estima el score de ataque. Regla exacta: `probability_attack > threshold`.
   Una igualdad con el umbral se clasifica como normal.
4. Solo las filas marcadas ataque pasan a XGBoost, que elige una de nueve categorías.
5. Retornar un diccionario por fila, conservando el orden de entrada.

```json
{
  "label": 1,
  "attack_cat": "Generic",
  "probability_attack": 0.91,
  "unknown_categories": []
}
```

Este ejemplo de formato es ilustrativo; las predicciones reales reproducibles
están en `models/final/example_expected.json`. `probability_attack` es el score RF,
no la confianza de la categoría XGB ni una probabilidad calibrada. `label=0`
siempre implica `attack_cat="Normal"`; `label=1` implica alguna categoría de ataque.

| Clase | Interpretación |
|---|---|
| Normal | Tráfico etiquetado como normal |
| Analysis | Ataques de análisis |
| Backdoor | Puertas traseras (nombre exacto del CSV en singular) |
| DoS | Denegación de servicio |
| Exploits | Explotación de vulnerabilidades |
| Fuzzers | Entradas anómalas para provocar fallos |
| Generic | Categoría genérica del dataset |
| Reconnaissance | Reconocimiento |
| Shellcode | Ejecución/inyección de shellcode |
| Worms | Gusanos |

## Consumo desde Python

```python
import json
from pathlib import Path
from ids_ml.artifacts import load_pipeline

pipeline = load_pipeline(Path("models/final"))
rows = json.loads(Path("models/final/example_input.json").read_text())
results = pipeline.predict(rows)
```

La ruta se resuelve respecto del proceso consumidor; el futuro backend debe
configurar una ruta absoluta. La carpeta requiere `pipeline.joblib`,
`metadata.json`, `feature_contract.json` y `artifact_manifest.json`. El loader
comprueba SHA-256 y versiones antes de deserializar. Los hashes detectan corrupción;
no autentican el origen. Cargar únicamente artefactos propios/confiables: joblib
usa pickle. No ejecutar `fit` ni scripts de entrenamiento desde el servicio.

`ValueError` indica contrato, integridad o versiones incompatibles; los archivos
faltantes producen `FileNotFoundError`. Todo el lote se rechaza si una fila es
inválida. El futuro adaptador API debe convertir estos errores a su respuesta
pública y limitar el tamaño de lote; ese adaptador todavía no existe.

## Límite de esta entrega

No hay extractor de tráfico implementado. Las descripciones de esta tabla no
sustituyen las fórmulas, unidades, dirección del flujo, timeout y ventanas
históricas del extractor original. **No rellenar con cero features imposibles de
obtener.** Antes de usar PCAP real, contrastar el extractor con la definición oficial
y construir casos de paridad feature por feature. Los CSV locales no incluyen
esa implementación ni identificadores temporales suficientes para verificarla.

La [fuente primaria UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
remite al archivo `UNSW-NB15_features.csv` y a la tesis del autor para los algoritmos
de extracción. El contrato numérico está definido y probado; la equivalencia de
un futuro extractor con esas definiciones sigue pendiente.
