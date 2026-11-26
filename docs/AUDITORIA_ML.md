# Auditoría y cierre técnico de ML

## Alcance inspeccionado

Se leyeron los tres scripts originales, README, requirements, todas las carpetas
y reportes/configuraciones de preparación y entrenamiento. No había notebooks,
tests, AGENTS.md, .gitignore ni configuración de entrenamiento centralizada.
El repositorio Git no tenía commits: los archivos de trabajo estaban sin seguimiento.
Los tres PDF de docs/referencias son documentos académicos; se conservaron y no
participaron en entrenamiento. No se desarrolló backend ni dashboard.

## Flujo anterior reconstruido

CSV → retirar id/targets → purgar train coincidente con test → split de cinco folds
por grupos idénticos → one-hot/passthrough → matrices npz → RF binario + XGB sobre
ataques con pesos → validación y barrido auxiliar de umbrales → joblib separados.

Había tres ubicaciones de preparación y dos de modelos. Los reportes más antiguos
indicaban 82.332 filas en train oficial y 175.341 en test, nombres intercambiados.
Los artefactos corregidos de src/ids_preparado usaban los tamaños correctos y
separación por grupos. Sus nueve archivos de matrices/labels/preprocesador/features
coincidían byte a byte con data/processed_step2; el resumen difería en rutas.

## Hallazgos y decisiones

1. **Exposición histórica al test, no reparable con código.** README anterior admite
   que el test a 0,50 motivó cambiar el umbral. Se preserva literalmente en
   reports/historico/README_original.md. Esta entrega no usa test para selección;
   la estimación independiente sigue pendiente. No crear otro split y llamarlo
   nuevo holdout de material ya observado.
2. **Solapamiento exacto:** 10.279 filas de train compartían features con test.
   Se conservó la purga previa y el split por grupos. Test sigue completo. La purga
   consulta features del test pero no etiquetas, aspecto explícito en README.
3. **Ambigüedad en los datos:** train original tiene 229 grupos de features con
   label contradictorio (940 filas), y 1.772 con múltiples attack_cat (30.528 filas).
   Después de purgar quedan 131 grupos binarios conflictivos (429 filas) y 1.536
   multiclass (28.292 filas). Se mantienen; no hay evidencia para corregir targets.
4. **Calidad:** cero nulos/infinitos originales. Schema explícito, validación de
   categorías/tipos, no clipping ni imputación arbitraria. Se registran cuantiles
   numéricos de train y desbalance por partición. Test tiene cinco filas con state
   desconocido en train; el encoder no se reajusta y la inferencia lo informa.
5. **Artefactos fragmentados:** se reemplazan con IDSPipeline y un loader con checksums,
   versiones y contrato. El validador se comparte entre preparación e inferencia.
6. **Selección y regularización explícita en validación:** configuración previa fijaba .55.
   Se definió antes de evaluar test una búsqueda de regularización de RF (`min_samples_leaf`
   en `[1, 2, 5]`), 12 umbrales por F1 ataque en validación y comparación de cinco esquemas de
   pesos de clase en XGB (`none`, `power_0.25`, `power_0.5`, `power_0.75`, `balanced`) por
   macro F1 de cascada. El resultado selecciona `min_samples_leaf=2`, umbral `0.50` y pesos
   suavizados `power_0.75` (elevando el F1 macro de cascada en validación de `0.5947` a `0.6329`
   y el F1 ponderado de `0.8032` a `0.8178`). No se agregaron otros algoritmos.
7. **Métricas incompletas:** se agregaron ROC-AUC, AP, PR-AUC trapezoidal, FPR/FNR,
   errores de detección por clase y evaluación guardada de la interfaz pública.
   Se distinguen métricas condicionales XGB de la cascada real.
8. **Reproducibilidad:** se fijaron versiones, seed, jobs, configuración, parámetros,
   hashes de fuentes/datos, split y tiempos. Instalación editable del paquete
   evita depender de rutas de importación improvisadas.
9. **Fallos de ejecución:** una primera invocación de entrenamiento se lanzó antes
   de acabar la preparación y falló por manifest.json inexistente; no produjo
   modelo ni métricas. Se esperó la finalización correcta y se ejecutó la corrida
   final completa. No se modificaron datos o hiperparámetros por ese fallo.

## Resultado y límites

Ver resultados exactos en reports/final y tablas de README. Validación RF detecta
21.473 ataques, omite 687 y genera 850 falsas alarmas (7,83 %). El test diagnóstico detecta
44.560, omite 772 y genera 8.734 falsas alarmas (23,61 % del tráfico normal).
Los pesos suavizados (`power_0.75`) mejoraron sustancialmente `DoS` (F1 val `0.3631 → 0.5002`),
`Analysis` (`0.1496 → 0.2343`), `Backdoor` (`0.1077 → 0.1922`) y `Worms` (`0.5574 → 0.5926`),
pero `Analysis` y `Backdoor` siguen presentando baja calidad absoluta por ambigüedad de firmas.
No afirmar disponibilidad para producción ni ocultar estos errores detrás de accuracy global.

Los scores no están calibrados. El agrupamiento evita duplicados exactos entre
particiones, pero no excluye toda correlación temporal o por host. No hay datos
nuevos independientes disponibles en el repositorio, ni extractor PCAP de paridad
verificada. La fase ML es **operativa técnicamente y parcial metodológicamente**.

## Limpieza con evidencia

Se preservaron métricas, matrices, predicciones y resúmenes históricos antes de
retirar artefactos. El inventario original incluye tamaño y SHA-256. Se conservaron
los datasets originales y todos los PDF. Los modelos viejos de particiones
intercambiadas no son consumidos por el código final; el modelo corregido .55
queda reproducido por la nueva ejecución y documentado por sus métricas.

Se eliminaron data/processed_step2, src/ids_preparado, src/ids_modelos y los tres
joblib antiguos directamente bajo models. Los outputs viejos de reports se
reubicaron en reports/historico. data/processed se regeneró y centralizó.
Lista exacta, tamaño y motivo: reports/final/cleanup.json. Se retiraron 476.900.147
bytes de rutas obsoletas (antes de descontar el espacio de artefactos nuevos y del
archivo histórico). También se eliminaron caches Python del proyecto y copias de evidencia histórica
idénticas por SHA-256; cleanup.json identifica la copia conservada en cada caso.
El total retirado, incluyendo esa deduplicación y caches, fue 479.429.621 bytes.

La evidencia histórica no es output duplicado activo: registra experimentos previos
con particiones diferentes y sus limitaciones. No utilizar sus métricas como
resultado final. No se tocaron .git ni .venv.

## Verificaciones finales

Las 13 pruebas pasaron sin skips. La inferencia se comprobó también desde /tmp
con rutas absolutas. Los hashes de originales y código coinciden con la procedencia
del modelo. La evaluación recargada reproduce exactamente las matrices y métricas
de clasificación. AP difiere aproximadamente 2,3e-10 y PR-AUC 1,3e-14 entre lotes;
se verificó con tolerancia absoluta 1e-8, sin modificar scores ni etiquetas. La
acumulación numérica paralela de RF puede alterar el orden de scores casi iguales.
No se exige igualdad bit a bit a esas integrales. La evidencia está en
reports/final/verification.json.

## Condición para cerrar la fase ML global

Obtener datos independientes (sin overlap exacto y preferiblemente de otras sesiones/
tiempo), etiquetados y con extracción compatible. Congelar el modelo, umbral,
criterios de coste y métricas antes de acceder a esos resultados. Evaluar detección
y clases minoritarias, falsas alarmas, incertidumbre y shift. No se simula esa
validación con los datos actuales. Puede iniciarse el adaptador backend para
features preparadas con esta limitación declarada.
