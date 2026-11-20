"""MVP de Backend para el IDS basado en Machine Learning (UNSW-NB15).

Carga una única vez el artefacto congelado de models/final, delega íntegramente
la validación y predicción en IDSPipeline.predict sin entrenar ni duplicar
preprocesamiento, y registra alertas detectadas.
"""

from backend.adapter import (
    BatchSizeError,
    ContractValidationError,
    InferenceAdapter,
    ServiceNotReadyError,
)
from backend.alerts import AlertStore
from backend.server import IDSBackendApp, create_http_server

__all__ = [
    "AlertStore",
    "BatchSizeError",
    "ContractValidationError",
    "IDSBackendApp",
    "InferenceAdapter",
    "ServiceNotReadyError",
    "create_http_server",
]
