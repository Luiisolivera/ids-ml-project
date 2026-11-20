"""Carga verificada de artefactos locales de confianza (joblib usa pickle)."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib

from ids_ml.pipeline import IDSPipeline


def load_pipeline(directory='models/final') -> IDSPipeline:
    directory = Path(directory)
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Versión de artefacto no soportada')
    for name in ['pipeline.joblib', 'metadata.json', 'feature_contract.json']:
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != manifest['sha256'][name]:
            raise ValueError(f'Integridad inválida: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    for name, expected in metadata['versions'].items():
        if version(name) != expected:
            raise ValueError(f'Versión incompatible de {name}: se requiere {expected}')
    pipeline = joblib.load(directory / 'pipeline.joblib')
    if not isinstance(pipeline, IDSPipeline):
        raise ValueError('Tipo de artefacto inesperado')
    if pipeline.contract != json.loads((directory / 'feature_contract.json').read_text()):
        raise ValueError('Contrato externo distinto al encapsulado')
    return pipeline
