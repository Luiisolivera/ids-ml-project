"""Evalúa artefactos guardados a través de la interfaz pública de inferencia.

El test oficial es un diagnóstico históricamente expuesto, no un holdout independiente.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids_ml.artifacts import load_pipeline
from ids_ml.contract import FEATURES
from ids_ml.metrics import evaluate_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=Path('models/final'))
    ap.add_argument('--data', type=Path, default=Path('data/processed'))
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    pipeline = load_pipeline(args.model)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    # Vincular la preparación al entrenamiento de este modelo.
    for name, digest in pipeline.metadata['prepared_input_sha256'].items():
        if manifest.get(name) != digest:
            raise ValueError(f'Preparación incompatible con modelo: {name}')
    path = args.data / f'raw_{args.split}.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[path.name]:
        raise ValueError(f'Datos alterados: {path}')
    frame = pd.read_csv(path)
    predictions = []
    for start in range(0, len(frame), 4096):
        predictions.extend(pipeline.predict(frame.iloc[start:start+4096][FEATURES]))
    labels = np.array([p['label'] for p in predictions])
    categories = np.array([p['attack_cat'] for p in predictions])
    scores = np.array([p['probability_attack'] for p in predictions])
    report = evaluate_predictions(frame, labels, categories, scores)
    report['split'] = args.split
    report['independent_holdout'] = False
    report['interpretation'] = ('Diagnóstico del test oficial históricamente expuesto; NO usar para tuning ni '
                                'presentar como evaluación independiente.' if args.split == 'test' else
                                'Validación usada para selección; estimación optimista de generalización.')
    report['pipeline_sha256'] = hashlib.sha256((args.model / 'pipeline.joblib').read_bytes()).hexdigest()
    out = args.out or Path('reports/final') / f'evaluation_{args.split}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['split', 'rows', 'counts', 'fpr', 'fnr', 'roc_auc', 'average_precision', 'interpretation']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
