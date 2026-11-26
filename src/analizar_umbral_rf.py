"""Compara umbrales de decisión de Random Forest usando solo validación."""
from pathlib import Path
import argparse

import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from ids_ml.artifacts import load_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/processed'))
    ap.add_argument('--model', type=Path, default=Path('models/final'))
    ap.add_argument('--out', type=Path, default=Path('reports/umbrales_random_forest.csv'))
    ap.add_argument('--thresholds', nargs='*', type=float,
                    default=[0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    args = ap.parse_args()
    if not args.thresholds or not all(0 <= t <= 1 for t in args.thresholds):
        ap.error('Los umbrales deben estar entre 0 y 1')
    pipeline = load_pipeline(args.model)
    import hashlib
    for name in ['X_val.npz', 'y_val.csv']:
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != pipeline.metadata['prepared_input_sha256'][name]:
            raise ValueError(f'Validación incompatible con el modelo: {name}')
    model = pipeline.random_forest
    X = load_npz(args.data / 'X_val.npz').tocsr()
    y = pd.read_csv(args.data / 'y_val.csv')['label'].astype(int).to_numpy()
    pos_col = list(model.classes_).index(1)
    probability = model.predict_proba(X)[:, pos_col]
    rows = []
    for threshold in args.thresholds:
        # Use > so exact 0.5 ties match RandomForestClassifier.predict(),
        # which resolves equal class votes by the first class (0).
        pred = (probability > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            'umbral': threshold,
            'precision_ataque': precision_score(y, pred, zero_division=0),
            'recall_ataque': recall_score(y, pred, zero_division=0),
            'f1_ataque': f1_score(y, pred, zero_division=0),
            'verdaderos_negativos': int(tn),
            'falsas_alarmas': int(fp),
            'ataques_omitidos': int(fn),
            'ataques_detectados': int(tp),
        })
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print(f'\nGuardado en: {args.out.resolve()}')


if __name__ == '__main__':
    main()
