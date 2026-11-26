"""Entrena la cascada, selecciona con validación y guarda un único artefacto.

Nunca carga X_test, raw_test o y_test. No hace refit con validación.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ids_ml.contract import FEATURES
from ids_ml.metrics import evaluate_predictions
from ids_ml.pipeline import IDSPipeline


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def valid_xgb_weighting(scheme: str) -> bool:
    if scheme in {'balanced', 'none'}:
        return True
    if scheme.startswith('power_'):
        try:
            alpha = float(scheme.split('_', 1)[1])
            return 0.0 < alpha < 1.0
        except ValueError:
            return False
    return False


def compute_xgb_weights(weighting: str, y_attack: np.ndarray) -> np.ndarray | None:
    """Calcula pesos de muestra en train: 'none', 'balanced' o suavizados 'power_<alpha>'."""
    if weighting == 'none':
        return None
    base = compute_sample_weight('balanced', y_attack)
    if weighting == 'balanced':
        return base
    if weighting.startswith('power_'):
        alpha = float(weighting.split('_', 1)[1])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'Exponente fuera de (0, 1): {weighting}')
        weights = np.power(base, alpha)
        return weights / weights.mean()
    raise ValueError(f'Pesos XGB inválidos: {weighting}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/processed'))
    ap.add_argument('--out', type=Path, default=Path('models/final'))
    ap.add_argument('--reports', type=Path, default=Path('reports/final'))
    ap.add_argument('--config', type=Path, default=Path('config/training.json'))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['threshold_selection'] != 'max_attack_f1_then_recall_on_validation' or cfg['model_selection'] != 'max_cascade_macro_f1_on_validation':
        raise ValueError('Criterio de selección no soportado')
    if cfg['trees'] < 1 or not cfg['thresholds'] or not all(0 <= t <= 1 for t in cfg['thresholds']):
        raise ValueError('Configuración inválida')
    rf_leaves = cfg.get('rf_min_samples_leaf', [1])
    if not rf_leaves or not all(isinstance(v, int) and v >= 1 for v in rf_leaves):
        raise ValueError('Configuración rf_min_samples_leaf inválida')
    if not cfg['xgb_weighting'] or not all(valid_xgb_weighting(w) for w in cfg['xgb_weighting']):
        raise ValueError('Pesos XGB inválidos')
    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    inputs = ['X_train.npz', 'X_val.npz', 'y_train.csv', 'y_val.csv', 'preprocesador.joblib',
              'feature_contract.json', 'raw_val.csv', 'example_input.json', 'resumen.json']
    for name in inputs:
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f'Artefacto preparado alterado: {name}')
    X_train, X_val = [load_npz(args.data / f'X_{s}.npz').tocsr() for s in ['train', 'val']]
    y_train, y_val = [pd.read_csv(args.data / f'y_{s}.csv') for s in ['train', 'val']]
    preprocessor = joblib.load(args.data / 'preprocesador.joblib')
    contract = json.loads((args.data / 'feature_contract.json').read_text())

    rf_candidates = []
    rf, selected_threshold, selected_thresholds_table, selected_leaf = None, None, None, None
    best_rf_key = (-1.0, -1.0)
    rf_total_seconds = 0.0
    for min_leaf in rf_leaves:
        print(f'Entrenando Random Forest (min_samples_leaf={min_leaf})...', flush=True)
        phase = time.perf_counter()
        cand_rf = RandomForestClassifier(
            n_estimators=cfg['trees'], class_weight='balanced_subsample',
            min_samples_leaf=min_leaf, random_state=cfg['seed'],
            n_jobs=cfg['jobs'], max_features='sqrt')
        cand_rf.fit(X_train, y_train['label'])
        fit_sec = time.perf_counter() - phase
        rf_total_seconds += fit_sec
        probability = cand_rf.predict_proba(X_val)[:, list(cand_rf.classes_).index(1)]
        thresholds = []
        for threshold in cfg['thresholds']:
            pred = (probability > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val['label'], pred, labels=[0, 1]).ravel()
            thresholds.append({'threshold': threshold, 'f1_attack': f1_score(y_val['label'], pred),
                               'precision_attack': precision_score(y_val['label'], pred, zero_division=0),
                               'recall_attack': recall_score(y_val['label'], pred),
                               'fpr': float(fp / (fp + tn)), 'fnr': float(fn / (fn + tp)),
                               'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
        best_row = max(thresholds, key=lambda r: (r['f1_attack'], r['recall_attack']))
        rf_candidates.append({
            'min_samples_leaf': min_leaf, 'fit_seconds': fit_sec,
            'best_threshold': best_row['threshold'], 'best_threshold_metrics': best_row})
        print(f"RF min_samples_leaf={min_leaf}: {fit_sec:.1f}s, mejor umbral={best_row['threshold']} "
              f"(F1={best_row['f1_attack']:.6f}, FPR={best_row['fpr']:.6f})", flush=True)
        cand_key = (best_row['f1_attack'], best_row['recall_attack'])
        if cand_key > best_rf_key:
            best_rf_key = cand_key
            rf = cand_rf
            selected_threshold = best_row['threshold']
            selected_thresholds_table = thresholds
            selected_leaf = min_leaf
            rf_seconds = fit_sec

    pd.DataFrame(selected_thresholds_table).to_csv(args.reports / 'thresholds_validation.csv', index=False)
    print(f'RF seleccionado: min_samples_leaf={selected_leaf}, umbral={selected_threshold}', flush=True)

    attack_mask = y_train['label'].to_numpy() == 1
    encoder = LabelEncoder()
    y_attack = encoder.fit_transform(y_train.loc[attack_mask, 'attack_cat'])
    val_attack_mask = y_val['label'].to_numpy() == 1
    candidates, best, best_score = [], None, -1
    for weighting in cfg['xgb_weighting']:
        print(f'Entrenando XGBoost: pesos={weighting}...', flush=True)
        booster = XGBClassifier(objective='multi:softprob', num_class=len(encoder.classes_),
                                eval_metric='mlogloss', n_estimators=cfg['trees'], max_depth=6,
                                learning_rate=.1, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
                                tree_method='hist', n_jobs=cfg['jobs'], random_state=cfg['seed'], verbosity=0)
        weights = compute_xgb_weights(weighting, y_attack)
        phase = time.perf_counter()
        booster.fit(X_train[attack_mask], y_attack, sample_weight=weights)
        seconds = time.perf_counter() - phase
        pipeline = IDSPipeline(preprocessor, rf, booster, encoder, selected_threshold, contract, {})
        labels, categories, scores = pipeline.predict_transformed(X_val)
        metrics = evaluate_predictions(y_val, labels, categories, scores)
        conditional = encoder.inverse_transform(booster.predict(X_val[val_attack_mask]).astype(int))
        metrics['xgb_conditional_actual_attacks'] = classification_report(
            y_val.loc[val_attack_mask, 'attack_cat'], conditional, labels=encoder.classes_,
            output_dict=True, zero_division=0)
        candidate = {'weighting': weighting, 'fit_seconds': seconds, 'metrics': metrics}
        candidates.append(candidate)
        macro_f1 = metrics['cascade']['macro avg']['f1-score']
        weighted_f1 = metrics['cascade']['weighted avg']['f1-score']
        print(f'XGB {weighting}: {seconds:.1f}s, F1 macro cascada={macro_f1:.6f}, F1 ponderado={weighted_f1:.6f}', flush=True)
        if macro_f1 > best_score:
            best, best_score, winner = pipeline, macro_f1, candidate
    elapsed = time.perf_counter() - start
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'selected_rf_min_samples_leaf': selected_leaf,
        'selected_xgb_weighting': winner['weighting'], 'threshold': selected_threshold,
        'fit_rf_seconds': rf_seconds, 'fit_rf_candidates_total_seconds': rf_total_seconds,
        'training_selection_seconds': elapsed,
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib']},
        'prepared_input_sha256': {name: manifest[name] for name in inputs},
        'raw_sha256': json.loads((args.data / 'resumen.json').read_text())['raw_sha256'],
        'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path('src').rglob('*.py'))},
        'evaluation_status': 'Validation used for selection. Official test historically exposed; diagnostic only. Independent holdout pending.',
        'refit_on_validation': False,
        'random_forest_params': rf.get_params(),
        'xgboost_params': {**best.xgboost.get_params(), 'missing': 'NaN (library default)'},
    }
    best.metadata = metadata
    # Comprobar interfaz pública frente a matrices usadas al entrenar.
    raw_val = pd.read_csv(args.data / 'raw_val.csv')
    public = best.predict(raw_val[FEATURES])
    direct = best.predict_transformed(X_val)
    np.testing.assert_array_equal([p['label'] for p in public], direct[0])
    np.testing.assert_array_equal([p['attack_cat'] for p in public], direct[1])
    np.testing.assert_allclose([p['probability_attack'] for p in public], direct[2], rtol=0, atol=1e-12)
    model_path = args.out / 'pipeline.joblib'
    joblib.dump(best, model_path, compress=3)
    save_json(args.out / 'metadata.json', metadata)
    save_json(args.out / 'feature_contract.json', contract)
    save_json(args.out / 'artifact_manifest.json', {
        'schema_version': 1, 'sha256': {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                                    for name in ['pipeline.joblib', 'metadata.json', 'feature_contract.json']}})
    save_json(args.reports / 'model_selection.json', {
        'selection': cfg, 'rf_candidates': rf_candidates,
        'selected_rf_min_samples_leaf': selected_leaf,
        'selected_threshold': selected_threshold,
        'candidates': candidates, 'winner': winner['weighting']})
    save_json(args.reports / 'validation.json', winner['metrics'])
    example = json.loads((args.data / 'example_input.json').read_text())
    save_json(args.out / 'example_input.json', example)
    save_json(args.out / 'example_expected.json', best.predict(example))
    pd.DataFrame(public).drop(columns='unknown_categories').to_csv(args.reports / 'predictions_validation.csv', index=False)
    print(f'Artefacto: {model_path}; tiempo entrenamiento/selección: {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()

