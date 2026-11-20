"""Métricas binarias y de cascada; nunca ajusta modelos."""
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score, precision_recall_curve, auc)

from ids_ml.contract import CLASSES


def evaluate_predictions(truth, labels, categories, probability):
    binary = truth["label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(binary, labels, labels=[0, 1]).ravel()
    both = len(np.unique(binary)) == 2
    precision, recall, _ = precision_recall_curve(binary, probability) if both else (None, None, None)
    per_attack_missed = {}
    for name in CLASSES[1:]:
        mask = truth["attack_cat"].to_numpy() == name
        per_attack_missed[name] = {
            "support": int(mask.sum()), "missed_as_normal": int((labels[mask] == 0).sum()),
            "fnr": float((labels[mask] == 0).mean()) if mask.any() else None}
    return {
        "rows": len(truth),
        "binary": classification_report(binary, labels, labels=[0, 1], output_dict=True, zero_division=0),
        "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "roc_auc": float(roc_auc_score(binary, probability)) if both else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None,
        "average_precision": float(average_precision_score(binary, probability)) if both else None,
        "cascade": classification_report(truth["attack_cat"], categories, labels=CLASSES,
                                         output_dict=True, zero_division=0),
        "binary_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "cascade_class_order": CLASSES,
        "cascade_confusion_matrix": confusion_matrix(truth["attack_cat"], categories, labels=CLASSES).tolist(),
        "attack_detection_by_class": per_attack_missed,
    }
