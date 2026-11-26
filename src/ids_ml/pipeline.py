"""Artefacto único: contrato + preprocesador + RF + XGB + clases + umbral."""
from dataclasses import dataclass

import numpy as np

from ids_ml.contract import CATEGORICAL, validate_features


@dataclass
class IDSPipeline:
    preprocessor: object
    random_forest: object
    xgboost: object
    label_encoder: object
    threshold: float
    contract: dict
    metadata: dict

    def predict_transformed(self, matrix):
        probability = self.random_forest.predict_proba(matrix)[:, list(self.random_forest.classes_).index(1)]
        labels = (probability > self.threshold).astype(int)
        categories = np.full(matrix.shape[0], "Normal", dtype=object)
        routed = labels == 1
        if routed.any():
            categories[routed] = self.label_encoder.inverse_transform(
                self.xgboost.predict(matrix[routed]).astype(int))
        return labels, categories, probability

    def predict(self, data) -> list[dict]:
        frame = validate_features(data)
        labels, categories, probability = self.predict_transformed(self.preprocessor.transform(frame))
        known = self.preprocessor.named_transformers_["categorical"].categories_
        unknown = {name: ~frame[name].isin(values).to_numpy()
                   for name, values in zip(CATEGORICAL, known)}
        return [{"label": int(labels[i]), "attack_cat": str(categories[i]),
                 "probability_attack": float(probability[i]),
                 "unknown_categories": [name for name in CATEGORICAL if unknown[name][i]]}
                for i in range(len(frame))]
