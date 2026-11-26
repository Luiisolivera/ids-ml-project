"""Integración real: requiere ejecutar preparación y entrenamiento primero."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from ids_ml.artifacts import load_pipeline
from ids_ml.contract import FEATURES, CATEGORICAL, CLASSES
from src.preparar_datos_ids import feature_signatures


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_pipeline()
        cls.example = json.loads(Path('models/final/example_input.json').read_text())

    def test_reload_in_fresh_process_without_fit(self):
        code = '''
import json
from pathlib import Path
from unittest.mock import patch
from ids_ml.artifacts import load_pipeline
with patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('fit forbidden')), patch('xgboost.XGBClassifier.fit', side_effect=AssertionError('fit forbidden')):
    print(json.dumps(load_pipeline().predict(json.loads(Path('models/final/example_input.json').read_text()))))
'''
        output = subprocess.check_output([sys.executable, '-c', code], text=True)
        expected = json.loads(Path('models/final/example_expected.json').read_text())
        actual = json.loads(output)
        for a, e in zip(actual, expected):
            self.assertEqual(a['label'], e['label'])
            self.assertEqual(a['attack_cat'], e['attack_cat'])
            self.assertAlmostEqual(a['probability_attack'], e['probability_attack'], places=12)
        self.assertEqual(len(actual), len(expected))

    def test_output_and_no_input_mutation(self):
        frame = pd.DataFrame(self.example)
        original = frame.copy(deep=True)
        result = self.model.predict(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(len(result), len(frame))
        for r in result:
            self.assertEqual(set(r), {'label', 'attack_cat', 'probability_attack', 'unknown_categories'})
            self.assertIn(r['attack_cat'], CLASSES)
            self.assertEqual(r['label'] == 0, r['attack_cat'] == 'Normal')
            self.assertTrue(0 <= r['probability_attack'] <= 1)

    def test_unknown_category(self):
        example = {**self.example[0], 'proto': '__unseen_protocol__'}
        result = self.model.predict(example)
        self.assertEqual(result[0]['unknown_categories'], ['proto'])

    def test_same_preprocessing_and_no_fit(self):
        frame = pd.read_csv('data/processed/raw_val.csv', nrows=100)
        prepared = load_npz('data/processed/X_val.npz')[:100]
        actual = self.model.preprocessor.transform(frame[FEATURES])
        self.assertEqual((prepared != actual).nnz, 0)
        with patch.object(self.model.preprocessor, 'fit', side_effect=AssertionError('fit forbidden')):
            self.model.predict(frame[FEATURES])

    def test_group_partitions_and_train_only_categories(self):
        frames = {s: pd.read_csv(f'data/processed/raw_{s}.csv') for s in ['train', 'val', 'test']}
        signatures = {s: set(feature_signatures(d[FEATURES])) for s, d in frames.items()}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(signatures[a] & signatures[b])
        for name, values in zip(CATEGORICAL, self.model.preprocessor.named_transformers_['categorical'].categories_):
            self.assertEqual(set(values), set(frames['train'][name]))
        for s in ['train', 'val']:
            self.assertEqual(set(frames[s]['attack_cat']), set(CLASSES))
        manifest = pd.read_csv('data/processed/split_manifest.csv')
        self.assertEqual(manifest.groupby('signature')['partition'].nunique().max(), 1)

    def test_modified_artifact_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            path.joinpath('artifact_manifest.json').write_text(json.dumps({
                'schema_version': 1, 'sha256': {'pipeline.joblib': 'wrong'}}))
            path.joinpath('pipeline.joblib').write_bytes(b'not a trusted model')
            with patch('joblib.load', side_effect=AssertionError('must not load')), self.assertRaises(ValueError):
                load_pipeline(path)

    def test_prepared_provenance(self):
        for name, digest in self.model.metadata['prepared_input_sha256'].items():
            self.assertEqual(hashlib.sha256((Path('data/processed') / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
