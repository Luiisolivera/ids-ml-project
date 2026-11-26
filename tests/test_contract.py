import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ids_ml.contract import FEATURES, CATEGORICAL, validate_features
from ids_ml.pipeline import IDSPipeline
from src.entrenar_ids import compute_xgb_weights, valid_xgb_weighting
from src.preparar_datos_ids import feature_signatures


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {f: ('tcp' if f in CATEGORICAL else 0) for f in FEATURES}

    def test_named_columns_reordered(self):
        frame = validate_features(dict(reversed(list(self.row.items()))))
        self.assertEqual(list(frame), FEATURES)

    def test_reject_invalid(self):
        cases = []
        missing = self.row.copy(); missing.pop('dur'); cases.append(missing)
        cases.append({**self.row, 'label': 0})
        for name, value in [('dur', None), ('dur', float('inf')), ('dur', -1),
                            ('dur', '1'), ('spkts', .5), ('proto', ''),
                            ('proto', ' tcp'), ('state', 1), ('dur', True),
                            ('is_sm_ips_ports', 2)]:
            cases.append({**self.row, name: value})
        cases.extend([[], [[0] * len(FEATURES)]])
        for data in cases:
            with self.subTest(data=str(data)[:60]), self.assertRaises(ValueError):
                validate_features(data)

    def test_duplicate_columns(self):
        frame = pd.DataFrame([self.row])
        frame = pd.concat([frame, frame[['dur']]], axis=1)
        with self.assertRaises(ValueError):
            validate_features(frame)

    def test_threshold_exact_tie_and_routing(self):
        rf, xgb, encoder = Mock(), Mock(), Mock()
        rf.classes_ = np.array([0, 1])
        rf.predict_proba.return_value = np.array([[.45, .55], [.44, .56]])
        xgb.predict.return_value = np.array([0])
        encoder.inverse_transform.return_value = np.array(['Generic'])
        pipeline = IDSPipeline(None, rf, xgb, encoder, .55, {}, {})
        labels, names, _ = pipeline.predict_transformed(csr_matrix([[1], [2]]))
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(names, ['Normal', 'Generic'])
        self.assertEqual(xgb.predict.call_args.args[0].shape[0], 1)

    def test_signatures_exclude_index(self):
        a = pd.DataFrame([self.row, self.row])
        self.assertEqual(*feature_signatures(a))

    def test_smoothed_xgb_weights(self):
        y = np.array([0] * 100 + [1] * 10 + [2] * 1)
        self.assertIsNone(compute_xgb_weights('none', y))
        for scheme in ['power_0.25', 'power_0.5', 'power_0.75', 'balanced']:
            self.assertTrue(valid_xgb_weighting(scheme))
            w = compute_xgb_weights(scheme, y)
            self.assertAlmostEqual(float(w.mean()), 1.0, places=12)
            self.assertTrue(w[0] < w[100] < w[-1])
        for invalid in ['power_0', 'power_1', 'power_1.5', 'power_abc', 'unknown']:
            self.assertFalse(valid_xgb_weighting(invalid))
            with self.assertRaises(ValueError):
                compute_xgb_weights(invalid, y)


if __name__ == '__main__':
    unittest.main()

