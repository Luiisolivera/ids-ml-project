"""Pruebas de integración y contrato del MVP de backend."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.adapter import (
    BatchSizeError,
    ContractValidationError,
    InferenceAdapter,
    ServiceNotReadyError,
)
from backend.alerts import AlertStore
from backend.server import IDSBackendApp, create_http_server


class BackendMVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = Path("models/final").resolve()
        cls.example_input = json.loads((cls.model_dir / "example_input.json").read_text())
        cls.example_expected = json.loads((cls.model_dir / "example_expected.json").read_text())
        cls.adapter = InferenceAdapter(model_dir=cls.model_dir, max_batch_size=4, eager_load=True)

    def setUp(self) -> None:
        self.alerts = AlertStore(":memory:")
        self.app = IDSBackendApp(adapter=self.adapter, alert_store=self.alerts)

    def tearDown(self) -> None:
        self.alerts.close()

    def test_adapter_uses_absolute_path_and_never_calls_fit(self) -> None:
        rel_adapter = InferenceAdapter(model_dir=Path("models/final"), max_batch_size=10)
        self.assertTrue(rel_adapter.model_dir.is_absolute())
        self.assertTrue(rel_adapter.is_ready)

        with (
            patch.object(
                rel_adapter._pipeline.preprocessor,
                "fit",
                side_effect=AssertionError("fit prohibido en backend"),
            ),
            patch(
                "sklearn.ensemble.RandomForestClassifier.fit",
                side_effect=AssertionError("fit prohibido en backend"),
            ),
            patch(
                "xgboost.XGBClassifier.fit",
                side_effect=AssertionError("fit prohibido en backend"),
            ),
        ):
            _, predictions = rel_adapter.predict(self.example_input)

        self.assertEqual(len(predictions), len(self.example_expected))
        for actual, expected in zip(predictions, self.example_expected):
            self.assertEqual(actual["label"], expected["label"])
            self.assertEqual(actual["attack_cat"], expected["attack_cat"])
            self.assertAlmostEqual(
                actual["probability_attack"],
                expected["probability_attack"],
                places=12,
            )
            self.assertEqual(actual["unknown_categories"], expected["unknown_categories"])

    def test_predict_endpoint_matches_example_and_records_alerts(self) -> None:
        status, body = self.app.handle_request(
            "POST",
            "/predict",
            json.dumps(self.example_input).encode("utf-8"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["attacks_detected"], 1)
        self.assertEqual(len(body["alert_ids"]), 1)
        self.assertEqual(body["predictions"][1]["label"], 1)
        self.assertEqual(body["predictions"][1]["attack_cat"], "Reconnaissance")

        status_alerts, alerts_body = self.app.handle_request("GET", "/alerts?limit=10")
        self.assertEqual(status_alerts, 200)
        self.assertEqual(alerts_body["total_alerts"], 1)
        self.assertEqual(alerts_body["by_category"], {"Reconnaissance": 1})
        self.assertEqual(len(alerts_body["alerts"]), 1)
        self.assertEqual(alerts_body["alerts"][0]["attack_cat"], "Reconnaissance")

    def test_schema_and_batch_errors_mapped_correctly(self) -> None:
        # 1. Lote que supera max_batch_size=4 -> 413
        too_many = [self.example_input[0]] * 5
        status, body = self.app.handle_request(
            "POST",
            "/predict",
            json.dumps(too_many).encode("utf-8"),
        )
        self.assertEqual(status, 413)
        self.assertEqual(body["error"], "batch_size_exceeded")

        # 2. Columna extra prohibida (id / label) -> 422
        with_id = {**self.example_input[0], "id": 99}
        status, body = self.app.handle_request(
            "POST",
            "/predict",
            json.dumps(with_id).encode("utf-8"),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "contract_validation_error")

        # 3. Columna faltante -> 422
        missing_dur = dict(self.example_input[0])
        missing_dur.pop("dur")
        status, body = self.app.handle_request(
            "POST",
            "/predict",
            json.dumps(missing_dur).encode("utf-8"),
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "contract_validation_error")

        # 4. JSON malformado o con NaN -> 400
        status, body = self.app.handle_request("POST", "/predict", b'{"dur": NaN}')
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_readiness_when_artifact_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_adapter = InferenceAdapter(model_dir=tmp, eager_load=False)
            with self.assertRaises(ServiceNotReadyError):
                bad_adapter.load()
            unready_app = IDSBackendApp(adapter=bad_adapter)

            status_health, health_body = unready_app.handle_request("GET", "/health")
            self.assertEqual(status_health, 200)
            self.assertFalse(health_body["ready"])

            status_ready, ready_body = unready_app.handle_request("GET", "/ready")
            self.assertEqual(status_ready, 503)
            self.assertFalse(ready_body["ready"])

            status_pred, pred_body = unready_app.handle_request(
                "POST",
                "/predict",
                json.dumps(self.example_input[0]).encode("utf-8"),
            )
            self.assertEqual(status_pred, 503)
            self.assertEqual(pred_body["error"], "service_not_ready")

    def test_live_http_server_roundtrip(self) -> None:
        server = create_http_server(host="127.0.0.1", port=0, app=self.app)
        host, port = server.server_address[:2]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://{host}:{port}"
            with urlopen(f"{base_url}/ready") as resp:
                self.assertEqual(resp.status, 200)
                ready_data = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(ready_data["ready"])

            with urlopen(f"{base_url}/") as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/html", resp.headers.get("Content-Type", ""))
                html_body = resp.read().decode("utf-8")
                self.assertIn("IDS basado en Machine Learning", html_body)

            with urlopen(f"{base_url}/styles.css") as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/css", resp.headers.get("Content-Type", ""))

            with urlopen(f"{base_url}/app.js") as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("application/javascript", resp.headers.get("Content-Type", ""))

            with urlopen(f"{base_url}/metrics") as resp:
                self.assertEqual(resp.status, 200)
                metrics_data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("validation", metrics_data)
                self.assertIn("test_diagnostic", metrics_data)
                self.assertEqual(len(metrics_data["example_input"]), 3)

            req = Request(
                f"{base_url}/predict",
                data=json.dumps({"rows": self.example_input}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                pred_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(pred_data["count"], 3)
                self.assertEqual(pred_data["attacks_detected"], 1)

            bad_req = Request(
                f"{base_url}/predict",
                data=json.dumps({**self.example_input[0], "dur": -5}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(bad_req)
            try:
                self.assertEqual(ctx.exception.code, 422)
            finally:
                ctx.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_frontend_static_and_metrics_handlers(self) -> None:
        static_index = self.app.handle_static("GET", "/")
        self.assertIsNotNone(static_index)
        assert static_index is not None
        status, ctype, raw = static_index
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("FPR diagnóstico: 23.61 %", raw.decode("utf-8"))

        bad_method = self.app.handle_static("POST", "/")
        self.assertIsNotNone(bad_method)
        assert bad_method is not None
        self.assertEqual(bad_method[0], 405)

        status_m, metrics_payload = self.app.handle_request("GET", "/metrics")
        self.assertEqual(status_m, 200)
        self.assertIsNotNone(metrics_payload["validation"])
        self.assertIsNotNone(metrics_payload["test_diagnostic"])
        self.assertAlmostEqual(metrics_payload["test_diagnostic"]["fpr"], 0.23605405405405405, places=9)
        self.assertEqual(len(metrics_payload["thresholds_validation"]), 12)


if __name__ == "__main__":
    unittest.main()

