"""Servidor HTTP REST + Dashboard Frontend para el IDS (librería estándar de Python).

Endpoints API:
- GET  /health   -> Estado de vida del proceso (200).
- GET  /ready    -> Disponibilidad e integridad del artefacto ML (200 / 503).
- GET  /contract -> Contrato verificable de las 42 features (200 / 503).
- GET  /metrics  -> Métricas auditadas (validación, test diagnóstico, umbrales) y ejemplos (200).
- POST /predict  -> Inferencia de lote de flujos y registro de alertas (200 / 400 / 413 / 422 / 503).
- GET  /alerts   -> Consulta de alertas detectadas y resumen por categoría (200 / 400).

Archivos estáticos del Dashboard:
- GET  /           -> frontend/index.html
- GET  /styles.css -> frontend/styles.css
- GET  /app.js     -> frontend/app.js
"""
from __future__ import annotations

import argparse
import csv
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend.adapter import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MODEL_DIR,
    BatchSizeError,
    ContractValidationError,
    InferenceAdapter,
    ServiceNotReadyError,
)
from backend.alerts import AlertStore

DEFAULT_MAX_BODY_BYTES = 4 * 1024 * 1024  # 4 MiB
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DIR = REPO_ROOT / "frontend"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "final"


def _reject_nonfinite_json(constant: str) -> Any:
    raise ValueError(f"Constante JSON no permitida: {constant}")


def _load_json_safe(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_safe(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k is not None and v is not None})
    return rows


class IDSBackendApp:
    """Aplicación de backend desacoplada del transporte para facilitar pruebas y HTTP."""

    def __init__(
        self,
        adapter: InferenceAdapter,
        alert_store: AlertStore | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        reports_dir: Path | None = None,
        frontend_dir: Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.alerts = alert_store if alert_store is not None else AlertStore(":memory:")
        self.max_body_bytes = max_body_bytes
        self.reports_dir = (reports_dir or DEFAULT_REPORTS_DIR).expanduser().resolve()
        self.frontend_dir = (frontend_dir or DEFAULT_FRONTEND_DIR).expanduser().resolve()

    def handle_static(self, method: str, raw_url: str) -> tuple[int, str, bytes] | None:
        """Si la ruta corresponde a un activo del frontend, retorna (status, content_type, bytes)."""
        parsed = urlparse(raw_url)
        path = parsed.path or "/"
        static_map: dict[str, tuple[str, str]] = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
        }
        if path not in static_map:
            return None
        if method != "GET":
            err = json.dumps({"error": "Method not allowed"}).encode("utf-8")
            return int(HTTPStatus.METHOD_NOT_ALLOWED), "application/json; charset=utf-8", err

        filename, content_type = static_map[path]
        target = (self.frontend_dir / filename).resolve()
        if not target.is_relative_to(self.frontend_dir) or not target.is_file():
            err = json.dumps({"error": "not_found", "detail": f"Activo no encontrado: {filename}"}).encode("utf-8")
            return int(HTTPStatus.NOT_FOUND), "application/json; charset=utf-8", err

        return int(HTTPStatus.OK), content_type, target.read_bytes()

    def _build_metrics_payload(self) -> dict[str, Any]:
        val_eval = _load_json_safe(self.reports_dir / "evaluation_val.json")
        test_eval = _load_json_safe(self.reports_dir / "evaluation_test.json")
        selection = _load_json_safe(self.reports_dir / "model_selection.json")
        thresholds = _load_csv_safe(self.reports_dir / "thresholds_validation.csv")
        examples = _load_json_safe(self.adapter.model_dir / "example_input.json")
        expected = _load_json_safe(self.adapter.model_dir / "example_expected.json")

        selection_summary = None
        if isinstance(selection, dict):
            selection_summary = {
                "selected_rf_min_samples_leaf": selection.get("selected_rf_min_samples_leaf"),
                "selected_threshold": selection.get("selected_threshold"),
                "rf_candidates": selection.get("rf_candidates", []),
                "xgb_candidates": [
                    {
                        "weighting": c.get("weighting"),
                        "fit_seconds": c.get("fit_seconds"),
                        "macro_f1": c.get("metrics", {})
                        .get("cascade", {})
                        .get("macro avg", {})
                        .get("f1-score"),
                        "weighted_f1": c.get("metrics", {})
                        .get("cascade", {})
                        .get("weighted avg", {})
                        .get("f1-score"),
                    }
                    for c in selection.get("candidates", [])
                ],
            }

        return {
            "evaluation_status": (
                "Fase ML parcial: validación participó en selección; el test oficial está "
                "históricamente expuesto y es solo diagnóstico (FPR 23.61 %). "
                "Falta evaluación con un conjunto independiente."
            ),
            "validation": val_eval,
            "test_diagnostic": test_eval,
            "thresholds_validation": thresholds,
            "selection_summary": selection_summary,
            "example_input": examples if isinstance(examples, list) else [],
            "example_expected": expected if isinstance(expected, list) else [],
        }

    def handle_request(
        self,
        method: str,
        raw_url: str,
        body: bytes = b"",
    ) -> tuple[int, dict[str, Any]]:
        """Procesa una solicitud HTTP de la API y retorna (status_code, json_payload)."""
        parsed = urlparse(raw_url)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query, keep_blank_values=True)

        if path == "/health":
            if method != "GET":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            return int(HTTPStatus.OK), {
                "status": "ok",
                "ready": self.adapter.is_ready,
            }

        if path == "/ready":
            if method != "GET":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            info = self.adapter.readiness()
            status = HTTPStatus.OK if info["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            return int(status), info

        if path == "/contract":
            if method != "GET":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            try:
                return int(HTTPStatus.OK), self.adapter.contract()
            except ServiceNotReadyError as exc:
                return int(HTTPStatus.SERVICE_UNAVAILABLE), {
                    "error": "service_not_ready",
                    "detail": str(exc),
                }

        if path == "/metrics":
            if method != "GET":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            return int(HTTPStatus.OK), self._build_metrics_payload()

        if path == "/alerts":
            if method != "GET":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            try:
                limit = int(query["limit"][0]) if "limit" in query else 50
                attack_cat = query["attack_cat"][0] if "attack_cat" in query else None
                if attack_cat == "":
                    attack_cat = None
                items = self.alerts.list_alerts(limit=limit, attack_cat=attack_cat)
                summary = self.alerts.summary()
                return int(HTTPStatus.OK), {
                    "total_alerts": summary["total_alerts"],
                    "by_category": summary["by_category"],
                    "alerts": items,
                }
            except (ValueError, KeyError) as exc:
                return int(HTTPStatus.BAD_REQUEST), {
                    "error": "invalid_query",
                    "detail": str(exc),
                }

        if path == "/predict":
            if method != "POST":
                return int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "Method not allowed"}
            if len(body) > self.max_body_bytes:
                return int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE), {
                    "error": "payload_too_large",
                    "detail": f"El cuerpo supera max_body_bytes={self.max_body_bytes}",
                }
            if not body:
                return int(HTTPStatus.BAD_REQUEST), {
                    "error": "invalid_json",
                    "detail": "El cuerpo de la petición está vacío",
                }
            try:
                payload = json.loads(
                    body.decode("utf-8"),
                    parse_constant=_reject_nonfinite_json,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                return int(HTTPStatus.BAD_REQUEST), {
                    "error": "invalid_json",
                    "detail": str(exc),
                }

            try:
                rows, predictions = self.adapter.predict(payload)
            except ServiceNotReadyError as exc:
                return int(HTTPStatus.SERVICE_UNAVAILABLE), {
                    "error": "service_not_ready",
                    "detail": str(exc),
                }
            except BatchSizeError as exc:
                return int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE), {
                    "error": "batch_size_exceeded",
                    "detail": str(exc),
                }
            except ContractValidationError as exc:
                return int(HTTPStatus.UNPROCESSABLE_ENTITY), {
                    "error": "contract_validation_error",
                    "detail": str(exc),
                }

            alert_ids = self.alerts.record_predictions(rows, predictions)
            return int(HTTPStatus.OK), {
                "count": len(predictions),
                "attacks_detected": len(alert_ids),
                "alert_ids": alert_ids,
                "predictions": predictions,
            }

        return int(HTTPStatus.NOT_FOUND), {
            "error": "not_found",
            "detail": f"Ruta no encontrada: {path}",
        }


def create_http_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    app: IDSBackendApp | None = None,
) -> ThreadingHTTPServer:
    """Crea un ThreadingHTTPServer enlazado a IDSBackendApp."""
    if app is None:
        adapter = InferenceAdapter(eager_load=True)
        app = IDSBackendApp(adapter=adapter)

    class RequestHandler(BaseHTTPRequestHandler):
        def _dispatch(self, method: str) -> None:
            static_resp = app.handle_static(method, self.path)
            if static_resp is not None:
                status, content_type, raw = static_resp
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            status, payload = app.handle_request(method, self.path, body)
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), RequestHandler)
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="Host de escucha (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Puerto de escucha (default: 8000)")
    ap.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Ruta al directorio del artefacto final (se resuelve a absoluta)",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Máximo de filas por petición a /predict",
    )
    ap.add_argument(
        "--alerts-db",
        default=":memory:",
        help="Ruta SQLite para alertas o ':memory:' (default: :memory:)",
    )
    args = ap.parse_args()

    adapter = InferenceAdapter(
        model_dir=args.model,
        max_batch_size=args.max_batch_size,
        eager_load=True,
    )
    alerts = AlertStore(args.alerts_db)
    app = IDSBackendApp(adapter=adapter, alert_store=alerts)
    server = create_http_server(host=args.host, port=args.port, app=app)
    host, port = server.server_address[:2]
    print(
        f"Servidor IDS MVP + Dashboard activo en http://{host}:{port} | "
        f"modelo={adapter.model_dir} | max_batch_size={adapter.max_batch_size}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        alerts.close()


if __name__ == "__main__":
    main()

