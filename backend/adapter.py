"""Adaptador de inferencia para el backend MVP del IDS.

Responsabilidades:
- Resolver ruta absoluta hacia la carpeta del artefacto final (models/final).
- Cargar una única vez IDSPipeline al iniciar mediante load_pipeline (nunca fit).
- Exponer estado de readiness según disponibilidad e integridad del artefacto.
- Controlar tamaño máximo de lote (max_batch_size).
- Delegar validación de contrato, preprocesamiento e inferencia íntegramente
  en IDSPipeline.predict y mapear errores de schema/lote.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ids_ml.artifacts import load_pipeline
from ids_ml.pipeline import IDSPipeline

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "final"
DEFAULT_MAX_BATCH_SIZE = 256


class ServiceNotReadyError(RuntimeError):
    """El artefacto del modelo no está disponible o falló la verificación."""


class BatchSizeError(ValueError):
    """El lote recibido supera el máximo configurado para el servicio."""


class ContractValidationError(ValueError):
    """Los datos de entrada incumplen el contrato de 42 features del IDS."""


class InferenceAdapter:
    """Adaptador sin estado de entrenamiento que envuelve IDSPipeline."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        max_batch_size: int | None = None,
        eager_load: bool = True,
    ) -> None:
        raw_dir = model_dir or os.environ.get("IDS_MODEL_DIR") or DEFAULT_MODEL_DIR
        self.model_dir: Path = Path(raw_dir).expanduser().resolve()
        if not self.model_dir.is_absolute():
            raise ValueError("model_dir debe resolverse a una ruta absoluta")

        raw_batch = (
            max_batch_size
            if max_batch_size is not None
            else int(os.environ.get("IDS_MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE))
        )
        if not isinstance(raw_batch, int) or isinstance(raw_batch, bool) or raw_batch < 1:
            raise ValueError("max_batch_size debe ser un entero positivo")
        self.max_batch_size: int = raw_batch

        self._pipeline: IDSPipeline | None = None
        self._load_error: str | None = None

        if eager_load:
            self.load()

    def load(self) -> IDSPipeline:
        """Carga y verifica el artefacto una sola vez desde model_dir."""
        try:
            self._pipeline = load_pipeline(self.model_dir)
            self._load_error = None
            return self._pipeline
        except Exception as exc:
            self._pipeline = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise ServiceNotReadyError(
                f"Artefacto no disponible en {self.model_dir}: {self._load_error}"
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None and self._load_error is None

    def readiness(self) -> dict[str, Any]:
        """Devuelve el estado de disponibilidad y metadatos básicos del modelo cargado."""
        if not self.is_ready or self._pipeline is None:
            return {
                "ready": False,
                "model_dir": str(self.model_dir),
                "max_batch_size": self.max_batch_size,
                "error": self._load_error or "Modelo no cargado",
            }
        meta = self._pipeline.metadata
        return {
            "ready": True,
            "model_dir": str(self.model_dir),
            "max_batch_size": self.max_batch_size,
            "schema_version": self._pipeline.contract.get("schema_version", 1),
            "threshold": float(self._pipeline.threshold),
            "selected_rf_min_samples_leaf": meta.get("selected_rf_min_samples_leaf"),
            "selected_xgb_weighting": meta.get("selected_xgb_weighting"),
            "evaluation_status": meta.get("evaluation_status"),
        }

    def contract(self) -> dict[str, Any]:
        """Devuelve el contrato de features encapsulado en el artefacto."""
        if not self.is_ready or self._pipeline is None:
            raise ServiceNotReadyError(self._load_error or "Modelo no cargado")
        return self._pipeline.contract

    @staticmethod
    def normalize_payload(payload: Any) -> list[dict[str, Any]]:
        """Normaliza dict, list[dict] o sobre {'rows': [...]} a una lista de filas."""
        if isinstance(payload, dict):
            if set(payload.keys()) == {"rows"} and isinstance(payload["rows"], list):
                rows = payload["rows"]
            else:
                rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ContractValidationError(
                "El cuerpo JSON debe ser un objeto de features, una lista de objetos o {'rows': [...]}"
            )

        if not rows:
            raise ContractValidationError("Se requiere al menos una fila de features")
        if not all(isinstance(row, dict) for row in rows):
            raise ContractValidationError("Cada fila debe ser un objeto JSON con las 42 features")
        return rows

    def predict(self, payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Valida tamaño de lote y delega íntegramente en IDSPipeline.predict.

        Retorna (rows_normalizadas, predicciones).
        """
        if not self.is_ready or self._pipeline is None:
            raise ServiceNotReadyError(self._load_error or "Modelo no cargado")

        rows = self.normalize_payload(payload)
        if len(rows) > self.max_batch_size:
            raise BatchSizeError(
                f"Tamaño de lote {len(rows)} excede max_batch_size={self.max_batch_size}"
            )

        try:
            predictions = self._pipeline.predict(rows)
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc

        return rows, predictions
