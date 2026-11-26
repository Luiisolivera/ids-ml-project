"""Almacenamiento ligero de alertas para el MVP del backend (SQLite)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ids_ml.contract import ATTACKS


class AlertStore:
    """Persistencia ligera en SQLite (en memoria o archivo) para ataques detectados."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            resolved = Path(self.db_path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(resolved)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    attack_cat TEXT NOT NULL,
                    probability_attack REAL NOT NULL,
                    unknown_categories TEXT NOT NULL,
                    proto TEXT NOT NULL,
                    service TEXT NOT NULL,
                    state TEXT NOT NULL,
                    dur REAL NOT NULL,
                    sbytes INTEGER NOT NULL,
                    dbytes INTEGER NOT NULL
                )
                """
            )

    def record_predictions(
        self,
        rows: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[int]:
        """Registra como alerta cada fila cuya predicción sea ataque (label == 1)."""
        now_utc = datetime.now(timezone.utc).isoformat()
        alert_ids: list[int] = []
        with self._lock, self._conn:
            for row, pred in zip(rows, predictions):
                if int(pred["label"]) != 1:
                    continue
                cursor = self._conn.execute(
                    """
                    INSERT INTO alerts (
                        created_utc, attack_cat, probability_attack, unknown_categories,
                        proto, service, state, dur, sbytes, dbytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_utc,
                        str(pred["attack_cat"]),
                        float(pred["probability_attack"]),
                        json.dumps(list(pred.get("unknown_categories", []))),
                        str(row["proto"]),
                        str(row["service"]),
                        str(row["state"]),
                        float(row["dur"]),
                        int(row["sbytes"]),
                        int(row["dbytes"]),
                    ),
                )
                alert_ids.append(int(cursor.lastrowid))
        return alert_ids

    def list_alerts(
        self,
        limit: int = 50,
        attack_cat: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lista las alertas más recientes, opcionalmente filtradas por categoría."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 1000:
            raise ValueError("limit debe ser un entero entre 1 y 1000")
        if attack_cat is not None and attack_cat not in ATTACKS:
            raise ValueError(f"attack_cat inválida: {attack_cat}")

        query = """
            SELECT id, created_utc, attack_cat, probability_attack, unknown_categories,
                   proto, service, state, dur, sbytes, dbytes
            FROM alerts
        """
        params: list[Any] = []
        if attack_cat is not None:
            query += " WHERE attack_cat = ?"
            params.append(attack_cat)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        return [
            {
                "id": int(r["id"]),
                "created_utc": r["created_utc"],
                "attack_cat": r["attack_cat"],
                "probability_attack": float(r["probability_attack"]),
                "unknown_categories": json.loads(r["unknown_categories"]),
                "flow_summary": {
                    "proto": r["proto"],
                    "service": r["service"],
                    "state": r["state"],
                    "dur": float(r["dur"]),
                    "sbytes": int(r["sbytes"]),
                    "dbytes": int(r["dbytes"]),
                },
            }
            for r in rows
        ]

    def summary(self) -> dict[str, Any]:
        """Devuelve conteo total de alertas y desglose por categoría de ataque."""
        with self._lock:
            total = int(
                self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            )
            grouped = self._conn.execute(
                "SELECT attack_cat, COUNT(*) AS cnt FROM alerts GROUP BY attack_cat ORDER BY cnt DESC"
            ).fetchall()
        return {
            "total_alerts": total,
            "by_category": {row["attack_cat"]: int(row["cnt"]) for row in grouped},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
