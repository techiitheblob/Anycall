"""AnyCall SQLite Storage — Prototype bank and detection event persistence."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class PrototypeStore:
    """Persistent SQLite-backed storage for species prototypes and detection events.

    Tables
    ------
    prototypes  : Species label → centroid vector (stored as JSON float list) + metadata.
    detections  : Timestamped detection log (species, score, backbone, audio path).

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it does not exist.
        Pass `:memory:` for an in-memory database (testing).
    """

    def __init__(self, db_path: str | Path = "anycall.db") -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS prototypes (
                label        TEXT PRIMARY KEY,
                centroid     TEXT NOT NULL,      -- JSON float array
                n_support    INTEGER DEFAULT 0,
                backbone     TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at  TEXT NOT NULL,
                label        TEXT NOT NULL,
                score        REAL NOT NULL,
                backbone     TEXT DEFAULT '',
                audio_path   TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_detections_label
                ON detections (label);
            CREATE INDEX IF NOT EXISTS idx_detections_detected_at
                ON detections (detected_at);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Prototype CRUD
    # ------------------------------------------------------------------

    def save_prototype(
        self,
        label: str,
        centroid: np.ndarray,
        n_support: int = 0,
        backbone: str = "",
    ) -> None:
        """Insert or replace a species prototype centroid."""
        now = _utcnow()
        centroid_json = json.dumps(centroid.flatten().tolist())

        existing = self._conn.execute(
            "SELECT created_at FROM prototypes WHERE label = ?", (label,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now

        self._conn.execute(
            """
            INSERT INTO prototypes (label, centroid, n_support, backbone, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                centroid   = excluded.centroid,
                n_support  = excluded.n_support,
                backbone   = excluded.backbone,
                updated_at = excluded.updated_at
            """,
            (label, centroid_json, n_support, backbone, created_at, now),
        )
        self._conn.commit()

    def load_prototype(self, label: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load a prototype centroid by label.

        Returns
        -------
        (centroid, n_support) or None if the label is not found.
        """
        row = self._conn.execute(
            "SELECT centroid, n_support FROM prototypes WHERE label = ?", (label,)
        ).fetchone()
        if row is None:
            return None
        centroid = np.array(json.loads(row["centroid"]), dtype=np.float32)
        return centroid, row["n_support"]

    def load_all_prototypes(self) -> Dict[str, Tuple[np.ndarray, int]]:
        """Return all stored prototypes as {label: (centroid, n_support)}."""
        rows = self._conn.execute(
            "SELECT label, centroid, n_support FROM prototypes"
        ).fetchall()
        return {
            row["label"]: (
                np.array(json.loads(row["centroid"]), dtype=np.float32),
                row["n_support"],
            )
            for row in rows
        }

    def delete_prototype(self, label: str) -> None:
        """Remove a single prototype from the bank."""
        self._conn.execute("DELETE FROM prototypes WHERE label = ?", (label,))
        self._conn.commit()

    def list_labels(self) -> List[str]:
        """Return all enrolled species labels."""
        rows = self._conn.execute("SELECT label FROM prototypes").fetchall()
        return [row["label"] for row in rows]

    # ------------------------------------------------------------------
    # Detection log
    # ------------------------------------------------------------------

    def log_detection(
        self,
        label: str,
        score: float,
        backbone: str = "",
        audio_path: str = "",
    ) -> int:
        """Append a detection event. Returns the new row ID."""
        cur = self._conn.execute(
            """
            INSERT INTO detections (detected_at, label, score, backbone, audio_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_utcnow(), label, float(score), backbone, audio_path),
        )
        self._conn.commit()
        return cur.lastrowid

    def query_detections(
        self,
        label: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Fetch recent detection events, optionally filtered by species label."""
        if label:
            rows = self._conn.execute(
                "SELECT * FROM detections WHERE label = ? ORDER BY id DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "PrototypeStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
