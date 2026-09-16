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

            CREATE TABLE IF NOT EXISTS unidentified_sounds (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at  TEXT NOT NULL,
                embedding    TEXT NOT NULL,      -- JSON float array
                best_match   TEXT DEFAULT '',
                score        REAL NOT NULL,
                audio_path   TEXT DEFAULT '',
                cluster_id   TEXT DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_detections_label
                ON detections (label);
            CREATE INDEX IF NOT EXISTS idx_detections_detected_at
                ON detections (detected_at);
            CREATE INDEX IF NOT EXISTS idx_unidentified_cluster
                ON unidentified_sounds (cluster_id);
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
    # Unidentified Sounds CRUD
    # ------------------------------------------------------------------

    def save_unidentified(
        self,
        embedding: np.ndarray,
        best_match: str = "",
        score: float = 0.0,
        audio_path: str = "",
        detected_at: Optional[str] = None,
    ) -> int:
        """Store an unidentified out-of-bank sound for novel cluster discovery."""
        now = detected_at or _utcnow()
        emb_json = json.dumps(embedding.flatten().tolist())
        cursor = self._conn.execute(
            """
            INSERT INTO unidentified_sounds (detected_at, embedding, best_match, score, audio_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now, emb_json, best_match, float(score), audio_path),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_unidentified(self, cluster_id: Optional[str] = None, limit: int = 500) -> List[Dict]:
        """Query unidentified sound recordings, optionally filtered by cluster."""
        if cluster_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM unidentified_sounds WHERE cluster_id = ? ORDER BY id DESC LIMIT ?",
                (cluster_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM unidentified_sounds ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["embedding"] = np.array(json.loads(d["embedding"]), dtype=np.float32)
            results.append(d)
        return results

    def assign_unidentified_cluster(self, sound_ids: List[int], cluster_id: str) -> None:
        """Assign a set of unidentified sounds to a discovered novel cluster."""
        if not sound_ids:
            return
        placeholders = ",".join("?" * len(sound_ids))
        self._conn.execute(
            f"UPDATE unidentified_sounds SET cluster_id = ? WHERE id IN ({placeholders})",
            [cluster_id] + sound_ids,
        )
        self._conn.commit()

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


class DatabaseManager:
    """SQLite Database Manager matching Tier 3/4 E2E test interface specifications."""

    _instances: set = set()

    def __init__(self, db_path: str | Path = "anycall.db") -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        DatabaseManager._instances.add(self)

    @classmethod
    def close_all(cls) -> None:
        for inst in list(cls._instances):
            try:
                inst.close()
            except Exception:
                pass
        cls._instances.clear()

    def _create_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS species (
                species_id   TEXT PRIMARY KEY,
                common_name  TEXT DEFAULT '',
                taxon        TEXT DEFAULT '',
                prototype    BLOB NOT NULL,
                radius       REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                species_id   TEXT NOT NULL,
                confidence   REAL NOT NULL,
                audio_hash   TEXT DEFAULT '',
                threshold    REAL DEFAULT 0.0
            );
        """)
        self._conn.commit()

    def save_prototype(
        self,
        species_id: str,
        common_name: str,
        taxon: str,
        prototype: np.ndarray,
        radius: float = 0.0,
        sample_count: int = 1,
    ) -> None:
        proto_bytes = prototype.astype(np.float32).tobytes()
        now = _utcnow()
        self._conn.execute(
            """
            INSERT INTO species (species_id, common_name, taxon, prototype, radius, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(species_id) DO UPDATE SET
                common_name  = excluded.common_name,
                taxon        = excluded.taxon,
                prototype    = excluded.prototype,
                radius       = excluded.radius,
                sample_count = excluded.sample_count,
                updated_at   = excluded.updated_at
            """,
            (species_id, common_name, taxon, proto_bytes, float(radius), int(sample_count), now),
        )
        self._conn.commit()

    def get_prototype(self, species_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM species WHERE species_id = ?", (species_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["prototype"] = np.frombuffer(d["prototype"], dtype=np.float32)
        return d

    def log_detection(
        self,
        species_id: str,
        confidence: float,
        audio_hash: str = "",
        threshold: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> int:
        now = timestamp or _utcnow()
        cursor = self._conn.execute(
            """
            INSERT INTO detections (timestamp, species_id, confidence, audio_hash, threshold)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now, species_id, float(confidence), audio_hash, float(threshold)),
        )
        self._conn.commit()
        return cursor.lastrowid

    def list_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_detections(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.list_detections(limit=limit)

    def close(self) -> None:
        DatabaseManager._instances.discard(self)
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

