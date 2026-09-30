"""Reseed anycall.db prototypes from data/processed_roi after rebuild_roi_db.py finishes.

This replaces the old 5-clip VAD prototypes with new 30-clip HPSS prototypes.
Run this after rebuild_roi_db.py completes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

from anycall.data.species import SPECIES_CATALOG
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.storage.db import DatabaseManager


def reseed():
    print("=" * 60)
    print("  Reseeding DB from HPSS ROI clips (30 clips/species)")
    print("=" * 60)

    roi_dir = Path("data/processed_roi")
    if not roi_dir.exists():
        print("ERROR: data/processed_roi not found. Run rebuild_roi_db.py first.")
        return

    bb = BirdNetBackbone()
    db = DatabaseManager("anycall.db")

    # Wipe old prototypes
    db._conn.execute("DELETE FROM species")
    db._conn.commit()
    print("Cleared old prototypes from DB.")

    species_dirs = sorted([d for d in roi_dir.iterdir() if d.is_dir()])
    print(f"Found {len(species_dirs)} species directories.\n")

    total_enrolled = 0
    total_clips = 0

    for sp_dir in tqdm(species_dirs, desc="Enrolling"):
        sp_id = sp_dir.name
        wavs = sorted(list(sp_dir.glob("*.wav")))
        if not wavs:
            print(f"  SKIP {sp_id}: no WAVs")
            continue

        embeddings = []
        for wf in wavs:
            try:
                audio, sr = sf.read(str(wf), dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                emb = bb.embed(audio, sr=sr)
                embeddings.append(emb)
            except Exception as e:
                print(f"    Warning: {wf.name}: {e}")

        if not embeddings:
            print(f"  SKIP {sp_id}: all files failed")
            continue

        stacked = np.stack(embeddings, axis=0)
        centroid = np.mean(stacked, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid = (centroid / norm).astype(np.float32)

        meta = SPECIES_CATALOG.get(sp_id)
        common_name = meta.common_name if meta else sp_id.replace("_", " ").title()
        taxon = (meta.taxon.value if hasattr(meta.taxon, "value") else str(meta.taxon)).capitalize() if meta else "Aves"

        db.save_prototype(
            species_id=sp_id,
            common_name=common_name,
            taxon=taxon,
            prototype=centroid,
            radius=0.18,
            sample_count=len(embeddings),
        )
        total_enrolled += 1
        total_clips += len(embeddings)

    print(f"\nDone. Enrolled {total_enrolled} species from {total_clips} HPSS clips.")

    # Verify
    rows = db._conn.execute("SELECT species_id, sample_count FROM species ORDER BY sample_count DESC").fetchall()
    print("\nTop 5 species by clip count:")
    for r in rows[:5]:
        print(f"  {r[0]:40s}: {r[1]} clips")
    low = [(r[0], r[1]) for r in rows if r[1] < 10]
    if low:
        print(f"\nWARNING: {len(low)} species with < 10 clips: {[s for s,_ in low]}")


if __name__ == "__main__":
    reseed()
