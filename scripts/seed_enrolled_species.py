"""AnyCall Seed Enrolled Species Script.

Extracts embeddings for all 33 curated resident Indian wildlife species
from data/processed/ and enrolls their centroid prototypes into anycall.db.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

from anycall.data.species import SPECIES_CATALOG
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.storage.db import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Seed AnyCall Database with Curated Species Prototypes")
    parser.add_argument("--db", type=str, default="anycall.db", help="Path to SQLite database")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed audio directory")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum audio samples per species prototype")
    args = parser.parse_args()

    processed_path = Path(args.data_dir)
    if not processed_path.exists():
        print(f"Error: {processed_path} does not exist!")
        sys.exit(1)

    print("=" * 72)
    print("       AnyCall Species Prototypical Enrollment & Database Seeder     ")
    print("=" * 72)
    print(f"  • Database       : {args.db}")
    print(f"  • Data Directory : {args.data_dir}")
    print(f"  • Backbone       : BirdNET EfficientNet-B0 (1024-dim, 48kHz)")
    print(f"  • Max Samples/Sp : {args.max_samples}")
    print("=" * 72)

    db_mgr = DatabaseManager(args.db)
    print("[1/3] Initializing BirdNET backbone model...")
    backbone = BirdNetBackbone()

    species_dirs = sorted([d for d in processed_path.iterdir() if d.is_dir()])
    print(f"[2/3] Found {len(species_dirs)} species directories to enroll.\n")

    enrolled_by_taxon = {"Aves": 0, "Insecta": 0, "Amphibia": 0, "Mammalia": 0, "Other": 0}
    total_enrolled = 0
    total_clips_embedded = 0

    for idx, sp_dir in enumerate(species_dirs, start=1):
        sp_id = sp_dir.name
        wav_files = sorted(list(sp_dir.glob("*.wav")))
        if not wav_files:
            print(f"  [{idx:02d}/{len(species_dirs):02d}] ⚠️  {sp_id}: No WAV files found, skipping.")
            continue

        selected_wavs = wav_files[:args.max_samples]
        embeddings = []

        for wf in selected_wavs:
            try:
                audio, sr = sf.read(str(wf))
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                emb = backbone.embed(audio, sr=sr)
                embeddings.append(emb)
            except Exception as e:
                print(f"      Warning: failed reading {wf.name}: {e}")

        if not embeddings:
            print(f"  [{idx:02d}/{len(species_dirs):02d}] ❌ {sp_id}: Failed to extract embeddings.")
            continue

        # Compute L2-normalized centroid prototype
        stacked = np.stack(embeddings, axis=0)
        centroid = np.mean(stacked, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid /= norm

        meta = SPECIES_CATALOG.get(sp_id)
        if meta:
            common_name = meta.common_name
            raw_taxon = meta.taxon.value if hasattr(meta.taxon, "value") else str(meta.taxon)
            taxon = raw_taxon.capitalize()
        else:
            common_name = sp_id.replace("_", " ").title()
            taxon = "Aves"

        db_mgr.save_prototype(
            species_id=sp_id,
            common_name=common_name,
            taxon=taxon,
            prototype=centroid,
            radius=0.18,
            sample_count=len(embeddings),
        )

        taxon_key = taxon if taxon in enrolled_by_taxon else "Other"
        enrolled_by_taxon[taxon_key] += 1
        total_enrolled += 1
        total_clips_embedded += len(embeddings)

        print(f"  [{idx:02d}/{len(species_dirs):02d}] ✅ [{taxon:<8}] {common_name:<26} ({sp_id}) — {len(embeddings)} clips")

    print("\n" + "=" * 72)
    print("                        Enrollment Summary                        ")
    print("=" * 72)
    for tax, cnt in enrolled_by_taxon.items():
        if cnt > 0:
            print(f"  • {tax:<10} : {cnt} species")
    print("-" * 72)
    print(f"  TOTAL ENROLLED : {total_enrolled} species ({total_clips_embedded} audio clips processed)")
    print(f"  TARGET DB      : {args.db}")
    print("=" * 72)


if __name__ == "__main__":
    main()
