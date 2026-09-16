#!/usr/bin/env python3
"""Download and process all 33 Indian species from Xeno-Canto.

Runs the harvester + audio standardization pipeline for every species
in the catalog. Skips species that already have >= MIN_SEGMENTS processed WAV files.

Usage:
    python scripts/download_all_species.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from anycall.data.harvester import XenoCantoHarvester
from anycall.data.species import SPECIES_CATALOG
from anycall.audio.standardize import standardize_audio, slice_audio_segments

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
TARGET_PER_SPECIES = 25
MIN_SEGMENTS = 10  # skip species that already have this many processed WAVs

harvester = XenoCantoHarvester(
    raw_dir=RAW_DIR,
    rate_limit=1.5,       # polite: 1.5s between requests
    force_fallback=True,  # unauthenticated download (no API key needed)
)

# SPECIES_CATALOG is {species_id: SpeciesRecord}
all_species = list(SPECIES_CATALOG.values())

print(f"[Downloader] {len(all_species)} species in catalog.")
print(f"[Downloader] Target: {TARGET_PER_SPECIES} recordings/species, skip if >= {MIN_SEGMENTS} WAVs.\n")

total_new_segments = 0
failed_species = []

for sp in all_species:
    processed_sp_dir = PROCESSED_DIR / sp.species_id
    existing = list(processed_sp_dir.glob("*.wav")) if processed_sp_dir.exists() else []

    if len(existing) >= MIN_SEGMENTS:
        print(f"  [SKIP] {sp.species_id:40s} already has {len(existing)} WAVs")
        continue

    print(f"  [>>>]  {sp.common_name} ({sp.scientific_name})")

    # 1. Download raw recordings
    try:
        summary = harvester.harvest_species(
            sp,
            limit=TARGET_PER_SPECIES,
            country="india",
        )
    except Exception as exc:
        print(f"         [ERROR] Harvest failed: {exc}")
        failed_species.append((sp.species_id, str(exc)))
        continue

    raw_files = summary.valid_file_paths
    print(f"         Downloaded {len(raw_files)} raw recordings")

    if not raw_files:
        print(f"         [WARN] No raw files, skipping processing.")
        failed_species.append((sp.species_id, "no raw files downloaded"))
        continue

    # 2. Standardize + segment each raw file
    processed_sp_dir.mkdir(parents=True, exist_ok=True)
    n_segments = 0

    for raw_path in raw_files:
        try:
            waveform, sr = standardize_audio(raw_path)
            segments = slice_audio_segments(waveform, sr=sr, vad_filter=True)
            for i, seg in enumerate(segments):
                out_path = processed_sp_dir / f"{raw_path.stem}_seg{i:03d}.wav"
                sf.write(str(out_path), seg, sr, subtype="PCM_16")
                n_segments += 1
        except Exception as exc:
            print(f"         [WARN] Processing failed for {raw_path.name}: {exc}")
            continue

    total_new_segments += n_segments
    print(f"         Produced {n_segments} segments  ->  {processed_sp_dir.relative_to('.')}")

print(f"\n[Downloader] Finished.")
print(f"  Total new segments produced : {total_new_segments}")
print(f"  Failed species              : {len(failed_species)}")
for sid, reason in failed_species:
    print(f"    - {sid}: {reason}")
