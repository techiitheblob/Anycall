#!/usr/bin/env python3
"""
Top-up under-represented species by downloading globally (no country filter).
For species that keep producing 0 segments, turns off VAD filtering as fallback.
Replaces irreparably sparse species with alternative Indian resident species.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from anycall.data.harvester import XenoCantoHarvester
from anycall.audio.standardize import standardize_audio, slice_audio_segments

PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")
TARGET        = 15   # want at least this many segments
DL_LIMIT      = 60  # download up to 60 recordings globally

harvester = XenoCantoHarvester(
    raw_dir=RAW_DIR,
    rate_limit=1.2,
    force_fallback=True,
)

def process_raw(raw_files, processed_dir, use_vad=True, existing_names=None):
    """Process raw files into 3s segments. Returns count of new segments added."""
    if existing_names is None:
        existing_names = set(p.name for p in processed_dir.glob("*.wav"))
    n_new = 0
    for raw_path in raw_files:
        current = len(list(processed_dir.glob("*.wav")))
        if current >= TARGET:
            break
        try:
            waveform, sr = standardize_audio(raw_path)
            segments = slice_audio_segments(waveform, sr=sr, vad_filter=use_vad)
            for i, seg in enumerate(segments):
                out_path = processed_dir / f"{raw_path.stem}_seg{i:03d}.wav"
                if out_path.name not in existing_names:
                    sf.write(str(out_path), seg, sr, subtype="PCM_16")
                    existing_names.add(out_path.name)
                    n_new += 1
        except Exception as exc:
            print(f"    [WARN] {raw_path.name}: {exc}")
    return n_new

# ── Species to fill ────────────────────────────────────────────────────────
from anycall.data.species import SPECIES_CATALOG

low_species = []
for sp_id, sp in SPECIES_CATALOG.items():
    d = PROCESSED_DIR / sp_id
    n = len(list(d.glob("*.wav"))) if d.exists() else 0
    if n < TARGET:
        low_species.append((n, sp_id, sp))
low_species.sort()

print(f"[TopUp] {len(low_species)} species below target ({TARGET} segments).\n")

still_low = []

for existing_n, sp_id, sp in low_species:
    print(f"  [{existing_n:2d}] {sp.common_name} ({sp.scientific_name})")
    d = PROCESSED_DIR / sp_id
    d.mkdir(parents=True, exist_ok=True)

    # Download globally, force fresh to get different recordings
    try:
        summary = harvester.harvest_species(sp, limit=DL_LIMIT, country=None, force=True)
    except Exception as exc:
        print(f"       [ERROR] {exc}")
        still_low.append(sp_id)
        continue

    raw_files = summary.valid_file_paths
    print(f"       Got {len(raw_files)} raw files")

    # First pass: with VAD
    n_new = process_raw(raw_files, d, use_vad=True)
    total = len(list(d.glob("*.wav")))
    print(f"       +{n_new} (VAD on)  -> {total} total")

    # Second pass: without VAD if still low
    if total < TARGET:
        n_new2 = process_raw(raw_files, d, use_vad=False)
        total = len(list(d.glob("*.wav")))
        print(f"       +{n_new2} (VAD off) -> {total} total")

    if total < TARGET:
        still_low.append(sp_id)
        print(f"       [STILL LOW]")
    else:
        print(f"       [OK]")

print(f"\n[TopUp] Done.")
if still_low:
    print(f"  Still low ({len(still_low)}): {still_low}")
    print(f"  These will be replaced with alternative species.")
else:
    print(f"  All species now have >= {TARGET} segments!")
