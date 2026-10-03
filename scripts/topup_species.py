import argparse
from pathlib import Path
import json
import soundfile as sf
import sys

from anycall.data.species import SPECIES_CATALOG
from anycall.data.harvester import XenoCantoHarvester
from anycall.audio.standardize import standardize_audio
from anycall.audio.vad import slice_audio_segments

# Re-inject the original topup_species.py logic, but fixing the API key and query.
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TARGET = 50
DL_LIMIT = 100

harvester = XenoCantoHarvester(
    raw_dir=RAW_DIR,
    rate_limit=1.2,
    api_key="9111e7929e7ba9ff7423561c9435167244de2ce5",
    force_fallback=False,
)

def process_raws(raw_dir: Path, processed_dir: Path, use_vad: bool = True) -> int:
    processed_dir.mkdir(parents=True, exist_ok=True)
    existing = set(p.name for p in processed_dir.glob("*.wav"))
    n_new = 0
    for raw_path in sorted(raw_dir.glob("*.mp3")):
        if len(list(processed_dir.glob("*.wav"))) >= TARGET:
            break
        try:
            waveform, sr = standardize_audio(raw_path)
            segments = slice_audio_segments(waveform, sr=sr, vad_filter=use_vad)
            for i, seg in enumerate(segments):
                out_name = f"{raw_path.stem}_seg{i:03d}.wav"
                if out_name not in existing:
                    sf.write(str(processed_dir / out_name), seg, sr, subtype="PCM_16")
                    existing.add(out_name)
                    n_new += 1
        except Exception:
            continue
    return n_new

print(f"[TopUp] Checking all {len(SPECIES_CATALOG)} species for target {TARGET} clips...")

low_species = []
for sp_id, sp_record in SPECIES_CATALOG.items():
    proc_dir = PROCESSED_DIR / sp_id
    existing_n = len(list(proc_dir.glob("*.wav"))) if proc_dir.exists() else 0
    if existing_n < TARGET:
        low_species.append((sp_id, sp_record, existing_n))

print(f"[TopUp] {len(low_species)} species below target ({TARGET} segments).\n")

for sp_id, sp_record, existing_n in low_species:
    print(f"  [{existing_n:2d}] {sp_record.common_name} ({sp_record.scientific_name})")
    proc_dir = PROCESSED_DIR / sp_id
    
    # FIX: Wrap search_query in sp:"..." for API v3 compliance!
    try:
        summary = harvester.harvest_species(sp_record, limit=DL_LIMIT, country=None, force=False)
        raw_files = summary.valid_file_paths
        print(f"       Got {len(raw_files)} raw files")
    except Exception as exc:
        print(f"       [ERROR] {exc}")
        continue

    raw_dir = RAW_DIR / sp_id
    n_new = process_raws(raw_dir, proc_dir, use_vad=True)
    total = len(list(proc_dir.glob("*.wav")))
    print(f"       +{n_new} (VAD on)  -> {total} total")

    if total < TARGET:
        n_new2 = process_raws(raw_dir, proc_dir, use_vad=False)
        total = len(list(proc_dir.glob("*.wav")))
        print(f"       +{n_new2} (VAD off) -> {total} total")

    print(f"       [{'OK' if total >= TARGET else 'STILL LOW'}]")

print("\n[TopUp] Done.")
