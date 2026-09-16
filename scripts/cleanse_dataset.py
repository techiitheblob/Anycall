"""
Script to purge contaminated copy-pasted files and replace with verified audio:
1. Replaces cinnyris_asiaticus with clean verified recordings (XC149562, XC685137, XC74154).
2. Replaces pycnonotus_cafer with clean verified recording (XC129368).
3. Removes duplicated recording IDs across species so each species has genuine data.
"""
import shutil
from pathlib import Path
import soundfile as sf
from anycall.audio.standardize import standardize_audio, slice_audio_segments

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

# 1. Clean Purple Sunbird (cinnyris_asiaticus)
sunbird_proc = PROCESSED / "cinnyris_asiaticus"
if sunbird_proc.exists():
    shutil.rmtree(sunbird_proc)
sunbird_proc.mkdir(parents=True, exist_ok=True)

sunbird_raws = list(Path("data/raw/real_sunbird").glob("*.mp3"))
sunbird_count = 0
for r in sunbird_raws:
    w, sr = standardize_audio(r)
    segs = slice_audio_segments(w, sr=sr, vad_filter=True)
    for i, s in enumerate(segs):
        out = sunbird_proc / f"{r.stem}_seg{i:03d}.wav"
        sf.write(str(out), s, sr, subtype="PCM_16")
        sunbird_count += 1
print(f"Cleaned cinnyris_asiaticus: {sunbird_count} verified segments.")

# 2. Clean Red-vented Bulbul (pycnonotus_cafer)
bulbul_proc = PROCESSED / "pycnonotus_cafer"
if bulbul_proc.exists():
    shutil.rmtree(bulbul_proc)
bulbul_proc.mkdir(parents=True, exist_ok=True)

bulbul_raw = Path("data/raw/bulbul_129368.mp3")
bulbul_count = 0
if bulbul_raw.exists():
    w, sr = standardize_audio(bulbul_raw)
    segs = slice_audio_segments(w, sr=sr, vad_filter=True)
    for i, s in enumerate(segs):
        out = bulbul_proc / f"{bulbul_raw.stem}_seg{i:03d}.wav"
        sf.write(str(out), s, sr, subtype="PCM_16")
        bulbul_count += 1
print(f"Cleaned pycnonotus_cafer: {bulbul_count} verified segments.")

# 3. Purge duplicate crow clips that were wrongly placed in other bird folders
# Specifically: 1094670, 744704, 683047, 604023, 796825, 826276, 913509
crow_polluted = ["1094670", "744704", "683047", "604023", "796825", "826276", "913509"]
for sp_dir in PROCESSED.iterdir():
    if not sp_dir.is_dir() or sp_dir.name in ["corvus_splendens", "corvus_culminatus"]:
        continue
    removed = 0
    for wav in list(sp_dir.glob("*.wav")):
        if any(bad_id in wav.name for bad_id in crow_polluted):
            wav.unlink()
            removed += 1
    if removed > 0:
        print(f"Removed {removed} crow-contaminated files from {sp_dir.name}")

print("\nData cleansing complete.")
