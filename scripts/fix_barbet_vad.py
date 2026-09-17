import shutil
import soundfile as sf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anycall.audio.standardize import standardize_audio, slice_audio_segments

sp_id = "psilopogon_haemacephalus"
raw_dir = Path("data/raw") / sp_id
proc_dir = Path("data/processed") / sp_id

if proc_dir.exists():
    shutil.rmtree(proc_dir)
proc_dir.mkdir(parents=True, exist_ok=True)

count = 0
for raw_path in raw_dir.glob("*.mp3"):
    try:
        w, sr = standardize_audio(raw_path)
        # DISABLE VAD for Barbet because calls are < 300ms pulses
        segs = slice_audio_segments(w, sr=sr, vad_filter=False)
        for j, s in enumerate(segs):
            # Limit to 20 segments per file to not explode DB
            if j >= 20: 
                break
            out = proc_dir / f"{raw_path.stem}_seg{j:03d}.wav"
            sf.write(str(out), s, sr, subtype="PCM_16")
            count += 1
    except Exception as e:
        print(f"Failed {raw_path.name}: {e}")

print(f"Added {count} clean segments for {sp_id} (VAD Disabled).")
