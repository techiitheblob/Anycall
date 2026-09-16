from pathlib import Path
from anycall.audio.standardize import standardize_audio, slice_audio_segments
from anycall.embeddings.birdnet import BirdNetBackbone
import numpy as np

bn = BirdNetBackbone()
lbl_file = Path(r"C:\Users\kahaa\teamwork_projects\anycall\.venv\Lib\site-packages\birdnet_analyzer\labels\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
with open(lbl_file, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

sunbird_idx = [i for i, l in enumerate(labels) if "cinnyris asiaticus" in l.lower()][0]

for mp3 in Path("data/raw/real_sunbird").glob("*.mp3"):
    waveform, sr = standardize_audio(mp3)
    segs = slice_audio_segments(waveform, sr=sr, vad_filter=True)
    print(f"\n{mp3.name} ({len(segs)} segments):")
    for i, seg in enumerate(segs[:5]):
        _, logits = bn.extract_with_logits(seg, sr=sr)
        top_idx = int(np.argmax(logits))
        top_label = labels[top_idx]
        top_score = float(logits[top_idx])
        sunbird_score = float(logits[sunbird_idx])
        is_sunbird = (top_idx == sunbird_idx)
        status = "CORRECT!" if is_sunbird else "MISIDENTIFIED"
        print(f"  Seg {i}: Top Guess: {top_label:45s} ({top_score:.3f}) | Sunbird: {sunbird_score:.3f} | {status}")
