import requests
from pathlib import Path
from anycall.audio.standardize import standardize_audio, slice_audio_segments
from anycall.embeddings.birdnet import BirdNetBackbone
import numpy as np

bn = BirdNetBackbone()
lbl_file = Path(r"C:\Users\kahaa\teamwork_projects\anycall\.venv\Lib\site-packages\birdnet_analyzer\labels\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
with open(lbl_file, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

test_downloads = [
    ("myna_129571", "https://xeno-canto.org/129571/download", "acridotheres tristis"),
    ("bulbul_129368", "https://xeno-canto.org/129368/download", "pycnonotus cafer"),
]

for name, url, sci in test_downloads:
    p = Path(f"data/raw/{name}.mp3")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    print(f"\nDownloading {name}: HTTP {r.status_code}, {len(r.content)} bytes")
    if r.status_code == 200:
        p.write_bytes(r.content)
        waveform, sr = standardize_audio(p)
        segs = slice_audio_segments(waveform, sr=sr, vad_filter=True)
        target_idx = [i for i, l in enumerate(labels) if sci in l.lower()][0]
        print(f"  {name} sliced into {len(segs)} segments. Target: {labels[target_idx]}")
        correct = 0
        for i, s in enumerate(segs[:5]):
            _, logits = bn.extract_with_logits(s, sr=sr)
            top_i = int(np.argmax(logits))
            top_g = labels[top_i]
            is_c = (top_i == target_idx)
            if is_c:
                correct += 1
            status = "CORRECT!" if is_c else "MISIDENTIFIED"
            print(f"    Seg {i}: {top_g} | {status}")
        print(f"  Accuracy on {name}: {correct}/{min(len(segs), 5)}")
