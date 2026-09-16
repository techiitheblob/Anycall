from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pathlib import Path

analyzer = Analyzer()
sunbird_wavs = list(Path("data/processed/cinnyris_asiaticus").glob("*.wav"))[:3]

for w in sunbird_wavs:
    print(f"\nTesting file: {w.name}")
    # With location (India, Delhi: lat 28.6, lon 77.2)
    rec_loc = Recording(analyzer, str(w), lat=28.6139, lon=77.2090, min_conf=0.01)
    rec_loc.analyze()
    print("  With Location Filter (Delhi, India):")
    for d in rec_loc.detections[:5]:
        print(f"    {d['common_name']} ({d['scientific_name']}): conf={d['confidence']:.3f}")
    
    # Without location
    rec_raw = Recording(analyzer, str(w), min_conf=0.01)
    rec_raw.analyze()
    print("  Without Location Filter:")
    for d in rec_raw.detections[:5]:
        print(f"    {d['common_name']} ({d['scientific_name']}): conf={d['confidence']:.3f}")
