from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.embeddings.perch import PerchBackbone
from anycall.embeddings.panns import PannsBackbone
from pathlib import Path
import numpy as np

wav = list(Path("data/processed/corvus_splendens").glob("*.wav"))[0]
print("Test WAV:", wav.name)

print("Loading BirdNET...")
bn = BirdNetBackbone()
e = bn.embed(wav)
print(f"  BirdNET: dim={len(e)}, norm_err={abs(float(np.linalg.norm(e))-1.0):.2e}")

print("Loading Perch...")
p = PerchBackbone()
e = p.embed(wav)
print(f"  Perch:   dim={len(e)}, norm_err={abs(float(np.linalg.norm(e))-1.0):.2e}")

print("Loading PANNs...")
pa = PannsBackbone()
e = pa.embed(wav)
print(f"  PANNs:   dim={len(e)}, norm_err={abs(float(np.linalg.norm(e))-1.0):.2e}")

print("All 3 backbones OK.")
