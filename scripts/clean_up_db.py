import os
import shutil
import sqlite3
import numpy as np
import soundfile as sf
from pathlib import Path

# 1. Delete corrupted directories
proc_dir = Path("data/processed")
for bad_dir in ["acridotheres_tristis_corrupted", "halcyon_smyrnensis_corrupted"]:
    path = proc_dir / bad_dir
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted {path}")

# 2. Delete from DB
db_path = "anycall.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("DELETE FROM species WHERE species_id LIKE '%corrupted%'")
conn.commit()
print(f"Deleted corrupted species from DB. Rows affected: {c.rowcount}")
conn.close()

# 3. Filter silence from centropus_sinensis
coucal_dir = proc_dir / "centropus_sinensis"
if coucal_dir.exists():
    wavs = list(coucal_dir.glob("*.wav"))
    energies = []
    
    # Calculate RMS energy for all clips
    for w in wavs:
        audio, sr = sf.read(str(w))
        rms = np.sqrt(np.mean(audio**2))
        energies.append((rms, w))
    
    # Sort by energy
    energies.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 40%, delete the bottom 60% (which are likely pure background/silence)
    keep_count = max(int(len(energies) * 0.40), 1)
    to_delete = energies[keep_count:]
    
    for rms, w in to_delete:
        w.unlink()
    
    print(f"Coucal: Kept {keep_count} high-energy clips, deleted {len(to_delete)} silent/noise clips.")
