import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from anycall.stream.spectrogram import SpectrogramEventDetector

def reprocess_raw_with_spectrogram():
    print("=" * 60)
    print("  Re-Enrolling Database using Spectrogram ROIs")
    print("=" * 60)
    
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed_roi")
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    
    from anycall.data.species import SPECIES_CATALOG
    
    detector = SpectrogramEventDetector(energy_threshold=0.2)
    
    target_species = set(SPECIES_CATALOG.keys())
    species_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name in target_species]
    print(f"Found {len(species_dirs)} valid curated species in raw data.")
    
    for sp_dir in tqdm(species_dirs, desc="Reprocessing"):
        sp_out = out_dir / sp_dir.name
        sp_out.mkdir(exist_ok=True)
        
        raw_files = list(sp_dir.glob("*.mp3")) + list(sp_dir.glob("*.wav"))
        
        clips_saved = 0
        for rfile in raw_files:
            if clips_saved >= 30: # Max 30 clips per species
                break
                
            try:
                y, sr = librosa.load(str(rfile), sr=48000)
                events = detector.find_events(y)
                
                for (start_t, end_t) in events:
                    if clips_saved >= 30: break  # FIX: was incorrectly 5
                    
                    event_center = (start_t + end_t) / 2.0
                    win_start = max(0.0, event_center - 1.5)
                    win_end = win_start + 3.0
                    
                    start_idx = int(win_start * 48000)
                    end_idx = int(win_end * 48000)
                    clip = y[start_idx:end_idx]
                    
                    if len(clip) < 48000 * 3:
                        clip = np.pad(clip, (0, 48000*3 - len(clip)))
                        
                    # Save
                    out_path = sp_out / f"{rfile.stem}_roi_{clips_saved:03d}.wav"
                    sf.write(str(out_path), clip, 48000, subtype="PCM_16", format="WAV")
                    clips_saved += 1
            except Exception as e:
                pass

if __name__ == "__main__":
    reprocess_raw_with_spectrogram()
