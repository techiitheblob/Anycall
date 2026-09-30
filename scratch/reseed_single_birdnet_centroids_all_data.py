"""Reseed anycall.db with single L2-normalized BirdNET mean centroids using all newly harvested audio data."""
import sqlite3
import numpy as np
import librosa
from pathlib import Path
from anycall.embeddings import get_backbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "anycall.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def hpss_extract_clips_birdnet(audio_path, target_sr=48000, clip_duration=3.0):
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        if len(y) < target_sr * 1.0:
            return []
        y_harmonic, _ = librosa.effects.hpss(y)
        clip_samples = int(target_sr * clip_duration)
        clips = []
        hop_len = clip_samples // 2
        for start in range(0, len(y_harmonic) - clip_samples + 1, hop_len):
            chunk = y_harmonic[start:start + clip_samples]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.005:
                clips.append((rms, chunk))
        clips.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in clips[:4]]
    except Exception:
        return []

def main():
    print("=== RESEEDING ALL 53 SPECIES WITH SINGLE BIRDNET CENTROIDS (ALL HARVESTED DATA) ===")
    backbone = get_backbone("birdnet")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT species_id, common_name, taxon FROM species")
    species_rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(species_rows)} enrolled species in database.")
    reseeded = 0
    
    for sp_id, c_name, taxon in species_rows:
        sp_dir = PROCESSED_DIR / sp_id
        if not sp_dir.exists() or len(list(sp_dir.glob("*.wav"))) == 0:
            sp_dir = RAW_DIR / sp_id
            
        if not sp_dir.exists():
            print(f"[{sp_id}] Directory not found, skipping...")
            continue
            
        audio_files = sorted(list(sp_dir.glob("*.wav")) + list(sp_dir.glob("*.mp3")))
        if not audio_files:
            continue
            
        # Enrollment files (indices 0..3 for low-sample species, or up to index 30 for harvested species)
        # Preserve held-out test clips (indices 3:8) when audio_files > 3
        if len(audio_files) > 8:
            # Use clips [0..3] + [8..30] so held-out clips [3..8] remain strictly un-trained
            enrollment_files = audio_files[:3] + audio_files[8:30]
        else:
            enrollment_files = audio_files[:3]
            
        embeddings = []
        for af in enrollment_files:
            clips = hpss_extract_clips_birdnet(af)
            for clip in clips:
                try:
                    emb = backbone.embed(clip, sr=48000)
                    if len(emb) == 1024:
                        embeddings.append(emb)
                except Exception:
                    pass
                    
        if not embeddings:
            print(f"[{sp_id}] No valid embeddings extracted!")
            continue
            
        embeddings = np.array(embeddings, dtype=np.float32)
        mean_c = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_c)
        centroid = (mean_c / norm) if norm > 1e-12 else mean_c
        
        # Store single 1024-dim BirdNET centroid in anycall.db
        proto_bytes = centroid.astype(np.float32).tobytes()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE species SET prototype = ?, sample_count = ? WHERE species_id = ?",
            (proto_bytes, len(embeddings), sp_id)
        )
        conn.commit()
        conn.close()
        reseeded += 1
        print(f"[{sp_id}] Reseeded single BirdNET centroid (dim: 1024) from {len(embeddings)} clips ({len(enrollment_files)} files)")
        
    print(f"\nSUCCESS: Reseeded {reseeded} species with single BirdNET mean centroids!")

if __name__ == "__main__":
    main()
