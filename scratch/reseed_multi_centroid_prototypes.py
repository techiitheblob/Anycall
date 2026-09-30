"""Reseed anycall.db with Multi-Centroid Sub-Prototypes (k=3) extracting all vocal clips from enrollment files."""
import sqlite3
import numpy as np
import librosa
from pathlib import Path
from sklearn.cluster import KMeans
from anycall.embeddings import get_backbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "anycall.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def hpss_extract_all_clips(audio_path, target_sr=48000, clip_duration=3.0):
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
            if rms > 0.003: # Sensitive energy threshold
                clips.append((rms, chunk))
        clips.sort(key=lambda x: x[0], reverse=True)
        # Return up to top 10 clips per enrollment audio file
        return [c[1] for c in clips[:10]]
    except Exception:
        return []

def main():
    print("=== ENRICHED MULTI-CENTROID SUB-PROTOTYPE RESEEDING (k=3) ===")
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
            
        # Use enrollment files strictly (indices 0..2) to preserve held-out files (indices 3+)
        enrollment_files = audio_files[:3]
        
        embeddings = []
        for af in enrollment_files:
            clips = hpss_extract_all_clips(af)
            for clip in clips:
                try:
                    emb = backbone.embed(clip, sr=48000)
                    embeddings.append(emb)
                except Exception:
                    pass
                    
        if not embeddings:
            print(f"[{sp_id}] No valid embeddings extracted!")
            continue
            
        embeddings = np.array(embeddings, dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        embeddings = embeddings / norms
        
        # Sub-clustering logic
        n_samples = len(embeddings)
        if n_samples >= 3:
            k = min(3, n_samples)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            centroids = kmeans.cluster_centers_
            # L2-normalize centroids
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            c_norms[c_norms < 1e-12] = 1.0
            centroids = centroids / c_norms
        else:
            mean_c = np.mean(embeddings, axis=0, keepdims=True)
            norm = np.linalg.norm(mean_c)
            centroids = mean_c / norm if norm > 1e-12 else mean_c
            
        # Store in anycall.db
        proto_bytes = centroids.astype(np.float32).tobytes()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE species SET prototype = ?, sample_count = ? WHERE species_id = ?",
            (proto_bytes, n_samples, sp_id)
        )
        conn.commit()
        conn.close()
        reseeded += 1
        print(f"[{sp_id}] Reseeded {centroids.shape[0]} sub-centroid(s) from {n_samples} clips (shape: {centroids.shape})")
        
    print(f"\nSUCCESS: Reseeded {reseeded} species with multi-centroid sub-prototypes!")

if __name__ == "__main__":
    main()
