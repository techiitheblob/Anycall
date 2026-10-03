import os
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
from sklearn.cluster import DBSCAN

base_dir = Path("C:/Users/kahaa/teamwork_projects/anycall")
sys.path.insert(0, str(base_dir))

from anycall.embeddings.birdnet import BirdNetBackbone

def clean_isolated_outliers(eps_threshold=0.35, min_cluster_size=2):
    """
    Finds and moves isolated embeddings to an 'outliers' folder.
    Uses DBSCAN: if a point has no other points within `eps_threshold` cosine distance,
    it is marked as noise (-1). If it has at least one neighbor (cluster size >= 2), it is kept!
    """
    bb = BirdNetBackbone(offline_fallback=True)
    proc_dir = base_dir / "data" / "processed"
    
    total_outliers = 0
    total_processed = 0

    for sp_id in os.listdir(proc_dir):
        sp_dir = proc_dir / sp_id
        if not sp_dir.is_dir(): continue
        
        wavs = list(sp_dir.glob("*.wav"))
        if len(wavs) < 30:
            continue
            
        print(f"Scanning {sp_id} ({len(wavs)} clips)...")
        embs = []
        valid_wavs = []
        for w in wavs:
            try:
                data, sr = sf.read(str(w), dtype="float32")
                e, _ = bb.extract_with_logits(data, sr=sr)
                embs.append(e)
                valid_wavs.append(w)
            except: pass
            
        if not embs: continue
        
        X = np.stack(embs)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.where(norms > 1e-12, norms, 1.0)
        
        # DBSCAN perfectly implements the user's logic!
        # metric='cosine', min_samples=2 means a point must have at least 1 neighbor within eps to be kept.
        db = DBSCAN(eps=eps_threshold, min_samples=min_cluster_size, metric='cosine')
        labels = db.fit_predict(X_norm)
        
        outliers = [valid_wavs[i] for i, l in enumerate(labels) if l == -1]
        
        if outliers:
            outlier_dir = sp_dir / "outliers"
            outlier_dir.mkdir(exist_ok=True)
            for o in outliers:
                shutil.move(str(o), str(outlier_dir / o.name))
            print(f"  -> Found {len(outliers)} isolated anomalies. Moved to {outlier_dir.name}/ (Kept {len(wavs) - len(outliers)} valid clips)")
            total_outliers += len(outliers)
        else:
            print(f"  -> All clips are part of valid clusters. No outliers found.")
            
        total_processed += 1

    print(f"\nCleanup Complete! Processed {total_processed} species, safely quarantined {total_outliers} isolated files.")
    print("These can be instantly restored by moving them back from the 'outliers' folders.")

if __name__ == "__main__":
    clean_isolated_outliers()
