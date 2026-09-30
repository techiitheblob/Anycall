import os
import sys
import time
import sqlite3
import numpy as np
import requests
import librosa
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anycall.embeddings import get_backbone

API_KEY = "9111e7929e7ba9ff7423561c9435167244de2ce5"
XC_API_URL = "https://xeno-canto.org/api/3/recordings"

BIRDS_TO_HARVEST = [
    ("psittacula_krameri", "Rose-ringed Parakeet", "Aves", "gen:Psittacula sp:krameri"),
    ("pavo_cristatus", "Indian Peafowl", "Aves", "gen:Pavo sp:cristatus"),
    ("milvus_migrans", "Black Kite", "Aves", "gen:Milvus sp:migrans"),
    ("halcyon_smyrnensis", "White-throated Kingfisher", "Aves", "gen:Halcyon sp:smyrnensis"),
    ("cinnyris_asiaticus", "Purple Sunbird", "Aves", "gen:Cinnyris sp:asiaticus"),
    ("pycnonotus_cafer", "Red-vented Bulbul", "Aves", "gen:Pycnonotus sp:cafer"),
    ("athene_brama", "Spotted Owlet", "Aves", "gen:Athene sp:brama")
]

def hpss_extract_clips(audio_path, target_sr=48000, clip_duration=3.0):
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

def harvest_species(sp_id, c_name, query, max_recs=25):
    raw_dir = PROJECT_ROOT / "data" / "raw" / sp_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))
    if len(existing) >= max_recs:
        return existing
        
    print(f"[{sp_id}] Harvesting up to {max_recs} recordings from Xeno-Canto...")
    params = {"query": query, "key": API_KEY, "page": 1}
    headers = {"User-Agent": "AnyCallHarvester/1.0"}
    
    try:
        resp = requests.get(XC_API_URL, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("recordings", [])[:max_recs]
        
        for r in recs:
            rec_id = r.get("id")
            dl_url = r.get("file")
            if not dl_url:
                dl_url = f"https://xeno-canto.org/{rec_id}/download"
                
            out_file = raw_dir / f"xc_{rec_id}.mp3"
            if out_file.exists() and out_file.stat().st_size > 5000:
                continue
                
            try:
                r_dl = requests.get(dl_url, headers=headers, timeout=15)
                if r_dl.status_code == 200 and len(r_dl.content) > 5000:
                    with open(out_file, "wb") as f:
                        f.write(r_dl.content)
                    print(f"  Downloaded {out_file.name} ({len(r_dl.content)//1024} KB)")
                time.sleep(0.3)
            except Exception as e_dl:
                print(f"  Failed downloading {rec_id}: {e_dl}")
                
    except Exception as e:
        print(f"[{sp_id}] Query failed: {e}")
        
    return list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))

def main():
    print("=== HARVESTING & RESEEDING ALL 7 UNDERPERFORMING BIRD SPECIES ===")
    backbone = get_backbone("birdnet")
    
    db_path = PROJECT_ROOT / "anycall.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for sp_id, c_name, taxon, query in BIRDS_TO_HARVEST:
        files = harvest_species(sp_id, c_name, query, max_recs=25)
        print(f"[{sp_id}] Extracting HPSS clips for {len(files)} raw audio files...")
        
        # EXPLICIT HELD-OUT DATA SPLIT:
        # Clips from files 0..2 are used for Prototype Creation (Enrollment)
        # Clips from files 3..end are STRICTLY RESERVED for Blind Testing Evaluation
        enrollment_files = files[:3] if len(files) >= 3 else files
        
        all_embeddings = []
        for f in enrollment_files:
            clips = hpss_extract_clips(f)
            for clip in clips:
                try:
                    emb = backbone.embed(clip, sr=48000)
                    all_embeddings.append(emb)
                except Exception:
                    pass
                    
        if not all_embeddings:
            continue
            
        centroid = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid = centroid / norm
            
        proto_blob = centroid.astype(np.float32).tobytes()
        sample_count = len(all_embeddings)
        
        cur.execute(
            """UPDATE species 
               SET prototype = ?, sample_count = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE species_id = ?""",
            (proto_blob, sample_count, sp_id)
        )
        conn.commit()
        print(f"SUCCESS: Updated prototype centroid for {c_name} (K={sample_count} enrollment clips)")
        
    conn.close()
    print("\nAll 7 species prototypes successfully updated in anycall.db!")

if __name__ == "__main__":
    main()
