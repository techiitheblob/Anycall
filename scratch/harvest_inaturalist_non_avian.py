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

INAT_API_URL = "https://api.inaturalist.org/v1/observations"

NON_AVIAN_SPECIES = [
    ("duttaphrynus_melanostictus", "Common Indian Toad", "Amphibia", "Duttaphrynus melanostictus"),
    ("euphlyctis_cyanophlyctis", "Skittering Frog", "Amphibia", "Euphlyctis cyanophlyctis"),
    ("hydrophylax_bahuvistara", "Fungoid Frog", "Amphibia", "Hydrophylax bahuvistara"),
    ("hoplobatrachus_tigerinus", "Indian Bullfrog", "Amphibia", "Hoplobatrachus tigerinus"),
    ("polypedates_maculatus", "Indian Tree Frog", "Amphibia", "Polypedates maculatus"),
    ("macaca_mulatta", "Rhesus Macaque", "Mammalia", "Macaca mulatta"),
    ("semnopithecus_entellus", "Hanuman Langur", "Mammalia", "Semnopithecus entellus"),
    ("funambulus_palmarum", "Indian Palm Squirrel", "Mammalia", "Funambulus palmarum"),
    ("muntiacus_muntjak", "Barking Deer", "Mammalia", "Muntiacus muntjak"),
    ("canis_aureus", "Golden Jackal", "Mammalia", "Canis aureus"),
    ("oecanthus_indicus", "Tree Cricket", "Insecta", "Oecanthus indicus"),
    ("gryllotalpa_orientalis", "Mole Cricket", "Insecta", "Gryllotalpa orientalis"),
    ("teleogryllus_occipitalis", "Asian Field Cricket", "Insecta", "Teleogryllus occipitalis"),
    ("gryllus_bimaculatus", "Two-spotted Cricket", "Insecta", "Gryllus bimaculatus"),
    ("mecopoda_elongata", "Short-winged Katydid", "Insecta", "Mecopoda elongata")
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
            if rms > 0.003:
                clips.append((rms, chunk))
        clips.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in clips[:4]]
    except Exception:
        return []

def harvest_inaturalist_species(sp_id, c_name, taxon_name, max_recs=20):
    raw_dir = PROJECT_ROOT / "data" / "raw" / sp_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "AnyCallHarvester/1.0 (+https://github.com/anycall/anycall)"}
    params = {"sounds": "true", "taxon_name": taxon_name, "per_page": max_recs}
    
    print(f"[{sp_id}] Fetching iNaturalist sound observations for {c_name}...")
    try:
        resp = requests.get(INAT_API_URL, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        
        idx = 0
        for obs in results:
            obs_id = obs.get("id")
            for sound in obs.get("sounds", []):
                file_url = sound.get("file_url") or sound.get("audio_url")
                if not file_url:
                    continue
                ext = ".mp3"
                if ".m4a" in file_url.lower(): ext = ".m4a"
                elif ".wav" in file_url.lower(): ext = ".wav"
                
                out_file = raw_dir / f"inat_{obs_id}_{idx}{ext}"
                idx += 1
                
                if out_file.exists() and out_file.stat().st_size > 3000:
                    continue
                    
                try:
                    r_audio = requests.get(file_url, headers=headers, timeout=15)
                    if r_audio.status_code == 200 and len(r_audio.content) > 3000:
                        with open(out_file, "wb") as f:
                            f.write(r_audio.content)
                        print(f"  Downloaded {out_file.name} ({len(r_audio.content)//1024} KB)")
                    time.sleep(0.3)
                except Exception as e_dl:
                    print(f"  Failed downloading {file_url}: {e_dl}")
    except Exception as e:
        print(f"[{sp_id}] iNaturalist API query failed: {e}")
        
    return list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.m4a")) + list(raw_dir.glob("*.wav"))

def main():
    print("=== INATURALIST NON-AVIAN HARVEST & PROTOTYPE RESEED ===")
    backbone = get_backbone("birdnet")
    
    db_path = PROJECT_ROOT / "anycall.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    updated = []
    
    for sp_id, c_name, taxon, taxon_name in NON_AVIAN_SPECIES:
        files = harvest_inaturalist_species(sp_id, c_name, taxon_name, max_recs=20)
        print(f"[{sp_id}] Processing {len(files)} audio files for HPSS embedding...")
        
        # Split enrollment files (0..2) and held-out files (3..end)
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
        print(f"SUCCESS: Updated {c_name} ({sp_id}) prototype in anycall.db (K={sample_count})")
        updated.append((c_name, sp_id, sample_count))
        
    conn.close()
    
    print("\n=== SUMMARY OF UPDATED NON-AVIAN SPECIES ===")
    for c_name, sp_id, K in updated:
        print(f"  • {c_name:<30} ({sp_id}) -> New K={K}")

if __name__ == "__main__":
    main()
