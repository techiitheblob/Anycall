import subprocess
import shutil
import sqlite3
import numpy as np
from pathlib import Path
import sys
import soundfile as sf

# Load AnyCall dependencies
from anycall.embeddings import get_backbone
from anycall.storage.db import DatabaseManager
from anycall.classifier.engine import PrototypicalClassifier
from anycall.data.harvester import XenoCantoHarvester
from anycall.data.species import SpeciesRecord, TaxonGroup
from anycall.audio.standardize import standardize_audio, slice_audio_segments

def load_species(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    names = [name.strip() for name in content.replace('\n', ',').split(',') if name.strip()]
    return list(set(names)) # unique

def get_enrolled_species(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT species_id FROM species")
    enrolled = {row[0] for row in c.fetchall()}
    conn.close()
    return enrolled

import urllib.request
import re

def get_clean_ids(sci_name):
    query = sci_name.replace(" ", "+") + "+q%3AA"
    url = f"https://xeno-canto.org/explore?query={query}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
        links = re.findall(r'href=.([^>]+download)', html)
        ids = [l.split("/")[-2] for l in links]
        return list(dict.fromkeys(ids))[:15]
    except Exception as e:
        print(f"Failed to fetch clean IDs for {sci_name}: {e}")
        return []

def main():
    atlas_species = load_species('ahmedabad_atlas_species.txt')
    print(f"Loaded {len(atlas_species)} species from Atlas.")
    
    db_mgr = DatabaseManager('anycall.db')
    enrolled = get_enrolled_species('anycall.db')
    
    print("[INIT] Loading BirdNET backbone...")
    backbone = get_backbone('birdnet')
    
    processed_count = 0
    skipped_count = 0
    
    for scientific_name in atlas_species:
        sp_id = scientific_name.lower().replace(' ', '_')
        if sp_id in enrolled:
            print(f"Skipping {scientific_name} (already enrolled).")
            skipped_count += 1
            continue
            
        print(f"\\n{'='*50}\\nProcessing: {scientific_name}\\n{'='*50}")
        
        record = SpeciesRecord(
            species_id=sp_id,
            common_name=scientific_name,
            scientific_name=scientific_name,
            taxon=TaxonGroup.AVES,
            vocalization_band_hz=(500, 8000),
            characteristic_features="N/A",
            search_query=scientific_name,
            target_recordings=15
        )
        
        # 1. Harvest audio
        print("  -> Harvesting from Xeno-Canto...")
        raw_dir = Path('data/raw') / sp_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        ids = get_clean_ids(scientific_name)
        if not ids:
            print(f"  [ERROR] No IDs found for {scientific_name}")
            continue
            
        for xc_id in ids:
            try:
                download_url = f"https://xeno-canto.org/{xc_id}/download"
                req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
                raw_path = raw_dir / f"{xc_id}.mp3"
                if not raw_path.exists():
                    with open(raw_path, 'wb') as f:
                        f.write(urllib.request.urlopen(req, timeout=15).read())
            except Exception as e:
                print(f"  [ERROR] Failed to download {xc_id}: {e}")
            
        # 2. Standardize audio
        print("  -> Standardizing to 48kHz WAV (--no-vad)...")
        processed_dir = Path('data/processed') / sp_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        if not raw_dir.exists():
            continue
            
        for raw_path in raw_dir.glob("*.mp3"):
            try:
                w, sr = standardize_audio(raw_path)
                # Ensure vad_filter=False as discussed
                segs = slice_audio_segments(w, sr=sr, vad_filter=False)
                for j, s in enumerate(segs):
                    if j >= 10: break # Avoid massive silence bloat
                    out = processed_dir / f"{raw_path.stem}_seg{j:03d}.wav"
                    sf.write(str(out), s, sr, subtype="PCM_16")
            except Exception as e:
                pass
                
        # 3. Extract embeddings and enroll
        if not processed_dir.exists() or not any(processed_dir.iterdir()):
            print(f"  [WARNING] No processed audio found for {scientific_name}")
            continue
            
        print("  -> Extracting embeddings...")
        embeddings = []
        for wav_path in processed_dir.glob("*.wav"):
            try:
                emb = backbone.extract_embedding(str(wav_path))
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                pass
                
        if not embeddings:
            print(f"  [WARNING] No valid embeddings extracted for {scientific_name}")
            continue
            
        # Compute prototype
        classifier = PrototypicalClassifier()
        centroid = classifier.compute_prototype(embeddings)
        
        # Save to database
        db_mgr.save_prototype(
            species_id=sp_id,
            common_name=scientific_name, # We use scientific as common fallback if not available
            taxon="Aves",
            prototype=centroid,
            radius=0.15,
            sample_count=len(embeddings)
        )
        print(f"  [SUCCESS] Enrolled {scientific_name} with {len(embeddings)} segments.")
        processed_count += 1
        
        # 4. Clean up disk space
        print("  -> Cleaning up disk space...")
        raw_dir = Path('data/raw') / sp_id
        if raw_dir.exists(): shutil.rmtree(raw_dir)
        if processed_dir.exists(): shutil.rmtree(processed_dir)
        
    print(f"\\nDONE! Processed: {processed_count} | Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
