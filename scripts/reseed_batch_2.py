import urllib.request
import re
import os
import shutil
import soundfile as sf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anycall.audio.standardize import standardize_audio, slice_audio_segments

species = {
    "psittacula_krameri": "Psittacula krameri",
    "centropus_sinensis": "Centropus sinensis",
    "milvus_migrans": "Milvus migrans"
}

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

def get_clean_ids(sci_name):
    query = sci_name.replace(" ", "+") + "+q%3AA"
    url = f"https://xeno-canto.org/explore?query={query}"
    try:
        html = urllib.request.urlopen(url).read().decode('utf-8')
        links = re.findall(r'href=.([^>]+download)', html)
        ids = [l.split("/")[-2] for l in links]
        return list(dict.fromkeys(ids))[:10]
    except Exception as e:
        print(f"Failed to fetch clean IDs for {sci_name}: {e}")
        return []

for sp_id, sci_name in species.items():
    print(f"Cleaning {sp_id} ({sci_name})...")
    # Clean up old
    sp_proc = PROCESSED / sp_id
    if sp_proc.exists():
        shutil.rmtree(sp_proc)
    sp_proc.mkdir(parents=True, exist_ok=True)
    
    sp_raw = RAW / sp_id
    if sp_raw.exists():
        shutil.rmtree(sp_raw)
    sp_raw.mkdir(parents=True, exist_ok=True)
    
    ids = get_clean_ids(sci_name)
    print(f"Found {len(ids)} clean IDs for {sp_id}: {ids}")
    
    count = 0
    for i, xc_id in enumerate(ids):
        try:
            download_url = f"https://xeno-canto.org/{xc_id}/download"
            req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            raw_path = sp_raw / f"{xc_id}.mp3"
            with open(raw_path, 'wb') as f:
                f.write(urllib.request.urlopen(req).read())
            
            w, sr = standardize_audio(raw_path)
            # VAD filter disabled to allow short impulse calls (e.g., parakeet screeches, coucal oops, kite whistles)
            segs = slice_audio_segments(w, sr=sr, vad_filter=False)
            for j, s in enumerate(segs):
                if j >= 10:  # limit to first 10 slices (30s) per recording to avoid excessive silence
                    break
                out = sp_proc / f"{xc_id}_seg{j:03d}.wav"
                sf.write(str(out), s, sr, subtype="PCM_16")
                count += 1
        except Exception as e:
            print(f"Failed {xc_id}: {e}")
            
    print(f"Added {count} clean segments for {sp_id}.")
