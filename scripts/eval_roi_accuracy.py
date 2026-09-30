import logging
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.stream.spectrogram import SpectrogramEventDetector

logging.basicConfig(level=logging.ERROR)

def main():
    print("=" * 60)
    print("  Evaluating Adaptive Spectrogram ROI Accuracy")
    print("=" * 60)
    
    # 1. Setup classifier and backbone
    backbone = BirdNetBackbone()
    clf = PrototypicalClassifier(threshold=0.5, n_subprototypes=3)
    
    # 2. Enroll using processed files (the standard 5-shots)
    print("Enrolling 5 shots per species using standard data...")
    species_dirs = list(Path("data/processed").glob("*"))
    enrolled_species = []
    
    for sp_dir in species_dirs:
        if not sp_dir.is_dir(): continue
        sp_name = sp_dir.name
        wavs = list(sp_dir.glob("*.wav"))
        
        vecs = []
        for w in wavs[:5]:
            try:
                vecs.append(backbone.embed(str(w)))
            except:
                pass
                
        if len(vecs) == 5:
            clf.enroll(sp_name, vecs)
            enrolled_species.append(sp_name)
            
    print(f"Enrolled {len(enrolled_species)} species.")
    
    # 3. Test on RAW audio files (continuous streams)
    detector = SpectrogramEventDetector()
    
    raw_dirs = [Path("data/raw") / sp for sp in enrolled_species]
    
    total_rois = 0
    correct_rois = 0
    
    print("\\nExtracting ROIs from continuous RAW audio and classifying...")
    for sp_dir in raw_dirs:
        if not sp_dir.exists(): continue
        sp_name = sp_dir.name
        raw_files = list(sp_dir.glob("*.mp3")) + list(sp_dir.glob("*.wav"))
        
        # Test on 3 raw files per species
        for raw_file in raw_files[:3]:
            try:
                # Load continuous raw audio
                y, sr = librosa.load(str(raw_file), sr=48000)
                
                # Find distinct acoustic events (chirps)
                events = detector.find_events(y)
                
                for (start_t, end_t) in events:
                    # Dynamically pad to exactly 3 seconds centered on the event
                    event_center = (start_t + end_t) / 2.0
                    win_start = max(0.0, event_center - 1.5)
                    win_end = win_start + 3.0
                    
                    start_idx = int(win_start * 48000)
                    end_idx = int(win_end * 48000)
                    clip = y[start_idx:end_idx]
                    
                    if len(clip) < 48000 * 3:
                        clip = np.pad(clip, (0, 48000*3 - len(clip)))
                        
                    # Classify the perfectly framed ROI
                    emb = backbone.embed(clip, sr=48000)
                    res = clf.predict(emb)
                    
                    total_rois += 1
                    if res.predicted_label == sp_name:
                        correct_rois += 1
                        
            except Exception as e:
                continue
                
    if total_rois > 0:
        acc = correct_rois / total_rois
        print("\\n" + "=" * 60)
        print("  SPECTROGRAM ROI ACCURACY RESULTS")
        print("=" * 60)
        print(f"Total isolated bird calls tested : {total_rois}")
        print(f"Correctly identified             : {correct_rois}")
        print(f"Overall Accuracy                 : {acc * 100:.2f}%")
        print("=" * 60)
    else:
        print("No ROIs extracted.")

if __name__ == "__main__":
    main()
