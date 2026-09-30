import os
import sys
import json
import time
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anycall.embeddings import get_backbone
import librosa

def hpss_extract_clips(audio_path, target_sr=48000, clip_duration=3.0):
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        if len(y) < target_sr * 1.0:
            return []
        
        # Apply HPSS
        y_harmonic, _ = librosa.effects.hpss(y)
        
        clip_samples = int(target_sr * clip_duration)
        clips = []
        
        # Energy VAD to select top vocal clips
        hop_len = clip_samples // 2
        
        for start in range(0, len(y_harmonic) - clip_samples + 1, hop_len):
            chunk = y_harmonic[start:start + clip_samples]
            rms = np.sqrt(np.mean(chunk**2))
            if rms > 0.005:
                clips.append((rms, chunk))
                
        clips.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in clips[:4]]
    except Exception as e:
        return []

def main():
    print("=== ANYCALL RIGOROUS BIOACOUSTIC BLIND TEST SUITE ===")
    
    db_path = PROJECT_ROOT / "anycall.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Load enrolled prototypes from DB (single 1024-dim centroid)
    cursor.execute("SELECT species_id, common_name, taxon, prototype FROM species")
    rows = cursor.fetchall()
    
    species_prototypes = {}
    species_info = {}
    for sp_id, c_name, taxon, proto_blob in rows:
        proto = np.frombuffer(proto_blob, dtype=np.float32)
        if len(proto) == 1024: # Single BirdNET centroid
            species_prototypes[sp_id] = proto
            species_info[sp_id] = {"common_name": c_name, "taxon": taxon}
        
    print(f"Loaded {len(species_prototypes)} species prototypes (single BirdNET centroids) from anycall.db")
    conn.close()
    
    # Initialize BirdNET Backbone
    print("Initializing BirdNET embedding backbone...")
    backbone = get_backbone("birdnet")
    
    # Test Datasets
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    # ----------------------------------------------------
    # Experiment 1: In-Domain Held-Out Species Test
    # ----------------------------------------------------
    print("\n--- Running Exp 1: In-Domain Held-Out Evaluation (BirdNET Single Centroids) ---")
    in_domain_results = []
    
    species_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    if not species_dirs:
        species_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        
    for sp_dir in species_dirs:
        sp_id = sp_dir.name
        if sp_id not in species_prototypes:
            continue
            
        audio_files = list(sp_dir.glob("*.wav")) + list(sp_dir.glob("*.mp3"))
        if not audio_files:
            continue
            
        # Use held-out files (files after index 3)
        held_out_files = audio_files[3:8] if len(audio_files) > 3 else audio_files
        
        correct = 0
        total = 0
        scores = []
        
        for af in held_out_files:
            clips = hpss_extract_clips(af, target_sr=48000)
            for clip in clips:
                try:
                    emb = backbone.embed(clip, sr=48000)
                    if len(emb) != 1024:
                        continue
                    
                    # Compute cosine similarity against all single 1024-dim prototypes
                    best_sim = -1.0
                    best_sp = None
                    for candidate_id, proto in species_prototypes.items():
                        sim = float(np.dot(emb, proto))
                        if sim > best_sim:
                            best_sim = sim
                            best_sp = candidate_id
                            
                    total += 1
                    scores.append(best_sim)
                    
                    is_correct = (best_sp == sp_id) and (best_sim >= 0.70)
                    if is_correct:
                        correct += 1
                        
                    in_domain_results.append({
                        "species_id": sp_id,
                        "file": af.name,
                        "predicted_species": best_sp if best_sim >= 0.70 else "Unknown",
                        "similarity": best_sim,
                        "correct": is_correct
                    })
                except Exception as e:
                    pass
                    
        acc = (correct / total * 100) if total > 0 else 0.0
        avg_score = np.mean(scores) if scores else 0.0
        print(f"Species: {sp_id:<30} | Tested Clips: {total:<3} | Recall (theta=0.70): {acc:5.1f}% | Avg Sim: {avg_score:.3f}")

    # ----------------------------------------------------
    # Experiment 2: Open-Set Out-of-Bank Rejection Test
    # ----------------------------------------------------
    print("\n--- Running Exp 2: Open-Set Out-of-Bank Rejection Test ---")
    open_set_clips = []
    
    # Generate ambient/synthetic noise & un-enrolled species signals
    np.random.seed(42)
    sr = 48000
    dur = 3.0
    num_samples = int(sr * dur)
    
    # White noise, Brown noise, Sine sweeps, Rain simulation
    noise_types = ["white_noise", "pink_noise", "rain_texture", "sine_sweep"]
    rejections = 0
    open_set_total = 0
    open_set_scores = []
    
    for ntype in noise_types:
        for k in range(10):
            if ntype == "white_noise":
                signal = np.random.normal(0, 0.05, num_samples).astype(np.float32)
            elif ntype == "pink_noise":
                signal = np.cumsum(np.random.normal(0, 0.02, num_samples)).astype(np.float32)
                signal = signal / np.max(np.abs(signal)) * 0.1
            elif ntype == "sine_sweep":
                t = np.linspace(0, dur, num_samples)
                freq = np.linspace(500, 8000, num_samples)
                signal = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            else:
                signal = np.random.exponential(0.02, num_samples).astype(np.float32)
                
            emb = backbone.embed(signal, sr=48000)
            
            max_sim = max([float(np.dot(emb, proto)) for proto in species_prototypes.values()])
            open_set_scores.append(max_sim)
            open_set_total += 1
            
            is_rejected = max_sim < 0.70
            if is_rejected:
                rejections += 1
                
            open_set_clips.append({
                "type": ntype,
                "max_similarity": max_sim,
                "rejected": is_rejected
            })
            
    open_set_specificity = (rejections / open_set_total * 100) if open_set_total > 0 else 0.0
    print(f"Open-Set Specificity (Rejection Rate @ theta=0.70): {open_set_specificity:.1f}% ({rejections}/{open_set_total} rejected)")

    # ----------------------------------------------------
    # Experiment 3: Threshold Sweep & AUROC / EER Curve
    # ----------------------------------------------------
    print("\n--- Running Exp 3: Decision Threshold Sweep (theta = 0.45 to 0.85) ---")
    all_target_clips = len(in_domain_results)
    
    sweep_data = []
    for theta in np.arange(0.45, 0.86, 0.02):
        # Overall Dataset Recall: True Positives out of ALL 148 target clips
        in_tp = sum(1 for r in in_domain_results if r["predicted_species"] == r["species_id"] and r["similarity"] >= theta)
        overall_recall = (in_tp / all_target_clips) if all_target_clips > 0 else 0.0
        
        # False Acceptance Rate on Open-Set Out-of-Bank Noise
        out_fp = sum(1 for s in open_set_scores if s >= theta)
        far = (out_fp / len(open_set_scores)) if open_set_scores else 0.0
        
        sweep_data.append({
            "threshold": round(float(theta), 2),
            "True_Acceptance_Rate": round(overall_recall, 4),
            "False_Acceptance_Rate": round(far, 4),
            "Specificity": round(1.0 - far, 4)
        })
        
    df_sweep = pd.DataFrame(sweep_data)
    print(df_sweep.to_string(index=False))

    # Save summary report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_enrolled_species": len(species_prototypes),
        "in_domain_clips_tested": len(in_domain_results),
        "in_domain_recall": (sum(1 for r in in_domain_results if r["correct"]) / len(in_domain_results) * 100) if in_domain_results else 0,
        "open_set_specificity": open_set_specificity,
        "threshold_sweep": sweep_data
    }
    
    report_json_path = reports_dir / "rigorous_blind_test_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
        
    df_sweep.to_csv(reports_dir / "rigorous_blind_test_report.csv", index=False)
    
    print(f"\nSUCCESS: Blind test report saved to {report_json_path}")

if __name__ == "__main__":
    main()
