"""AnyCall 33-Species Comprehensive Benchmark Script.

Evaluates few-shot prototypical classification performance across all 33 curated resident
Indian wildlife species (birds, insects, frogs, mammals) in data/processed_roi/.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

from anycall.storage.db import DatabaseManager
from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.data.species import SPECIES_CATALOG


def run_benchmark():
    print("=" * 80)
    print("        AnyCall Full 33-Species Prototypical Classification Benchmark       ")
    print("=" * 80)

    db_path = "anycall.db"
    roi_dir = Path("data/processed_roi")

    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found.")
        return

    if not roi_dir.exists():
        print(f"Error: ROI directory {roi_dir} not found.")
        return

    # Load Backbone & DB
    print("[1/3] Loading BirdNET backbone & enrolling 33 prototypes from DB...")
    bb = BirdNetBackbone()
    db = DatabaseManager(db_path)
    rows = db._conn.execute("SELECT species_id, prototype, sample_count FROM species").fetchall()

    if not rows:
        print("Error: No species enrolled in database.")
        return

    classifier = PrototypicalClassifier(threshold=0.60)
    enrolled_species_set = set()
    for r in rows:
        sp_id, proto_bytes, n = r
        vec = np.frombuffer(proto_bytes, dtype=np.float32)
        classifier.enroll(sp_id, [vec])
        enrolled_species_set.add(sp_id)

    print(f"Successfully loaded {len(enrolled_species_set)} enrolled species prototypes.\n")

    # Evaluate across thresholds
    thresholds_to_test = [0.55, 0.60, 0.65, 0.70]
    
    species_dirs = sorted([d for d in roi_dir.iterdir() if d.is_dir() and d.name in enrolled_species_set])
    print(f"[2/3] Benchmarking {len(species_dirs)} species directories...")

    per_species_results = {}
    all_eval_samples = []

    for sp_dir in tqdm(species_dirs, desc="Benchmarking Species"):
        sp_id = sp_dir.name
        wavs = sorted(list(sp_dir.glob("*.wav")))
        if not wavs:
            continue

        sp_correct_top1 = 0
        sp_correct_top3 = 0
        sp_scores = []
        sp_margins = []

        for wf in wavs:
            try:
                audio, sr = sf.read(str(wf), dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                emb = bb.embed(audio, sr=sr)
                pred = classifier.predict(emb)
                
                sorted_scores = sorted(pred.scores.items(), key=lambda x: x[1], reverse=True)
                top1_label, top1_score = sorted_scores[0]
                top3_labels = [s[0] for s in sorted_scores[:3]]

                # Competitor score (highest score of any OTHER species)
                competitors = [s[1] for s in sorted_scores if s[0] != sp_id]
                top_competitor_score = max(competitors) if competitors else 0.0
                true_score = pred.scores.get(sp_id, 0.0)
                margin = true_score - top_competitor_score

                is_top1 = (top1_label == sp_id)
                is_top3 = (sp_id in top3_labels)

                if is_top1:
                    sp_correct_top1 += 1
                if is_top3:
                    sp_correct_top3 += 1

                sp_scores.append(true_score)
                sp_margins.append(margin)

                all_eval_samples.append({
                    "true_species": sp_id,
                    "wav_name": wf.name,
                    "top1_species": top1_label,
                    "top1_score": top1_score,
                    "true_score": true_score,
                    "top_competitor_score": top_competitor_score,
                    "margin": margin,
                    "is_top1": is_top1,
                    "is_top3": is_top3,
                })

            except Exception as e:
                print(f"Warning error processing {wf.name}: {e}")

        total_sp_samples = len(wavs)
        per_species_results[sp_id] = {
            "sample_count": total_sp_samples,
            "top1_correct": sp_correct_top1,
            "top3_correct": sp_correct_top3,
            "top1_acc": (sp_correct_top1 / total_sp_samples) * 100.0 if total_sp_samples else 0.0,
            "top3_acc": (sp_correct_top3 / total_sp_samples) * 100.0 if total_sp_samples else 0.0,
            "avg_true_score": float(np.mean(sp_scores)) if sp_scores else 0.0,
            "avg_margin": float(np.mean(sp_margins)) if sp_margins else 0.0,
        }

    # Macro & Micro Totals
    total_samples = len(all_eval_samples)
    total_top1_correct = sum(1 for s in all_eval_samples if s["is_top1"])
    total_top3_correct = sum(1 for s in all_eval_samples if s["is_top3"])

    overall_top1_acc = (total_top1_correct / total_samples) * 100.0 if total_samples else 0.0
    overall_top3_acc = (total_top3_correct / total_samples) * 100.0 if total_samples else 0.0

    print("\n" + "=" * 80)
    print("                    PER-SPECIES BENCHMARK REPORT TABLE                      ")
    print("=" * 80)
    print(f"{'Species ID':<32} | {'Taxon':<8} | {'Clips':<5} | {'Top-1 Acc':<9} | {'Top-3 Acc':<9} | {'Avg Score':<9} | {'Avg Margin':<9}")
    print("-" * 80)

    for sp_id, stats in per_species_results.items():
        meta = SPECIES_CATALOG.get(sp_id)
        taxon = (meta.taxon.value if hasattr(meta.taxon, "value") else str(meta.taxon)).capitalize() if meta else "Aves"
        print(f"{sp_id:<32} | {taxon:<8} | {stats['sample_count']:<5} | {stats['top1_acc']:>8.1f}% | {stats['top3_acc']:>8.1f}% | {stats['avg_true_score']:>9.3f} | {stats['avg_margin']:>9.3f}")

    print("=" * 80)
    print(f" TOTAL EVALUATION CLIPS : {total_samples}")
    print(f" MICRO TOP-1 ACCURACY   : {overall_top1_acc:.2f}% ({total_top1_correct}/{total_samples})")
    print(f" MICRO TOP-3 ACCURACY   : {overall_top3_acc:.2f}% ({total_top3_correct}/{total_samples})")
    print("=" * 80)

    # Threshold Acceptance Analysis
    print("\n[3/3] Threshold Acceptance & Rejection Analysis:")
    print(f"{'Threshold (theta)':<18} | {'Accepted Match Rate':<22} | {'False Unknown Rate':<20}")
    print("-" * 65)

    for theta in thresholds_to_test:
        accepted = sum(1 for s in all_eval_samples if s["is_top1"] and s["top1_score"] >= theta and s["margin"] >= 0.01)
        rejected = total_samples - accepted
        acc_pct = (accepted / total_samples) * 100.0 if total_samples else 0.0
        rej_pct = (rejected / total_samples) * 100.0 if total_samples else 0.0
        print(f"theta = {theta:<10.2f} | {acc_pct:>20.1f}% | {rej_pct:>18.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
