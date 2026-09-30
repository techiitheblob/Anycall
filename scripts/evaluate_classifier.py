import json
import logging
import sys
from pathlib import Path
import numpy as np

from anycall.benchmark.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    print("=" * 60)
    print("  AnyCall Classifier Evaluation (Performance & Accuracy)")
    print("=" * 60)
    
    # Run the benchmark suite
    # Note: Using fewer folds and only birdnet/perch for speed
    runner = BenchmarkRunner(
        data_dir="data/processed_roi",
        k_shots_list=[5, 10], 
        n_folds=3,
        backbones=["birdnet", "perch"]
    )
    
    try:
        report = runner.run_all(
            output_json="reports/evaluation_results.json",
            output_csv="reports/evaluation_results.csv"
        )
    except Exception as e:
        print(f"\\nError during evaluation: {e}")
        sys.exit(1)
        
    print("\\n\\n============================================================")
    print("                      EVALUATION RESULTS")
    print("============================================================")
    
    # 1. Few-Shot Accuracy
    print("\\n[1] FEW-SHOT CLASSIFICATION ACCURACY")
    print("-" * 60)
    print(f"{'Backbone':<15} | {'Shots (K)':<10} | {'Mean Top-1 Accuracy':<20}")
    print("-" * 60)
    
    # Aggregate accuracy by backbone and K
    acc_map = {}
    for row in report.exp2_few_shot:
        bb = row["backbone"]
        k = row["k_shots"]
        acc = row["top1_accuracy"]
        key = (bb, k)
        if key not in acc_map:
            acc_map[key] = []
        acc_map[key].append(acc)
        
    for bb in runner._backbone_names:
        for k in runner._k_shots_list:
            key = (bb, k)
            if key in acc_map:
                mean_acc = np.mean(acc_map[key]) * 100.0
                print(f"{bb:<15} | {k:<10} | {mean_acc:>18.2f}%")
                
    # 2. Out-of-Distribution Rejection (EER & AUC)
    print("\\n[2] FALSE POSITIVE REJECTION (ROC)")
    print("-" * 60)
    print("How well does the system reject 'Unknown' novel sounds?")
    print(f"{'Backbone':<15} | {'EER (Lower is better)':<22} | {'AUC (Higher is better)':<22}")
    print("-" * 60)
    for bb in runner._backbone_names:
        metrics = next((m for m in report.exp4_roc if m.get("backbone") == bb), {})
        if metrics:
            eer = metrics.get("eer", 0) * 100.0
            auc = metrics.get("auroc", 0) * 100.0
            print(f"{bb:<15} | {eer:>21.2f}% | {auc:>21.2f}%")
            
    # 3. Cross-Taxa Performance
    print("\\n[3] CROSS-TAXA GENERALIZATION (BirdNET)")
    print("-" * 60)
    print("Does the BirdNET embedding space work on non-birds?")
    taxa_results = [m for m in report.exp3_cross_taxa if m.get("backbone") == "birdnet"]
    if taxa_results:
        for res in taxa_results:
            taxon = res.get("taxon", "unknown")
            acc = res.get("top1_accuracy", 0) * 100.0
            n_sp = res.get("n_species", 0)
            print(f"{taxon.capitalize():<15} | {acc:>18.2f}% | {n_sp:>14}")
    else:
        print("Not enough taxa data to compute cross-taxa generalization.")
        
    print("\\nReports saved to 'reports/' directory.")

if __name__ == "__main__":
    main()
