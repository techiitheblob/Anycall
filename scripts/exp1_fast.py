import json
from anycall.benchmark.runner import BenchmarkRunner
from pathlib import Path

def main():
    print("=" * 60)
    print("  Fast Evaluation: Exp 1 BirdNET Stock vs AnyCall")
    print("=" * 60)
    
    runner = BenchmarkRunner(data_dir="data/processed")
    species_map = runner._discover_species()
    
    print(f"[Benchmark] Found {len(species_map)} species.")
    print("[Benchmark] Running Exp 1 (BirdNET Stock vs AnyCall)...")
    
    results = runner._exp1_birdnet_failure(species_map)
    
    # Calculate means
    stock_acc = sum(r.birdnet_correct for r in results) / max(1, len(results))
    anycall_acc = sum(r.anycall_correct for r in results) / max(1, len(results))
    
    print("\\n" + "=" * 60)
    print("  RESULTS (BirdNET Indian Species Failure Analysis)")
    print("=" * 60)
    print(f"Total test clips: {len(results)}")
    print(f"Stock BirdNET Accuracy: {stock_acc * 100:.1f}%")
    print(f"AnyCall Prototypical:   {anycall_acc * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
