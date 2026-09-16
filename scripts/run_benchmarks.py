#!/usr/bin/env python3
"""AnyCall Benchmark Runner CLI.

Usage
-----
  python scripts/run_benchmarks.py
  python scripts/run_benchmarks.py --backbones mock
  python scripts/run_benchmarks.py --backbones birdnet perch panns --k-shots 5 10 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anycall.benchmark.runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="AnyCall Benchmarking Suite")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Root directory of processed audio segments (default: data/processed)",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["birdnet", "perch", "panns"],
        help="Backbone(s) to evaluate. Use 'mock' for fast testing.",
    )
    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="Support set sizes to sweep (default: 5 10 20)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--latency-trials",
        type=int,
        default=20,
        help="Number of forward passes for latency profiling (default: 20)",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_report.json",
        help="Path for JSON output (default: benchmark_report.json)",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_report.csv",
        help="Path for CSV output (default: benchmark_report.csv)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(
        data_dir=args.data_dir,
        k_shots_list=args.k_shots,
        n_folds=args.n_folds,
        backbones=args.backbones,
        n_latency_trials=args.latency_trials,
    )

    report = runner.run_all(
        output_json=args.output_json,
        output_csv=args.output_csv,
    )

    if report.errors:
        print(f"\n[Benchmark] Completed with {len(report.errors)} error(s):")
        for e in report.errors:
            print(f"  - {e[:120]}")
    else:
        print("\n[Benchmark] All experiments completed successfully.")

    print("\n[Benchmark] Summary:")
    for k, v in report.summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
