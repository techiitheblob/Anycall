#!/usr/bin/env python3
"""AnyCall E2E Test Suite Runner.

Discovers and executes tests in tests/e2e/, tracks execution metrics,
enforces execution time budgets (< 5.0s for fast offline tiers),
and produces human-readable ASCII tables and structured JSON reports.

Usage:
    python tests/run_e2e_tests.py [options]

Options:
    --tier {1,2,3,4,all}     Run tests for a specific tier or all tiers (default: all)
    -v, --verbose            Enable verbose test execution output
    --fast                   Enforce execution time budget (< 5.0 seconds)
    --json-output PATH       Export detailed test run metrics to JSON file
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple


# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MetricsTestResult(unittest.TestResult):
    """Custom TestResult collecting detailed per-test timing and outcome metrics."""

    def __init__(self, stream=None, descriptions=None, verbosity=1):
        super().__init__(stream, descriptions, verbosity)
        self.verbosity = verbosity
        self.test_timings: Dict[str, float] = {}
        self.test_outcomes: Dict[str, str] = {}
        self._start_time: Optional[float] = None
        self.test_details: List[Dict[str, Any]] = []

    def startTest(self, test):
        super().startTest(test)
        self._start_time = time.perf_counter()
        if self.verbosity > 1:
            test_id = test.id().split(".")[-1]
            sys.stdout.write(f"  - Running {test_id} ... ")
            sys.stdout.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        self.test_timings[test.id()] = elapsed
        self.test_outcomes[test.id()] = "PASSED"
        self.test_details.append({
            "test_id": test.id(),
            "status": "PASSED",
            "duration_sec": round(elapsed, 4),
        })
        if self.verbosity > 1:
            print(f"PASSED ({elapsed:.3f}s)")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        self.test_timings[test.id()] = elapsed
        self.test_outcomes[test.id()] = "FAILED"
        self.test_details.append({
            "test_id": test.id(),
            "status": "FAILED",
            "duration_sec": round(elapsed, 4),
            "error": self._exc_info_to_string(err, test),
        })
        if self.verbosity > 1:
            print(f"FAILED ({elapsed:.3f}s)")

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        self.test_timings[test.id()] = elapsed
        self.test_outcomes[test.id()] = "ERROR"
        self.test_details.append({
            "test_id": test.id(),
            "status": "ERROR",
            "duration_sec": round(elapsed, 4),
            "error": self._exc_info_to_string(err, test),
        })
        if self.verbosity > 1:
            print(f"ERROR ({elapsed:.3f}s)")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        self.test_timings[test.id()] = elapsed
        self.test_outcomes[test.id()] = "SKIPPED"
        self.test_details.append({
            "test_id": test.id(),
            "status": "SKIPPED",
            "reason": reason,
            "duration_sec": round(elapsed, 4),
        })
        if self.verbosity > 1:
            print(f"SKIPPED ({reason})")


def discover_suite_for_tier(tier: str) -> unittest.TestSuite:
    """Load test cases matching the requested tier."""
    loader = unittest.TestLoader()
    e2e_dir = PROJECT_ROOT / "tests" / "e2e"

    suite = unittest.TestSuite()

    if tier in ("1", "all"):
        t1_path = e2e_dir / "test_tier1_features.py"
        if t1_path.exists():
            suite.addTests(loader.discover(str(e2e_dir), pattern="test_tier1_features.py"))

    if tier in ("2", "all"):
        t2_path = e2e_dir / "test_tier2_boundaries.py"
        if t2_path.exists():
            suite.addTests(loader.discover(str(e2e_dir), pattern="test_tier2_boundaries.py"))

    if tier in ("3", "all"):
        t3_path = e2e_dir / "test_tier3_cross_feature.py"
        if t3_path.exists():
            suite.addTests(loader.discover(str(e2e_dir), pattern="test_tier3_cross_feature.py"))

    if tier in ("4", "all"):
        t4_path = e2e_dir / "test_tier4_real_world.py"
        if t4_path.exists():
            suite.addTests(loader.discover(str(e2e_dir), pattern="test_tier4_real_world.py"))

    return suite


def print_ascii_banner(title: str, character: str = "="):
    """Print formatted section banner."""
    width = 76
    print(character * width)
    print(f"  {title}".center(width - 2))
    print(character * width)


def print_summary_table(results_by_tier: Dict[str, Dict[str, Any]], total_elapsed: float):
    """Render a formatted ASCII table of test execution metrics."""
    print("\n" + "=" * 76)
    print(f"{'Tier':<10} | {'Total':<6} | {'Passed':<6} | {'Skipped':<7} | {'Failed':<6} | {'Errors':<6} | {'Time (s)':<8}")
    print("-" * 76)

    total_tests = 0
    total_passed = 0
    total_skipped = 0
    total_failed = 0
    total_errors = 0

    for tier, data in sorted(results_by_tier.items()):
        print(
            f"{tier:<10} | "
            f"{data['total']:<6} | "
            f"{data['passed']:<6} | "
            f"{data['skipped']:<7} | "
            f"{data['failed']:<6} | "
            f"{data['errors']:<6} | "
            f"{data['duration']:.3f}"
        )
        total_tests += data["total"]
        total_passed += data["passed"]
        total_skipped += data["skipped"]
        total_failed += data["failed"]
        total_errors += data["errors"]

    print("-" * 76)
    print(
        f"{'TOTAL':<10} | "
        f"{total_tests:<6} | "
        f"{total_passed:<6} | "
        f"{total_skipped:<7} | "
        f"{total_failed:<6} | "
        f"{total_errors:<6} | "
        f"{total_elapsed:.3f}"
    )
    print("=" * 76 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="AnyCall E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Tier selection (default: all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose test reporting",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enforce execution time budget (< 5.0 seconds)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save structured metrics JSON to destination path",
    )

    args = parser.parse_args()

    print_ascii_banner(f"AnyCall E2E Test Runner [Tier: {args.tier.upper()}]")
    print(f"  Working directory : {PROJECT_ROOT}")
    print(f"  Python executable : {sys.executable}")
    print(f"  Fast mode budget  : {'ENABLED (< 5.0s)' if args.fast else 'DISABLED'}")
    print()

    # Determine tiers to run
    tiers_to_run = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]
    results_by_tier: Dict[str, Dict[str, Any]] = {}
    all_details: List[Dict[str, Any]] = []

    total_start = time.perf_counter()
    overall_success = True

    for tier in tiers_to_run:
        tier_label = f"Tier {tier}"
        if args.verbose:
            print(f"\n--- Discovering {tier_label} Tests ---")

        suite = discover_suite_for_tier(tier)
        tier_start = time.perf_counter()

        runner_verbosity = 2 if args.verbose else 1
        result = MetricsTestResult(verbosity=runner_verbosity)
        suite.run(result)
        tier_elapsed = time.perf_counter() - tier_start

        passed_count = len(result.test_outcomes) - len(result.failures) - len(result.errors) - len(result.skipped)
        results_by_tier[tier_label] = {
            "total": result.testsRun,
            "passed": max(0, passed_count),
            "skipped": len(result.skipped),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "duration": tier_elapsed,
        }
        all_details.extend(result.test_details)

        if not result.wasSuccessful():
            overall_success = False

    total_elapsed = time.perf_counter() - total_start

    # Render summary table
    print_summary_table(results_by_tier, total_elapsed)

    # Budget assertion
    budget_passed = True
    if args.fast:
        if total_elapsed > 5.0:
            print(f"[BUDGET EXCEEDED] Fast mode target is < 5.0s, elapsed: {total_elapsed:.3f}s")
            budget_passed = False
        else:
            print(f"[BUDGET SATISFIED] Execution completed in {total_elapsed:.3f}s (< 5.0s budget)")

    # Status verdict banner
    if overall_success and budget_passed:
        print(">>> RESULT: ALL EXECUTED SUITES PASSED <<<")
        exit_code = 0
    else:
        print(">>> RESULT: TEST SUITE HAS FAILURES OR BUDGET VIOLATIONS <<<")
        exit_code = 1

    # Optional JSON output
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tier_selection": args.tier,
            "total_elapsed_sec": round(total_elapsed, 4),
            "success": overall_success and budget_passed,
            "tiers": results_by_tier,
            "tests": all_details,
        }
        out_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        print(f"Metrics saved to {out_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
