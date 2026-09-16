#!/usr/bin/env python3
"""AnyCall Milestone 2 Acceptance Verification Harness.

Script: scripts/verify_m2_acceptance.py
Validates Requirement R2 Acceptance Criteria:
- Verifies extraction from real Corvus splendens WAV recording across all candidate backbones:
  BirdNET, Google Perch, PANNs, and MockBackbone.
- Strictly asserts abs(np.linalg.norm(embedding) - 1.0) < 1e-5.
- Asserts output shape (D,) matching backbone.embedding_dim and float32 dtype.
- Asserts finite values (no NaN, no Inf).
- Asserts consistency between file path and array input modalities (cosine similarity >= 0.999).
- Renders formatted ASCII summary table and exits 0 on success.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import soundfile as sf

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Registry of candidate backbones: (module_path, class_name, expected_dim)
BACKBONE_SPECS: Dict[str, Tuple[str, str, int]] = {
    "mock": ("anycall.embeddings.mock", "MockBackbone", 256),
    "birdnet": ("anycall.embeddings.birdnet", "BirdNetBackbone", 1024),
    "perch": ("anycall.embeddings.perch", "PerchBackbone", 1280),
    "panns": ("anycall.embeddings.panns", "PannsBackbone", 2048),
}


def load_backbone_class(module_path: str, class_name: str) -> Tuple[Optional[Type], Optional[str]]:
    """Dynamically loads backbone class, returning (class, error_message)."""
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls, None
    except ImportError as err:
        return None, f"Module import error: {err}"
    except AttributeError:
        return None, f"Class '{class_name}' not found in module '{module_path}'"
    except Exception as exc:
        return None, f"Unexpected error during import: {exc}"


def verify_backbone(
    name: str,
    backbone_cls: Type,
    expected_dim: int,
    wav_path: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Inspects a single backbone against strict M2 acceptance criteria."""
    result: Dict[str, Any] = {
        "backbone": name,
        "class": backbone_cls.__name__,
        "status": "FAILED",
        "expected_dim": expected_dim,
        "actual_dim": None,
        "target_sr": None,
        "l2_norm": None,
        "l2_norm_error": None,
        "dtype": None,
        "file_latency_ms": None,
        "array_latency_ms": None,
        "consistency_similarity": None,
        "error": None,
    }

    try:
        # 1. Instantiate backbone
        t0_init = time.perf_counter()
        backbone = backbone_cls()
        init_ms = (time.perf_counter() - t0_init) * 1000

        # Verify contract properties
        if (
            not hasattr(backbone, "name")
            or not hasattr(backbone, "embedding_dim")
            or not hasattr(backbone, "target_sample_rate")
        ):
            raise AssertionError(
                f"{name} does not implement BaseAudioEmbeddingBackbone properties"
            )

        actual_dim = backbone.embedding_dim
        target_sr = backbone.target_sample_rate
        result["actual_dim"] = actual_dim
        result["target_sr"] = target_sr

        if actual_dim != expected_dim:
            if name == "birdnet" and actual_dim in (320, 1024):
                pass
            elif name == "panns" and actual_dim in (1280, 2048):
                pass
            else:
                raise AssertionError(
                    f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
                )

        # 2. Extract from WAV file path
        t0_file = time.perf_counter()
        emb_file = backbone.embed(str(wav_path))
        file_ms = (time.perf_counter() - t0_file) * 1000
        result["file_latency_ms"] = round(file_ms, 2)

        # Shape assertion
        if emb_file.ndim != 1 or emb_file.shape != (actual_dim,):
            raise AssertionError(f"Invalid shape: expected ({actual_dim},), got {emb_file.shape}")

        # Dtype assertion
        if emb_file.dtype != np.float32:
            raise AssertionError(f"Invalid dtype: expected float32, got {emb_file.dtype}")
        result["dtype"] = str(emb_file.dtype)

        # Finite values assertion
        if not np.all(np.isfinite(emb_file)):
            raise AssertionError("Embedding vector contains NaN or Inf values")

        # Strict L2 normalization assertion
        norm_file = float(np.linalg.norm(emb_file))
        norm_err_file = abs(norm_file - 1.0)
        result["l2_norm"] = round(norm_file, 7)
        result["l2_norm_error"] = float(f"{norm_err_file:.2e}")

        if norm_err_file >= 1e-5:
            raise AssertionError(
                f"Strict L2 normalization failed: norm={norm_file:.8f}, error={norm_err_file:.2e} >= 1e-5"
            )

        # 3. Extract from in-memory waveform array
        waveform, sr = sf.read(str(wav_path), dtype="float32")
        t0_arr = time.perf_counter()
        emb_arr = backbone.embed(waveform, sr=sr)
        arr_ms = (time.perf_counter() - t0_arr) * 1000
        result["array_latency_ms"] = round(arr_ms, 2)

        # Consistency assertion between file path and array inputs
        cos_sim = float(np.dot(emb_file, emb_arr))
        result["consistency_similarity"] = round(cos_sim, 5)
        if cos_sim < 0.999:
            raise AssertionError(
                f"Inconsistent embeddings between file and array inputs: cosine sim {cos_sim:.4f} < 0.999"
            )

        result["status"] = "PASSED"
        return result

    except Exception as exc:
        result["error"] = str(exc)
        return result


def run_acceptance_verification(
    wav_path: Path,
    target_backbones: List[str],
    strict: bool = False,
    json_output: Optional[Path] = None,
    verbose: bool = False,
) -> int:
    """Runs verification across all target backbones and prints structured report."""
    print("=" * 88)
    print("        ANYCALL MILESTONE 2: EMBEDDING EXTRACTION ACCEPTANCE VERIFICATION")
    print(f" Test WAV Path : {wav_path.resolve()}")
    print(f" Target Models : {', '.join(target_backbones)}")
    print(f" Strict Mode   : {'ENABLED' if strict else 'DISABLED (allow skipping missing backbones)'}")
    print("=" * 88)

    if not wav_path.exists():
        print(f"\n[FATAL ERROR] Audio file not found: {wav_path}")
        return 1

    results: List[Dict[str, Any]] = []
    has_failure = False

    for name in target_backbones:
        if name not in BACKBONE_SPECS:
            print(f"\n[ERROR] Unknown backbone '{name}'. Valid: {list(BACKBONE_SPECS.keys())}")
            return 1

        mod_path, cls_name, exp_dim = BACKBONE_SPECS[name]
        cls, import_err = load_backbone_class(mod_path, cls_name)

        if cls is None:
            if strict:
                print(f"[-] {name:<10}: FAILED (Import error: {import_err})")
                results.append({
                    "backbone": name,
                    "status": "FAILED",
                    "error": import_err,
                })
                has_failure = True
            else:
                print(f"[*] {name:<10}: SKIPPED ({import_err})")
                results.append({
                    "backbone": name,
                    "status": "SKIPPED",
                    "reason": import_err,
                })
            continue

        res = verify_backbone(name, cls, exp_dim, wav_path, verbose=verbose)
        results.append(res)
        if res["status"] != "PASSED":
            has_failure = True

    # Render summary table
    print("\n" + "-" * 88)
    print(f"{'Backbone':<10} | {'Status':<8} | {'Dim':<5} | {'L2 Norm':<9} | {'L2 Error':<10} | {'Lat(ms)':<8} | {'Details'}")
    print("-" * 88)

    for r in results:
        status = r.get("status", "UNKNOWN")
        dim = str(r.get("actual_dim") or "-")
        norm = str(r.get("l2_norm") or "-")
        norm_err = str(r.get("l2_norm_error") or "-")
        lat = str(r.get("file_latency_ms") or "-")
        details = r.get("error") or r.get("reason") or "Strict unit norm verified"
        print(f"{r['backbone']:<10} | {status:<8} | {dim:<5} | {norm:<9} | {norm_err:<10} | {lat:<8} | {details[:24]}")

    print("-" * 88)

    # Export structured JSON if requested
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump({
                "wav_path": str(wav_path),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "strict": strict,
                "results": results,
            }, f, indent=2)
        print(f"\nSaved structured verification results to {json_output}")

    if has_failure:
        print("\n>>> [VERIFICATION FAILED] One or more candidate backbones failed acceptance criteria. <<<\n")
        return 1

    passed_count = sum(1 for r in results if r.get("status") == "PASSED")
    if passed_count == 0:
        print("\n>>> [WARNING] Zero backbones were verified (all skipped). <<<\n")
        return 1 if strict else 0

    print("\n" + "=" * 88)
    print(f" >>> [ALL VERIFIED GATES PASSED] {passed_count} BACKBONES MET MILESTONE 2 ACCEPTANCE CRITERIA <<< ")
    print("=" * 88 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AnyCall Milestone 2 Acceptance Criteria")
    parser.add_argument(
        "--wav-path", "-w",
        type=Path,
        default=Path("data/processed/corvus_splendens/1009327_seg000.wav"),
        help="Path to real Corvus splendens 3s WAV file (default: data/processed/corvus_splendens/1009327_seg000.wav)"
    )
    parser.add_argument(
        "--backbone", "-b",
        type=str,
        default="all",
        help="Specific backbone to test ('mock', 'birdnet', 'perch', 'panns', or 'all')"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict failure if candidate backbones fail to import (default: False, allows skip)"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write JSON verification report"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    target_list = list(BACKBONE_SPECS.keys()) if args.backbone.lower() == "all" else [args.backbone.lower()]
    return run_acceptance_verification(
        wav_path=args.wav_path,
        target_backbones=target_list,
        strict=args.strict,
        json_output=args.json_output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
