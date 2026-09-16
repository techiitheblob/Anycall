"""AnyCall Milestone 1 Acceptance Verification Harness.

Script: scripts/verify_m1_acceptance.py
Validates Requirement R1:
- Downloads >= 20 recordings for sample species (Corvus splendens).
- Outputs uniform 3-second, 48kHz mono 16-bit PCM WAV files.
"""
import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import soundfile as sf

EXPECTED_SR = 48000
EXPECTED_CHANNELS = 1
EXPECTED_DURATION_SEC = 3.0
EXPECTED_SAMPLES = int(EXPECTED_SR * EXPECTED_DURATION_SEC)  # 144,000
EXPECTED_FORMAT = "WAV"
EXPECTED_SUBTYPE = "PCM_16"


def verify_single_file(file_path: Path) -> Tuple[bool, Dict[str, Any], str]:
    """Inspects a single WAV file against strict AnyCall M1 specifications."""
    try:
        info = sf.info(str(file_path))
    except Exception as e:
        return False, {}, f"Failed to open audio file with soundfile: {e}"

    details: Dict[str, Any] = {
        "file": file_path.name,
        "format": info.format,
        "subtype": info.subtype,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
    }

    # Format checks
    if info.format != EXPECTED_FORMAT:
        return False, details, f"Invalid format: expected {EXPECTED_FORMAT}, got {info.format}"
    if info.subtype != EXPECTED_SUBTYPE:
        return False, details, f"Invalid subtype: expected {EXPECTED_SUBTYPE}, got {info.subtype}"
    if info.channels != EXPECTED_CHANNELS:
        return False, details, f"Invalid channel count: expected {EXPECTED_CHANNELS} (mono), got {info.channels}"
    if info.samplerate != EXPECTED_SR:
        return False, details, f"Invalid sample rate: expected {EXPECTED_SR} Hz, got {info.samplerate} Hz"
    if info.frames != EXPECTED_SAMPLES:
        return False, details, f"Invalid sample count: expected {EXPECTED_SAMPLES} samples (3.0s), got {info.frames}"

    # Numerical checks
    try:
        data, _ = sf.read(str(file_path), dtype="float32")
    except Exception as e:
        return False, details, f"Failed to read waveform data: {e}"

    if len(data) != EXPECTED_SAMPLES:
        return False, details, f"Read array length {len(data)} != expected {EXPECTED_SAMPLES}"
    if np.isnan(data).any():
        return False, details, "Waveform contains NaN values"
    if np.isinf(data).any():
        return False, details, "Waveform contains Inf values"

    max_amp = float(np.max(np.abs(data)))
    details["peak_amp"] = max_amp
    if max_amp < 1e-4:
        return False, details, f"Degenerate silent file: peak amplitude {max_amp:.6f} < 1e-4"
    if max_amp > 1.0001:
        return False, details, f"Audio clipped: peak amplitude {max_amp:.4f} > 1.0"

    content_hash = hashlib.sha256(data.tobytes()).hexdigest()
    details["sha256"] = content_hash[:12]

    return True, details, "OK"


def run_verification(target_dir: Path, min_count: int = 20) -> int:
    """Executes the acceptance verification harness against target_dir."""
    print("=" * 80)
    print(" ANYCALL MILESTONE 1 ACCEPTANCE VERIFICATION HARNESS")
    print(f" Target Directory : {target_dir.resolve()}")
    print(f" Required Minimum : {min_count} uniform 3s 48kHz mono WAV segments")
    print("=" * 80)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"\n[FATAL FAIL] Directory does not exist: {target_dir}")
        return 1

    wav_files = sorted(list(target_dir.glob("*.wav")))
    total_files = len(wav_files)
    print(f"\nDiscovered {total_files} '.wav' files in target directory.\n")

    if total_files < min_count:
        print(f"[FAIL] Quantity check failed: found {total_files} files, minimum required is {min_count}.")
        return 1

    passed_files = 0
    failed_files = 0
    hashes: Set[str] = set()
    duplicate_count = 0

    print(f"{'#':<4} | {'Filename':<32} | {'SR (Hz)':<7} | {'Ch':<2} | {'Samples':<8} | {'Peak':<6} | {'Hash':<10} | {'Status'}")
    print("-" * 86)

    for i, fpath in enumerate(wav_files, 1):
        ok, details, reason = verify_single_file(fpath)
        if ok:
            passed_files += 1
            h = details.get("sha256", "")
            if h in hashes:
                duplicate_count += 1
                status = "WARN:DUP"
            else:
                hashes.add(h)
                status = "PASS"
            print(f"{i:<4} | {fpath.name:<32} | {details['samplerate']:<7} | {details['channels']:<2} | {details['frames']:<8} | {details.get('peak_amp', 0.0):<6.3f} | {h[:8]:<10} | {status}")
        else:
            failed_files += 1
            print(f"{i:<4} | {fpath.name:<32} | {'FAIL: ' + reason}")

    print("-" * 86)
    print(f"\nVerification Results:")
    print(f"  Total Inspected  : {total_files}")
    print(f"  Strict Passed    : {passed_files}")
    print(f"  Failed           : {failed_files}")
    print(f"  Unique Hashes    : {len(hashes)}")
    print(f"  Duplicates Found : {duplicate_count}")

    if failed_files > 0:
        print(f"\n[FAIL] {failed_files} files failed specification compliance checks.")
        return 1

    if passed_files < min_count:
        print(f"\n[FAIL] Only {passed_files} passed verification, required >= {min_count}.")
        return 1

    if len(hashes) < min_count // 2:
        print(f"\n[WARN/FAIL] Excessive duplication: only {len(hashes)} unique hashes found across {passed_files} segments.")
        return 1

    print("\n" + "=" * 80)
    print(" >>> [ALL GATES PASSED] MILESTONE 1 ACCEPTANCE CRITERIA VERIFIED <<< ")
    print("=" * 80 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AnyCall Milestone 1 Acceptance Criteria")
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=Path("data/processed/corvus_splendens"),
        help="Directory containing standardized 3s WAV segments (default: data/processed/corvus_splendens)"
    )
    parser.add_argument(
        "--min-count", "-n",
        type=int,
        default=20,
        help="Minimum required compliant segment count (default: 20)"
    )
    args = parser.parse_args()
    return run_verification(args.dir, args.min_count)


if __name__ == "__main__":
    sys.exit(main())
