"""AnyCall Audio Standardization & Slicing Module.

Handles audio decoding, channel downmixing to mono, rational polyphase resampling
to 48 kHz, peak normalization, fixed-duration 3.0s windowing (144,000 samples),
adaptive energy VAD filtering, and writing 16-bit PCM WAV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from anycall.audio.vad import EnergyVAD, is_active_vocalization

logger = logging.getLogger("anycall.audio.standardize")


class AudioFormatError(Exception):
    """Raised when an audio file cannot be loaded, decoded, or is corrupted."""
    pass


def load_audio(
    input_path: Union[str, Path],
    target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Loads an audio file from disk, downmixes to mono float32, and optionally resamples.

    Args:
        input_path: Path to input audio file (WAV, MP3, FLAC, OGG).
        target_sr: Optional destination sampling rate in Hz.

    Returns:
        Tuple[np.ndarray, int]: (mono_audio_float32, sample_rate)

    Raises:
        AudioFormatError: If file is not found, empty, corrupted, or unsupported.
    """
    path = Path(input_path)
    if not path.exists():
        raise AudioFormatError(f"Audio file not found: {path}")

    if not path.is_file():
        raise AudioFormatError(f"Path is not a regular file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioFormatError(f"Audio file is empty (0 bytes): {path}")

    # Inspect header bytes for basic container verification
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if len(header) < 4:
            raise AudioFormatError(f"File header too short ({len(header)} bytes): {path}")

        # If extension is .wav, enforce RIFF/WAVE header checks
        if path.suffix.lower() == ".wav":
            if len(header) < 12:
                raise AudioFormatError(f"Truncated WAV header ({len(header)} bytes): {path}")
            if not (header[:4] == b"RIFF" and header[8:12] == b"WAVE"):
                raise AudioFormatError(f"Invalid WAV RIFF header: {path}")
    except AudioFormatError:
        raise
    except Exception as exc:
        raise AudioFormatError(f"Failed to inspect file header: {exc}") from exc

    try:
        data, orig_sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        # Attempt fallback to librosa if installed
        try:
            import librosa
            data, orig_sr = librosa.load(str(path), sr=None, mono=False)
            data = data.T
        except Exception:
            raise AudioFormatError(f"Failed to decode audio file {path}: {exc}") from exc

    # Channel downmix to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = np.ascontiguousarray(data, dtype=np.float32)

    # Sanitize NaN/Inf
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    if len(data) == 0:
        raise AudioFormatError(f"Audio file contains 0 audio samples: {path}")

    # Resample if requested and different
    if target_sr is not None and orig_sr != target_sr:
        data = resample_audio(data, orig_sr=orig_sr, target_sr=target_sr)
        return data, target_sr

    return data, orig_sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 48000
) -> np.ndarray:
    """High-fidelity rational polyphase sinc resampling via scipy.signal.resample_poly.

    Args:
        audio: 1D float32 numpy array.
        orig_sr: Source sampling rate.
        target_sr: Destination sampling rate (default: 48000).

    Returns:
        np.ndarray: Resampled 1D float32 array.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def normalize_amplitude(
    audio: np.ndarray,
    peak_norm: float = 0.95,
    eps: float = 1e-10
) -> np.ndarray:
    """Peak-normalizes audio array to [-peak_norm, peak_norm].

    Args:
        audio: 1D float32 numpy array.
        peak_norm: Target peak absolute magnitude (default: 0.95).
        eps: Epsilon to prevent division by zero.

    Returns:
        np.ndarray: Normalized 1D float32 array bounded in [-1.0, 1.0].
    """
    if len(audio) == 0:
        return audio.copy()

    max_val = float(np.max(np.abs(audio)))
    if max_val < 1e-6:
        # Near-zero silence: preserve without blowing up quantization noise
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    scale = peak_norm / (max_val + eps)
    normalized = audio * scale
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def standardize_audio(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_sr: int = 48000
) -> Tuple[np.ndarray, int]:
    """Converts input audio to standardized 48kHz mono float32 PCM WAV.

    Args:
        input_path: Path to source audio.
        output_path: Optional destination WAV path.
        target_sr: Target sampling rate (default: 48000).

    Returns:
        Tuple[np.ndarray, int]: (standardized_waveform, target_sr)

    Raises:
        AudioFormatError: If loading or processing fails.
    """
    audio, sr = load_audio(input_path, target_sr=target_sr)
    audio = normalize_amplitude(audio, peak_norm=0.95)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, target_sr, subtype="PCM_16", format="WAV")

    return audio, target_sr


def slice_audio_segments(
    audio: np.ndarray,
    sr: int = 48000,
    segment_duration: float = 3.0,
    hop_duration: float = 3.0,
    vad_filter: bool = True,
    pad_tail: bool = False
) -> List[np.ndarray]:
    """Splits continuous audio into uniform 3.0-second segments (144,000 samples at 48kHz).

    Optionally discards silent or low-energy segments using EnergyVAD.

    Args:
        audio: 1D float32 array.
        sr: Sample rate in Hz (default: 48000).
        segment_duration: Duration per slice in seconds (default: 3.0).
        hop_duration: Stride between slices in seconds (default: 3.0).
        vad_filter: Whether to discard silent segments (default: True).
        pad_tail: Whether to zero-pad trailing remainder to full segment length (default: False).

    Returns:
        List[np.ndarray]: List of 1D float32 arrays, each of length int(round(sr * segment_duration)).
    """
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise AudioFormatError(f"Expected 1D audio array, got shape {audio.shape}")

    target_samples = int(round(sr * segment_duration))
    hop_samples = int(round(sr * hop_duration))

    if len(audio) == 0:
        return []

    vad = EnergyVAD(sr=sr) if vad_filter else None

    # Case 1: Audio shorter than target duration
    if len(audio) < target_samples:
        if len(audio) < 100:
            # Sub-frame audio under minimum threshold
            return []
        padded = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
        if not vad_filter or vad.is_speech_or_vocal(padded):
            return [padded.astype(np.float32)]
        return []

    # Case 2: Full sliding window
    segments: List[np.ndarray] = []
    total_len = len(audio)
    start = 0

    while start + target_samples <= total_len:
        seg = audio[start : start + target_samples]
        if not vad_filter or vad.is_speech_or_vocal(seg):
            segments.append(seg.astype(np.float32))
        start += hop_samples

    # Remainder handling (when pad_tail is explicitly enabled)
    if pad_tail and start < total_len and (start + target_samples > total_len):
        tail = audio[start:]
        if len(tail) >= int(sr * 0.30):
            padded_tail = np.pad(tail, (0, target_samples - len(tail)), mode="constant")
            if not vad_filter or vad.is_speech_or_vocal(padded_tail):
                segments.append(padded_tail.astype(np.float32))

    return segments


class AudioStandardizer:
    """Batch audio processor and segmenter for AnyCall dataset ingestion."""

    def __init__(
        self,
        target_sr: int = 48000,
        segment_duration: float = 3.0,
        hop_duration: float = 3.0,
        vad_filter: bool = True,
        vad_noise_alpha: float = 0.01,
        vad_trigger_mult: float = 3.0,
        vad_min_trigger_ms: float = 300.0,
        normalize_peak: bool = True,
        pad_tail: bool = False,
    ):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.vad_filter = vad_filter
        self.normalize_peak = normalize_peak
        self.pad_tail = pad_tail
        self.vad = EnergyVAD(
            sr=target_sr,
            alpha=vad_noise_alpha,
            trigger_ratio=vad_trigger_mult,
            min_trigger_duration_ms=vad_min_trigger_ms,
        )

    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: Optional[str] = None,
        force: bool = False,
    ) -> List[Path]:
        """Standardizes and slices a single audio file, writing segments to output_dir."""
        in_p = Path(input_path)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        audio, sr = load_audio(in_p, target_sr=self.target_sr)
        if self.normalize_peak:
            audio = normalize_amplitude(audio, peak_norm=0.95)

        segments = slice_audio_segments(
            audio,
            sr=self.target_sr,
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            vad_filter=self.vad_filter,
            pad_tail=self.pad_tail,
        )

        base_stem = prefix or in_p.stem
        output_paths: List[Path] = []

        for idx, seg in enumerate(segments):
            seg_filename = f"{base_stem}_seg{idx:03d}.wav"
            seg_path = out_d / seg_filename

            if seg_path.exists() and not force:
                output_paths.append(seg_path)
                continue

            # Ensure strict 16-bit PCM WAV
            sf.write(str(seg_path), seg, self.target_sr, subtype="PCM_16", format="WAV")
            output_paths.append(seg_path)

        return output_paths

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        species_filter: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """Batch processes audio files in input_dir into standardized WAV segments."""
        in_d = Path(input_dir)
        out_d = Path(output_dir)

        subdirs = [d for d in in_d.iterdir() if d.is_dir()] if in_d.exists() else []
        if species_filter:
            subdirs = [d for d in subdirs if d.name.lower() == species_filter.lower()]

        if not subdirs and in_d.is_dir():
            subdirs = [in_d]

        total_segments = 0

        for sp_dir in subdirs:
            species_slug = sp_dir.name
            target_species_dir = out_d if sp_dir == in_d else out_d / species_slug
            target_species_dir.mkdir(parents=True, exist_ok=True)

            audio_files = sorted([
                f for f in sp_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".aac"}
            ])

            manifest_entries: List[Dict[str, Any]] = []

            for af in audio_files:
                try:
                    seg_paths = self.process_file(
                        af,
                        target_species_dir,
                        prefix=f"{species_slug}_{af.stem}" if sp_dir != in_d else None,
                        force=force,
                    )
                    for spath in seg_paths:
                        info = sf.info(str(spath))
                        data, _ = sf.read(str(spath), dtype="float32")
                        h = hashlib.sha256(data.tobytes()).hexdigest()
                        manifest_entries.append({
                            "segment_filename": spath.name,
                            "source_file": af.name,
                            "channels": info.channels,
                            "sample_rate": info.samplerate,
                            "num_samples": info.frames,
                            "duration_sec": info.duration,
                            "format": info.format,
                            "subtype": info.subtype,
                            "peak_amplitude": float(np.max(np.abs(data))),
                            "rms_energy": float(np.sqrt(np.mean(data ** 2))),
                            "sha256": h,
                        })
                    total_segments += len(seg_paths)
                except Exception as exc:
                    logger.warning(f"Skipping corrupt or unreadable file {af}: {exc}")

            # Write processing manifest
            manifest_file = target_species_dir / "processing_manifest.json"
            manifest_data = {
                "species_slug": species_slug,
                "target_sample_rate": self.target_sr,
                "target_duration_sec": self.segment_duration,
                "target_samples": int(round(self.target_sr * self.segment_duration)),
                "total_segments_produced": len(manifest_entries),
                "segments": manifest_entries,
            }
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

        return total_segments


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for audio standardization."""
    parser = argparse.ArgumentParser(
        prog="python -m anycall.audio.standardize",
        description="AnyCall Audio Standardization & VAD Segmenter (48kHz, 16-bit Mono PCM, 3.0s WAV)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input directory containing raw audio files (default: data/raw)"
    )
    group.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Single audio file to process"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for standardized WAV segments (default: data/processed)"
    )
    parser.add_argument(
        "--species", "-s",
        type=str,
        default=None,
        help="Filter subfolder by species slug (e.g. 'corvus_splendens')"
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=48000,
        help="Standard sampling rate in Hz (default: 48000)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Standard segment duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--hop-duration",
        type=float,
        default=3.0,
        help="Sliding window hop duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable RMS energy VAD filter (default: True)"
    )
    parser.add_argument(
        "--vad-floor-alpha",
        type=float,
        default=0.01,
        help="VAD noise floor tracking rate alpha (default: 0.01)"
    )
    parser.add_argument(
        "--vad-trigger-mult",
        type=float,
        default=3.0,
        help="VAD RMS trigger multiplier over noise floor (default: 3.0)"
    )
    parser.add_argument(
        "--vad-min-trigger-ms",
        type=float,
        default=300.0,
        help="VAD minimum sustained trigger in ms (default: 300.0)"
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable peak normalization to 0.95 (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed segments"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose DEBUG logging"
    )
    return parser


def main() -> int:
    """CLI entry point for audio standardization."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    try:
        standardizer = AudioStandardizer(
            target_sr=args.target_sr,
            segment_duration=args.duration,
            hop_duration=args.hop_duration,
            vad_filter=args.vad,
            vad_noise_alpha=args.vad_floor_alpha,
            vad_trigger_mult=args.vad_trigger_mult,
            vad_min_trigger_ms=args.vad_min_trigger_ms,
            normalize_peak=args.normalize,
        )

        if args.input_file:
            segments = standardizer.process_file(args.input_file, args.output_dir, force=args.force)
            logger.info(f"Generated {len(segments)} segments from {args.input_file}")
            return 0 if len(segments) > 0 else 3
        else:
            total = standardizer.process_directory(
                args.input_dir,
                args.output_dir,
                species_filter=args.species,
                force=args.force,
            )
            logger.info(f"Batch processing completed: {total} total 3-second segments created.")
            return 0 if total > 0 else 3
    except Exception as exc:
        logger.error(f"Audio standardization failed: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
