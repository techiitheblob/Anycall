"""AnyCall Voice Activity Detection (VAD) & Adaptive Noise Floor Module.

Calculates short-term 50ms RMS frame energies, tracks an adaptive asymmetric
noise floor (alpha=0.01), and detects vocalization events exceeding 3.0x noise floor
for at least 300 ms.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class EnergyVAD:
    """Adaptive Energy Voice Activity Detector with Exponential Noise Floor Tracking.

    Operates on 50ms sliding RMS energy windows, updates a dynamic noise floor
    estimator with adaptation coefficient alpha=0.01 during non-vocal periods,
    and detects vocalization events when RMS > trigger_ratio * noise_floor sustained
    for >= min_trigger_duration_ms (300 ms).
    """

    def __init__(
        self,
        sr: Optional[int] = None,
        sample_rate: Optional[int] = None,
        frame_duration_ms: float = 50.0,
        hop_duration_ms: float = 25.0,
        alpha: Optional[float] = None,
        noise_alpha: Optional[float] = None,
        alpha_down: float = 0.05,
        trigger_ratio: Optional[float] = None,
        trigger_multiplier: Optional[float] = None,
        min_trigger_duration_ms: Optional[float] = None,
        min_trigger_ms: Optional[float] = None,
        floor_min: float = 1e-4,
    ):
        self.sr = int(sample_rate if sample_rate is not None else (sr if sr is not None else 48000))
        self.frame_len = max(1, int(round(self.sr * (frame_duration_ms / 1000.0))))
        self.hop_len = max(1, int(round(self.sr * (hop_duration_ms / 1000.0))))
        self.alpha = float(noise_alpha if noise_alpha is not None else (alpha if alpha is not None else 0.01))
        self.alpha_down = float(alpha_down)
        self.trigger_ratio = float(trigger_multiplier if trigger_multiplier is not None else (trigger_ratio if trigger_ratio is not None else 3.0))
        min_ms = min_trigger_ms if min_trigger_ms is not None else (min_trigger_duration_ms if min_trigger_duration_ms is not None else 300.0)
        self.min_consecutive_frames = max(1, int(np.ceil(min_ms / hop_duration_ms)))
        self.floor_min = float(floor_min)

    def compute_frame_rms(self, audio: np.ndarray) -> np.ndarray:
        """Computes sliding window RMS energy for 1D float audio array."""
        if len(audio) == 0:
            return np.empty(0, dtype=np.float32)

        if len(audio) < self.frame_len:
            audio = np.pad(audio, (0, self.frame_len - len(audio)), mode="constant")

        num_frames = max(1, 1 + (len(audio) - self.frame_len) // self.hop_len)
        rms_values = np.zeros(num_frames, dtype=np.float32)

        for m in range(num_frames):
            start = m * self.hop_len
            frame = audio[start : start + self.frame_len]
            rms = np.sqrt(np.mean(frame ** 2, dtype=np.float64) + 1e-12)
            rms_values[m] = float(rms)

        return rms_values

    def track_noise_floor(
        self, rms_frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tracks the adaptive noise floor and identifies triggered frames.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - noise_floor: Array of tracked noise floor values.
                - is_triggered: Boolean mask where RMS > trigger_ratio * noise_floor.
        """
        num_frames = len(rms_frames)
        noise_floor = np.zeros(num_frames, dtype=np.float32)
        is_triggered = np.zeros(num_frames, dtype=bool)

        if num_frames == 0:
            return noise_floor, is_triggered

        # Robust initialization from 15th percentile of first ~40 frames (~1.0s)
        init_window = min(num_frames, 40)
        p15 = float(np.percentile(rms_frames[:init_window], 15))
        current_floor = max(self.floor_min, p15)

        for m in range(num_frames):
            rms_m = float(rms_frames[m])

            if rms_m > (self.trigger_ratio * current_floor):
                is_triggered[m] = True
                # Freeze floor during active vocalization
            elif rms_m < current_floor:
                # Fast downward adaptation when ambient noise subsides
                current_floor = (1.0 - self.alpha_down) * current_floor + self.alpha_down * rms_m
            else:
                # Standard stationary background noise tracking
                current_floor = (1.0 - self.alpha) * current_floor + self.alpha * rms_m

            current_floor = max(self.floor_min, current_floor)
            noise_floor[m] = current_floor

        return noise_floor, is_triggered

    def detect_vocalization_intervals(
        self, audio: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Identifies active vocalization time intervals [(start_sec, end_sec), ...].

        Only intervals where triggered state is sustained for at least
        min_trigger_duration_ms are retained.
        """
        if len(audio) == 0:
            return []

        # Peak amplitude check: if entire audio peak is near silence, fast discard
        if float(np.max(np.abs(audio))) < 1e-4:
            return []

        rms_frames = self.compute_frame_rms(audio)
        _, is_triggered = self.track_noise_floor(rms_frames)

        intervals: List[Tuple[float, float]] = []
        in_event = False
        run_start = 0

        for m in range(len(is_triggered)):
            if is_triggered[m] and not in_event:
                in_event = True
                run_start = m
            elif not is_triggered[m] and in_event:
                in_event = False
                run_len = m - run_start
                if run_len >= self.min_consecutive_frames:
                    t_start = (run_start * self.hop_len) / self.sr
                    t_end = min(len(audio) / self.sr, ((m * self.hop_len) + self.frame_len) / self.sr)
                    intervals.append((t_start, t_end))

        if in_event:
            run_len = len(is_triggered) - run_start
            if run_len >= self.min_consecutive_frames:
                t_start = (run_start * self.hop_len) / self.sr
                t_end = len(audio) / self.sr
                intervals.append((t_start, t_end))

        return intervals

    def is_speech_or_vocal(self, audio: np.ndarray) -> bool:
        """Returns True if the audio clip contains at least one sustained vocalization event."""
        intervals = self.detect_vocalization_intervals(audio)
        return len(intervals) > 0

    def process_segment(self, audio: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Processes a segment and returns (is_vocal, metrics_dict)."""
        if len(audio) == 0:
            return False, {"peak_rms": 0.0, "noise_floor": self.floor_min, "intervals": []}

        rms_frames = self.compute_frame_rms(audio)
        noise_floors, is_triggered = self.track_noise_floor(rms_frames)
        intervals = self.detect_vocalization_intervals(audio)
        is_vocal = len(intervals) > 0

        peak_rms = float(np.max(rms_frames)) if len(rms_frames) > 0 else 0.0
        avg_floor = float(np.mean(noise_floors)) if len(noise_floors) > 0 else self.floor_min

        metrics = {
            "peak_rms": peak_rms,
            "noise_floor": avg_floor,
            "trigger_ratio": self.trigger_ratio,
            "intervals": intervals,
            "frame_count": len(rms_frames),
            "triggered_frame_count": int(np.sum(is_triggered)),
        }
        return is_vocal, metrics


def is_active_vocalization(audio: np.ndarray, sr: int = 48000) -> bool:
    """Convenience functional wrapper for single audio array."""
    return EnergyVAD(sr=sr).is_speech_or_vocal(audio)


# Forward-reference slice_audio_segments so importing from anycall.audio.vad works
def slice_audio_segments(*args, **kwargs):
    from anycall.audio.standardize import slice_audio_segments as _slice
    return _slice(*args, **kwargs)
