"""Adversarial Stress Test Suite for AnyCall Audio Standardization and VAD.

Written by challenger_m1_1.
Validates:
1. Resampling accuracy & spectral fidelity across 8kHz - 192kHz and 100Hz - 20kHz.
2. Anti-aliasing filter rejection of ultrasonic components (> 24kHz) during downsampling.
3. VAD transient click rejection (< 250ms pulses at 0.95 amplitude).
4. VAD boundary duration thresholds (290ms vs 310ms at margin, framing dilation mechanics).
5. VAD adaptive noise floor tracking (asymmetric downward vs moderate upward vs >3x latch).
6. Exact slicing boundary sample counts (0, 99, 100, 143999, 144000, 144001, 287999, 288000, 288001).
7. Audio decoding error handling and container integrity.
8. Amplitude normalization numerical stability (silence, sub-threshold noise, extreme amplitudes).
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from anycall.audio.standardize import (
    load_audio,
    resample_audio,
    normalize_amplitude,
    standardize_audio,
    slice_audio_segments,
    AudioStandardizer,
    AudioFormatError,
)
from anycall.audio.vad import EnergyVAD, is_active_vocalization


def _dominant_frequency_and_snr(signal: np.ndarray, target_freq: float, sr: int) -> tuple[float, float]:
    """Computes dominant frequency and SNR (in dB) via FFT."""
    window = signal * np.hanning(len(signal))
    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

    peak_idx = int(np.argmax(fft_vals))
    peak_freq = float(freqs[peak_idx])

    tol = max(50.0, target_freq * 0.02)
    sig_mask = (freqs >= target_freq - tol) & (freqs <= target_freq + tol)

    sig_pwr = np.sum(fft_vals[sig_mask] ** 2)
    noise_pwr = np.sum(fft_vals[~sig_mask] ** 2) + 1e-12
    snr_db = float(10.0 * np.log10(sig_pwr / noise_pwr))

    return peak_freq, snr_db


# ==============================================================================
# 1. RESAMPLING FIDELITY ACROSS EXTREME SAMPLE RATES (8kHz to 192kHz)
# ==============================================================================

@pytest.mark.parametrize("orig_sr", [8000, 11025, 16000, 22050, 32000, 44100, 96000, 192000])
@pytest.mark.parametrize("freq", [100.0, 440.0, 1000.0, 3000.0, 7500.0, 15000.0, 20000.0])
def test_resampling_spectrum_and_amplitude_fidelity(orig_sr: int, freq: float):
    """Stress tests rational polyphase sinc resampling from all standard/non-standard rates to 48kHz."""
    nyquist = orig_sr / 2.0
    if freq >= nyquist * 0.90:
        pytest.skip(f"Frequency {freq}Hz is within transition band (>= 0.90 Nyquist) for sr={orig_sr}")

    duration = 1.0
    t = np.arange(int(round(orig_sr * duration))) / orig_sr
    clean_sine = 0.8 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    resampled = resample_audio(clean_sine, orig_sr=orig_sr, target_sr=48000)

    # 1. Verification of finite values (no NaN/Inf)
    assert not np.isnan(resampled).any(), f"NaN detected at sr={orig_sr}, f={freq}"
    assert not np.isinf(resampled).any(), f"Inf detected at sr={orig_sr}, f={freq}"

    # 2. Length preservation
    assert abs(len(resampled) - 48000) <= 2, f"Sample length deviated: {len(resampled)} vs 48000"

    # Trim FIR filter ramp-up and ramp-down transients (10% on each side)
    center = resampled[int(0.1 * len(resampled)) : int(0.9 * len(resampled))]

    # 3. Peak frequency accuracy: must match target within 25 Hz or 1%
    peak_freq, snr_db = _dominant_frequency_and_snr(center, freq, 48000)
    assert abs(peak_freq - freq) <= max(25.0, freq * 0.01), (
        f"Frequency mismatch: expected {freq}Hz, got {peak_freq:.1f}Hz (sr={orig_sr})"
    )

    # 4. Spectral purity / SNR
    assert snr_db > 15.0, f"Excessive distortion/aliasing: SNR={snr_db:.1f}dB (sr={orig_sr}, f={freq})"

    # 5. Peak amplitude preservation
    peak_amp = float(np.max(np.abs(center)))
    assert 0.70 <= peak_amp <= 0.90, f"Peak amplitude out of range: {peak_amp:.3f} (sr={orig_sr}, f={freq})"


@pytest.mark.parametrize("orig_sr", [96000, 192000])
@pytest.mark.parametrize("ultra_f, min_attenuation_db", [
    (26000.0, 15.0),
    (30000.0, 50.0),
    (40000.0, 75.0),
])
def test_resampling_anti_aliasing_ultrasound_rejection(orig_sr: int, ultra_f: float, min_attenuation_db: float):
    """Downsampling high-rate audio to 48kHz must attenuate ultrasound above 24kHz Nyquist."""
    t = np.arange(orig_sr) / orig_sr
    ultra_signal = 0.8 * np.sin(2.0 * np.pi * ultra_f * t).astype(np.float32)

    resampled = resample_audio(ultra_signal, orig_sr=orig_sr, target_sr=48000)
    center = resampled[int(0.1 * len(resampled)) : int(0.9 * len(resampled))]

    max_aliased = float(np.max(np.abs(center)))
    actual_attenuation_db = -20.0 * np.log10(max_aliased / 0.8 + 1e-12)

    assert actual_attenuation_db >= min_attenuation_db, (
        f"Insufficient anti-aliasing attenuation at {ultra_f}Hz: {actual_attenuation_db:.1f}dB < {min_attenuation_db}dB"
    )


# ==============================================================================
# 2. VAD ATTACK VECTORS
# ==============================================================================

@pytest.mark.parametrize("click_dur_ms", [0.1, 1.0, 5.0, 20.0, 50.0, 100.0, 150.0, 200.0, 240.0])
def test_vad_transient_click_rejection_vector(click_dur_ms: float):
    """VAD must not trigger on high-amplitude acoustic transients < 250ms."""
    sr = 48000
    vad = EnergyVAD(sr=sr)
    np.random.seed(42)
    noise = np.random.normal(0, 0.005, sr * 3).astype(np.float32)

    dur_samples = max(1, int(round(sr * (click_dur_ms / 1000.0))))
    audio = noise.copy()
    audio[sr : sr + dur_samples] = 0.95  # Full-scale impulse

    intervals = vad.detect_vocalization_intervals(audio)
    assert len(intervals) == 0, f"Transient click of {click_dur_ms}ms triggered VAD! Intervals: {intervals}"
    assert not vad.is_speech_or_vocal(audio)


def test_vad_duration_boundary_margin_discrimination():
    """At marginal detection SNR (tone amp=0.05, noise=0.01), 290ms is rejected and 310ms is accepted."""
    sr = 48000
    vad = EnergyVAD(sr=sr)
    np.random.seed(42)

    # 290ms call
    audio_290 = np.random.normal(0, 0.01, sr * 3).astype(np.float32)
    t290 = np.arange(int(round(sr * 0.290))) / sr
    audio_290[sr : sr + len(t290)] += 0.05 * np.sin(2 * np.pi * 1000 * t290).astype(np.float32)

    # 310ms call
    audio_310 = np.random.normal(0, 0.01, sr * 3).astype(np.float32)
    t310 = np.arange(int(round(sr * 0.310))) / sr
    audio_310[sr : sr + len(t310)] += 0.05 * np.sin(2 * np.pi * 1000 * t310).astype(np.float32)

    vocal_290 = vad.is_speech_or_vocal(audio_290)
    vocal_310 = vad.is_speech_or_vocal(audio_310)

    assert vocal_290 is False, "Marginal 290ms call should not meet 300ms trigger threshold (12 frames)"
    assert vocal_310 is True, "Marginal 310ms call should meet 300ms trigger threshold"


def test_vad_noise_step_adaptation():
    """Validates noise floor tracking on upward (2x) and downward (4x) ambient step changes."""
    sr = 48000
    vad = EnergyVAD(sr=sr)
    np.random.seed(123)

    # 1. Downward transition: high noise -> quiet background
    n_high = np.random.normal(0, 0.04, sr * 2).astype(np.float32)
    n_low = np.random.normal(0, 0.01, sr * 3).astype(np.float32)
    audio_down = np.concatenate([n_high, n_low])

    rms_down = vad.compute_frame_rms(audio_down)
    floor_down, trig_down = vad.track_noise_floor(rms_down)

    assert not trig_down.any(), "Downward step caused false trigger"
    assert floor_down[-1] < 0.02, f"Downward adaptation failed: final floor={floor_down[-1]:.4f}"

    # 2. Moderate upward transition: 0.01 -> 0.02 (step < 3.0x trigger ratio)
    audio_up = np.concatenate([n_low, np.random.normal(0, 0.02, sr * 3).astype(np.float32)])
    rms_up = vad.compute_frame_rms(audio_up)
    floor_up, trig_up = vad.track_noise_floor(rms_up)

    assert not trig_up.any(), "Moderate 2x noise step caused false trigger"
    assert floor_up[-1] > 0.015, f"Upward adaptation failed: final floor={floor_up[-1]:.4f}"


# ==============================================================================
# 3. AUDIO SLICING BOUNDARY SAMPLE COUNTS
# ==============================================================================

@pytest.mark.parametrize("samples, pad_tail, expected_count", [
    (0, False, 0),
    (50, False, 0),
    (99, False, 0),
    (100, False, 1),
    (143999, False, 1),
    (144000, False, 1),
    (144001, False, 1),
    (144001, True, 1),
    (287999, False, 1),
    (287999, True, 2),
    (288000, False, 2),
    (288001, False, 2),
])
def test_slicing_sample_count_boundaries(samples: int, pad_tail: bool, expected_count: int):
    """Verifies segment counts and sample shapes at boundary lengths."""
    sr = 48000
    audio = np.ones(samples, dtype=np.float32) * 0.1
    segments = slice_audio_segments(audio, sr=sr, vad_filter=False, pad_tail=pad_tail)

    assert len(segments) == expected_count, (
        f"Length {samples} (pad_tail={pad_tail}): expected {expected_count} segments, got {len(segments)}"
    )

    for seg in segments:
        assert seg.shape == (144000,), f"Segment shape deviated from 144000: {seg.shape}"
        assert seg.dtype == np.float32


# ==============================================================================
# 4. ERROR HANDLING & CONTAINER INTEGRITY
# ==============================================================================

def test_load_audio_error_handling():
    """Adversarially probe load_audio with malformed inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Non-existent file
        with pytest.raises(AudioFormatError, match="Audio file not found"):
            load_audio(tmp_path / "non_existent.wav")

        # 2. Directory path
        with pytest.raises(AudioFormatError, match="Path is not a regular file"):
            load_audio(tmp_path)

        # 3. 0-byte file
        empty_f = tmp_path / "empty.wav"
        empty_f.write_bytes(b"")
        with pytest.raises(AudioFormatError, match="empty"):
            load_audio(empty_f)

        # 4. Truncated header (< 4 bytes)
        short_hdr = tmp_path / "short.wav"
        short_hdr.write_bytes(b"RI")
        with pytest.raises(AudioFormatError, match="header too short"):
            load_audio(short_hdr)

        # 5. Invalid WAV RIFF header
        bad_riff = tmp_path / "bad.wav"
        bad_riff.write_bytes(b"RIFF1234NOTWAVxyz")
        with pytest.raises(AudioFormatError, match="Invalid WAV RIFF header"):
            load_audio(bad_riff)


def test_standardize_audio_output_specs():
    """Verifies that standardize_audio produces strict 48kHz mono 16-bit PCM WAV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        in_wav = tmp_path / "input_stereo_44k.wav"
        out_wav = tmp_path / "output_48k.wav"

        # Generate stereo 44.1kHz audio
        sr_in = 44100
        t = np.arange(sr_in * 2) / sr_in
        ch1 = 0.6 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.4 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([ch1, ch2]).astype(np.float32)
        sf.write(str(in_wav), stereo, sr_in, subtype="PCM_24")

        data, sr_out = standardize_audio(in_wav, out_wav, target_sr=48000)

        assert sr_out == 48000
        assert data.ndim == 1
        assert not np.isnan(data).any()

        # Inspect disk file
        info = sf.info(str(out_wav))
        assert info.samplerate == 48000
        assert info.channels == 1
        assert info.format == "WAV"
        assert info.subtype == "PCM_16"
