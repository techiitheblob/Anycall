"""Unit tests for AnyCall Audio Subsystem (Milestone 1).

Covers:
- Audio standardization (resampling, mono downmix, peak normalization, PCM_16 WAV writing).
- Audio slicing and padding.
- Adaptive energy VAD and noise floor tracking.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from anycall.audio.standardize import (
    AudioFormatError,
    load_audio,
    normalize_amplitude,
    resample_audio,
    slice_audio_segments,
    standardize_audio,
)
from anycall.audio.vad import EnergyVAD, is_active_vocalization


def test_standardize_audio_resampling(tmp_path):
    """Verifies that arbitrary sample rates are resampled to target 48,000 Hz."""
    sr_orig = 44100
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    in_file = tmp_path / "test_44k.wav"
    sf.write(str(in_file), sine, sr_orig)

    audio, sr = standardize_audio(in_file, target_sr=48000)
    assert sr == 48000
    assert audio.ndim == 1
    assert len(audio) == 48000
    assert np.max(np.abs(audio)) > 0.4


def test_standardize_audio_downmix(tmp_path):
    """Verifies that 2-channel stereo audio downmixes to 1-channel mono."""
    sr = 48000
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 0.6  # Left
    stereo[:, 1] = 0.2  # Right
    in_file = tmp_path / "test_stereo.wav"
    sf.write(str(in_file), stereo, sr)

    audio, sr_out = standardize_audio(in_file, target_sr=48000)
    assert sr_out == 48000
    assert audio.ndim == 1
    assert len(audio) == sr


def test_slice_audio_exact_length():
    """Verifies that non-overlapping 3.0s slicing on 7.5s audio extracts 2 full slices."""
    sr = 48000
    audio = np.random.uniform(-0.5, 0.5, int(sr * 7.5)).astype(np.float32)
    segments = slice_audio_segments(audio, sr=sr, segment_duration=3.0, hop_duration=3.0, vad_filter=False)
    assert len(segments) == 2
    for seg in segments:
        assert len(seg) == 144000
        assert seg.dtype == np.float32


def test_slice_audio_short_padding():
    """Verifies that audio shorter than 3.0s is zero-padded to exactly 144,000 samples."""
    sr = 48000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    chirp = 0.8 * np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    segments = slice_audio_segments(chirp, sr=sr, segment_duration=3.0, vad_filter=False)
    assert len(segments) == 1
    assert len(segments[0]) == 144000
    assert np.all(segments[0][int(sr * 1.2):] == 0.0)


def test_vad_silence_rejection():
    """Verifies that pure digital silence is discarded by EnergyVAD."""
    vad = EnergyVAD(sample_rate=48000, frame_duration_ms=50.0, noise_alpha=0.01, trigger_multiplier=3.0)
    silence = np.zeros(48000 * 3, dtype=np.float32)
    triggered, metrics = vad.process_segment(silence)
    assert not triggered


def test_vad_active_call_trigger():
    """Verifies that a vocalization event exceeding trigger ratio is detected."""
    vad = EnergyVAD(sample_rate=48000, frame_duration_ms=50.0, noise_alpha=0.01, trigger_multiplier=3.0)
    t = np.linspace(0, 3.0, 48000 * 3, endpoint=False)
    audio = 0.005 * np.random.randn(len(t)).astype(np.float32)
    call_idx = slice(48000, int(48000 * 1.6))
    audio[call_idx] += 0.5 * np.sin(2 * np.pi * 1500 * t[call_idx]).astype(np.float32)

    triggered, metrics = vad.process_segment(audio)
    assert triggered
    assert metrics["peak_rms"] > metrics["noise_floor"] * 3.0


def test_vad_transient_click_rejection():
    """Verifies that impulsive transients shorter than 300ms do not trigger vocalization."""
    vad = EnergyVAD(sample_rate=48000, frame_duration_ms=50.0, min_trigger_duration_ms=300.0)
    audio = np.zeros(48000 * 2, dtype=np.float32)
    # 50ms transient burst
    click_len = int(48000 * 0.05)
    audio[1000 : 1000 + click_len] = 0.8 * np.sin(2 * np.pi * 3000 * np.linspace(0, 0.05, click_len))
    assert not vad.is_speech_or_vocal(audio)


def test_adaptive_noise_tracking():
    """Verifies that the noise floor estimator tracks slowly rising background noise."""
    vad = EnergyVAD(sample_rate=48000, frame_duration_ms=50.0, alpha=0.05)
    # Rising ambient noise envelope
    t = np.linspace(0, 4.0, 48000 * 4, endpoint=False)
    ramp = np.linspace(0.002, 0.01, len(t))
    noise = ramp * np.random.randn(len(t)).astype(np.float32)

    rms_frames = vad.compute_frame_rms(noise)
    floors, triggered = vad.track_noise_floor(rms_frames)
    assert floors[-1] > floors[0]


def test_corrupted_file_raises_error(tmp_path):
    """Verifies that non-audio files raise AudioFormatError."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_text("NOT A REAL WAV FILE CONTENT")
    with pytest.raises(AudioFormatError):
        standardize_audio(corrupt_file)


def test_wav_pcm16_output(tmp_path):
    """Verifies that written standardized audio strictly complies with 16-bit PCM WAV."""
    in_file = tmp_path / "in.wav"
    out_file = tmp_path / "out.wav"
    audio_orig = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000)).astype(np.float32)
    sf.write(str(in_file), audio_orig, 48000)

    arr, sr = standardize_audio(in_file, output_path=out_file, target_sr=48000)
    assert out_file.exists()
    info = sf.info(str(out_file))
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.samplerate == 48000
    assert info.channels == 1
