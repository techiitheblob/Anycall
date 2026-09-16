"""Pure-Python / NumPy Synthetic Audio Generator for AnyCall E2E Testing.

Generates deterministic synthetic audio signals (pure tones, harmonic chirps,
broadband noise, impulsive transients, and animal vocalizations) and provides
WAV serialization and corrupted file generation for testing without external files.
"""

import io
import math
import os
from pathlib import Path
import struct
from typing import List, Optional, Tuple, Union
import wave

import numpy as np


def generate_pure_tone(
    freq: float = 2000.0,
    duration: float = 3.0,
    sr: int = 48000,
    amplitude: float = 0.8,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a deterministic pure sine tone.

    Args:
        freq: Frequency in Hertz.
        duration: Duration in seconds.
        sr: Sample rate in Hertz (default: 48,000).
        amplitude: Peak amplitude in [0.0, 1.0].
        phase: Starting phase in radians.

    Returns:
        1D float32 numpy array of shape (int(sr * duration),).
    """
    n_samples = int(round(sr * duration))
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    signal = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return signal.astype(np.float32)


def generate_harmonic_chirp(
    start_freq: float = 1000.0,
    end_freq: float = 4000.0,
    harmonics: int = 4,
    duration: float = 3.0,
    sr: int = 48000,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a frequency-modulated harmonic chirp.

    Simulates avian calls (e.g. whistles or caws) using a linear frequency
    sweep with integer harmonic overtones.

    Args:
        start_freq: Fundamental starting frequency in Hz.
        end_freq: Fundamental ending frequency in Hz.
        harmonics: Number of harmonics to superimpose.
        duration: Duration in seconds.
        sr: Sample rate in Hz.
        amplitude: Peak amplitude in [0.0, 1.0].

    Returns:
        1D float32 numpy array.
    """
    n_samples = int(round(sr * duration))
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    
    # Linear chirp phase: phi(t) = 2*pi*(f0*t + 0.5*(f1-f0)/T * t^2)
    chirp_rate = (end_freq - start_freq) / max(duration, 1e-6)
    base_phase = 2.0 * np.pi * (start_freq * t + 0.5 * chirp_rate * (t ** 2))

    signal = np.zeros(n_samples, dtype=np.float64)
    total_weight = 0.0
    for h in range(1, harmonics + 1):
        weight = 1.0 / h
        signal += weight * np.sin(h * base_phase)
        total_weight += weight

    signal = (signal / total_weight) * amplitude
    return signal.astype(np.float32)


def generate_broadband_noise(
    duration: float = 3.0,
    sr: int = 48000,
    noise_type: str = "white",
    amplitude: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Generate broadband synthetic noise.

    Args:
        duration: Duration in seconds.
        sr: Sample rate in Hz.
        noise_type: 'white', 'pink', or 'brown'.
        amplitude: Root-mean-square amplitude scaling.
        seed: Random seed for exact determinism.

    Returns:
        1D float32 numpy array.
    """
    n_samples = int(round(sr * duration))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples).astype(np.float64)

    if noise_type == "white":
        signal = white
    elif noise_type == "pink":
        # Approximate 1/f noise via simple IIR filter poles
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        # Pure numpy recursion for deterministic IIR without scipy
        signal = np.zeros(n_samples, dtype=np.float64)
        for n in range(n_samples):
            for k in range(len(b)):
                if n - k >= 0:
                    signal[n] += b[k] * white[n - k]
            for k in range(1, len(a)):
                if n - k >= 0:
                    signal[n] -= a[k] * signal[n - k]
    elif noise_type == "brown":
        # Cumulative integration for 1/f^2 noise
        signal = np.cumsum(white)
        signal -= np.mean(signal)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Expected 'white', 'pink', or 'brown'.")

    # Normalize to peak amplitude
    max_val = np.max(np.abs(signal))
    if max_val > 1e-9:
        signal = (signal / max_val) * amplitude
    else:
        signal = np.zeros_like(signal)

    return signal.astype(np.float32)


def generate_transient_pulse(
    pulse_rate: float = 15.0,
    pulse_width_ms: float = 10.0,
    carrier_freq: float = 5000.0,
    duration: float = 3.0,
    sr: int = 48000,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate periodic impulsive transients.

    Simulates insect stridulations (clicks) or acoustic snapping.

    Args:
        pulse_rate: Repetition rate in Hz (pulses per second).
        pulse_width_ms: Width of each pulse in milliseconds.
        carrier_freq: High-frequency carrier oscillation in Hz.
        duration: Duration in seconds.
        sr: Sample rate in Hz.
        amplitude: Peak amplitude in [0.0, 1.0].

    Returns:
        1D float32 numpy array.
    """
    n_samples = int(round(sr * duration))
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)

    pulse_period = 1.0 / max(pulse_rate, 1e-3)
    pulse_width_sec = pulse_width_ms / 1000.0
    mod_t = np.fmod(t, pulse_period)

    # Rectangular window smoothed with half-cosine edge
    envelope = np.where(mod_t < pulse_width_sec, np.sin(np.pi * mod_t / pulse_width_sec), 0.0)
    carrier = np.sin(2.0 * np.pi * carrier_freq * t)
    signal = amplitude * envelope * carrier

    return signal.astype(np.float32)


def generate_animal_call(
    taxon: str = "aves",
    duration: float = 3.0,
    sr: int = 48000,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic synthetic animal call for a specific taxonomic group.

    Args:
        taxon: One of 'aves', 'bird', 'insecta', 'insect', 'amphibia', 'frog', 'mammalia', 'mammal'.
        duration: Duration in seconds (default: 3.0).
        sr: Sample rate in Hz (default: 48,000).
        seed: Random seed for noise and modulation.

    Returns:
        1D float32 numpy array normalized in [-1.0, 1.0].
    """
    tax = taxon.lower().strip()
    n_samples = int(round(sr * duration))
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)

    if tax in ("aves", "bird"):
        # Frequency-modulated whistle with crow/songbird harmonic structure (1.5 kHz - 4.5 kHz)
        f0 = 2200.0
        fm = 3.0  # 3 Hz vibrato
        f_dev = 600.0
        phase = 2.0 * np.pi * (f0 * t - (f_dev / fm) * np.cos(2.0 * np.pi * fm * t))
        
        # 3 harmonics with avian envelope
        harmonic_sig = (
            1.00 * np.sin(phase) +
            0.50 * np.sin(2.0 * phase) +
            0.25 * np.sin(3.0 * phase)
        )
        # Add call envelope: periodic 400ms calls every 800ms
        call_mod = np.fmod(t, 0.8)
        call_env = np.where(call_mod < 0.45, np.sin(np.pi * call_mod / 0.45) ** 2, 0.0)
        signal = harmonic_sig * call_env

    elif tax in ("insecta", "insect"):
        # High-frequency broadband stridulation: 6.5 kHz carrier with 25 Hz rapid pulses
        pulse_period = 1.0 / 25.0
        pulse_mod = np.fmod(t, pulse_period)
        pulse_env = np.where(pulse_mod < 0.015, np.sin(np.pi * pulse_mod / 0.015), 0.0)
        carrier = np.sin(2.0 * np.pi * 6500.0 * t) + 0.3 * np.sin(2.0 * np.pi * 9200.0 * t)
        signal = pulse_env * carrier

    elif tax in ("amphibia", "frog"):
        # Low-frequency resonant pulses: 450 Hz with fast exponential decay (croak)
        croak_period = 0.5  # croak every 500 ms
        croak_mod = np.fmod(t, croak_period)
        croak_env = np.where(croak_mod < 0.3, np.exp(-12.0 * croak_mod), 0.0)
        carrier = np.sin(2.0 * np.pi * 450.0 * t) + 0.5 * np.sin(2.0 * np.pi * 900.0 * t)
        signal = croak_env * carrier

    elif tax in ("mammalia", "mammal"):
        # Wideband chatter / squirrel alarm call: 1.2 kHz - 3.5 kHz chatter
        burst_period = 0.25
        burst_mod = np.fmod(t, burst_period)
        burst_env = np.where(burst_mod < 0.12, np.sin(np.pi * burst_mod / 0.12), 0.0)
        carrier = (
            0.8 * np.sin(2.0 * np.pi * 1400.0 * t) +
            0.5 * np.sin(2.0 * np.pi * 2800.0 * t) +
            0.3 * np.sin(2.0 * np.pi * 4200.0 * t)
        )
        signal = burst_env * carrier
    else:
        raise ValueError(f"Unknown taxon '{taxon}'. Expected aves, insecta, amphibia, or mammalia.")

    # Add slight background ambience
    rng = np.random.default_rng(seed)
    noise = 0.01 * rng.standard_normal(n_samples)
    signal = signal + noise

    # Normalize to [-0.85, 0.85]
    peak = np.max(np.abs(signal))
    if peak > 1e-6:
        signal = (signal / peak) * 0.85

    return signal.astype(np.float32)


def generate_silence(duration: float = 3.0, sr: int = 48000) -> np.ndarray:
    """Generate exact digital silence (all zeros).

    Args:
        duration: Duration in seconds.
        sr: Sample rate in Hz.

    Returns:
        1D float32 numpy array of zeros.
    """
    n_samples = int(round(sr * duration))
    return np.zeros(n_samples, dtype=np.float32)


def generate_dc_offset(
    offset: float = 0.5,
    duration: float = 3.0,
    sr: int = 48000,
    base_audio: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate audio with constant DC offset bias.

    Args:
        offset: Constant offset value added.
        duration: Duration in seconds if base_audio is None.
        sr: Sample rate in Hz.
        base_audio: Optional underlying audio array.

    Returns:
        1D float32 numpy array with DC offset.
    """
    if base_audio is not None:
        signal = base_audio.copy()
    else:
        signal = generate_pure_tone(freq=1000.0, duration=duration, sr=sr, amplitude=0.4)
    signal = signal + float(offset)
    return signal.astype(np.float32)


def generate_clipped_audio(
    freq: float = 1000.0,
    duration: float = 3.0,
    sr: int = 48000,
    gain: float = 10.0,
) -> np.ndarray:
    """Generate heavily saturated / hard-clipped audio (square-wave overflow).

    Args:
        freq: Frequency of the driving sinusoid in Hz.
        duration: Duration in seconds.
        sr: Sample rate in Hz.
        gain: Overdrive gain applied prior to hard clipping.

    Returns:
        1D float32 numpy array clipped at [-1.0, 1.0].
    """
    clean = generate_pure_tone(freq=freq, duration=duration, sr=sr, amplitude=1.0)
    overdriven = clean * float(gain)
    clipped = np.clip(overdriven, -1.0, 1.0)
    return clipped.astype(np.float32)


# ==============================================================================
# WAV File Serialization and Corrupted File Generators
# ==============================================================================

def save_wav_file(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = 48000,
    bit_depth: int = 16,
) -> Path:
    """Save a 1D or 2D float32 numpy array as a standard WAV PCM file.

    Uses pure standard-library `wave` module without external dependencies.

    Args:
        audio: 1D mono or 2D (samples, channels) numpy array scaled to [-1.0, 1.0].
        file_path: Destination file path.
        sr: Sample rate in Hz.
        bit_depth: 16 (default) or 8.

    Returns:
        Path object of written file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 1:
        n_channels = 1
        data = audio_arr.reshape(-1, 1)
    elif audio_arr.ndim == 2:
        n_channels = audio_arr.shape[1]
        data = audio_arr
    else:
        raise ValueError(f"Audio array must be 1D or 2D, got shape {audio_arr.shape}")

    # Clip to valid float range
    data = np.clip(data, -1.0, 1.0)

    if bit_depth == 16:
        pcm_data = (data * 32767.0).astype(np.int16).tobytes()
        sampwidth = 2
    elif bit_depth == 8:
        # 8-bit unsigned PCM: 0 to 255 with 128 as zero
        pcm_data = ((data + 1.0) * 127.5).astype(np.uint8).tobytes()
        sampwidth = 1
    else:
        raise ValueError(f"Unsupported bit_depth {bit_depth}. Only 8 and 16 are supported.")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(pcm_data)

    return path


def read_wav_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Read a standard WAV file into a float32 numpy array and sample rate.

    Args:
        file_path: Path to WAV file.

    Returns:
        Tuple of (audio_array, sample_rate).
        If mono, audio_array is shape (N,); if stereo/multichannel, shape is (N, C).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 1:
        arr = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels)

    return arr, sr


def create_multichannel_wav_file(
    channels: List[np.ndarray],
    file_path: Union[str, Path],
    sr: int = 48000,
) -> Path:
    """Create a multi-channel (e.g. stereo) WAV file from a list of 1D arrays.

    Args:
        channels: List of 1D numpy arrays of equal length.
        file_path: Target output path.
        sr: Sample rate.

    Returns:
        Path of written WAV file.
    """
    if not channels:
        raise ValueError("channels list cannot be empty")
    min_len = min(len(c) for c in channels)
    stacked = np.column_stack([c[:min_len] for c in channels])
    return save_wav_file(stacked, file_path, sr=sr, bit_depth=16)


def create_corrupted_wav_file(
    file_path: Union[str, Path],
    corruption_type: str = "truncated_header",
) -> Path:
    """Create an intentionally corrupted or degenerate audio file for boundary testing.

    Args:
        file_path: Destination path.
        corruption_type:
            - 'zero_byte': File with 0 bytes.
            - 'truncated_header': Only 12 bytes of RIFF header.
            - 'invalid_riff': Missing 'RIFF' magic bytes (e.g. 'ABCD').
            - 'corrupted_fmt': Malformed 'fmt ' chunk.
            - 'text_file': Plain text renamed to .wav.
            - 'random_garbage': 512 bytes of random binary noise.

    Returns:
        Path of created file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if corruption_type == "zero_byte":
        path.write_bytes(b"")
    elif corruption_type == "truncated_header":
        path.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
    elif corruption_type == "invalid_riff":
        path.write_bytes(b"ABCD\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00")
    elif corruption_type == "corrupted_fmt":
        # RIFF header with invalid format chunk size
        header = bytearray(b"RIFF\x40\x00\x00\x00WAVEfmt \x00\x00\x00\x00")
        path.write_bytes(header)
    elif corruption_type == "text_file":
        path.write_text("This is not a WAV file. It is plain UTF-8 text.", encoding="utf-8")
    elif corruption_type == "random_garbage":
        rng = np.random.default_rng(1234)
        garbage = rng.bytes(1024)
        path.write_bytes(garbage)
    else:
        raise ValueError(f"Unknown corruption_type: '{corruption_type}'")

    return path
