"""AnyCall Unified Abstract Base Class for Acoustic Embedding Backbones.

Location: anycall/embeddings/base.py
Enforces unified interface, standardized resampling, duration handling,
and strict unit L2-normalization for all embedding models.
"""
from __future__ import annotations

import abc
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class BaseAudioEmbeddingBackbone(abc.ABC):
    """Abstract base class for all AnyCall acoustic embedding backbones.

    Enforces unified interface, standardized resampling, duration handling,
    and strict L2-normalization across all backbones (BirdNET, Perch, PANNs, Mock).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """String identifier for the backbone (e.g. 'birdnet', 'perch', 'panns', 'mock')."""
        pass

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality D of the output feature vector."""
        pass

    @property
    @abc.abstractmethod
    def target_sample_rate(self) -> int:
        """Target audio sampling rate in Hz required by the model frontend."""
        pass

    @property
    def target_duration_seconds(self) -> float:
        """Expected duration of audio input in seconds (default: 3.0s)."""
        return 3.0

    @abc.abstractmethod
    def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
        """Model-specific embedding extraction implementation.

        Args:
            waveform: 1D float32 numpy array of length
                target_sample_rate * target_duration_seconds.

        Returns:
            Raw embedding numpy array of shape (embedding_dim,) or (1, embedding_dim).
        """
        pass

    def resample_waveform(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Fast, high-fidelity polyphase rational factor resampling using scipy.signal.resample_poly.

        Args:
            waveform: 1D float32 audio waveform.
            orig_sr: Original sampling rate in Hz.

        Returns:
            Resampled 1D float32 audio waveform at self.target_sample_rate.
        """
        if orig_sr == self.target_sample_rate:
            return waveform.astype(np.float32)
        gcd = math.gcd(orig_sr, self.target_sample_rate)
        up = self.target_sample_rate // gcd
        down = orig_sr // gcd
        resampled = resample_poly(waveform, up, down)
        return resampled.astype(np.float32)

    def preprocess_audio(
        self, audio: Union[str, Path, np.ndarray], sr: int = 48000
    ) -> np.ndarray:
        """Loads and standardizes input audio to a 1D float32 array of target length and sample rate.

        Performs:
        1. Loading from path (if str/Path) using soundfile.
        2. Channel downmixing to mono (mean across channels).
        3. Polyphase resampling to target_sample_rate if needed.
        4. Symmetrical padding or center-cropping to target_duration_seconds.

        Args:
            audio: File path or raw audio waveform array.
            sr: Sampling rate of input if numpy array is passed (default: 48000).

        Returns:
            1D float32 numpy array of shape (target_sample_rate * target_duration_seconds,).
        """
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            data, file_sr = sf.read(str(path), dtype="float32")
            waveform = data
            orig_sr = file_sr
        elif isinstance(audio, np.ndarray):
            waveform = audio.copy()
            orig_sr = sr
        else:
            raise TypeError(f"Expected path or numpy array, got {type(audio)}")

        # Ensure mono: downmix multi-channel audio
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)

        waveform = waveform.astype(np.float32)

        # Resample if sample rate does not match target
        if orig_sr != self.target_sample_rate:
            waveform = self.resample_waveform(waveform, orig_sr)

        # Standardize duration (symmetrical padding or center-cropping)
        target_len = int(round(self.target_sample_rate * self.target_duration_seconds))
        if len(waveform) < target_len:
            pad_total = target_len - len(waveform)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")
        elif len(waveform) > target_len:
            start = (len(waveform) - target_len) // 2
            waveform = waveform[start : start + target_len]

        return waveform.astype(np.float32)

    def embed(
        self, audio: Union[str, Path, np.ndarray], sr: int = 48000
    ) -> np.ndarray:
        """Public entrypoint to extract a strictly unit L2-normalized feature embedding vector.

        Args:
            audio: File path or raw audio waveform array.
            sr: Sampling rate of input if numpy array is passed (default: 48000).

        Returns:
            1D float32 numpy array of shape (embedding_dim,) strictly satisfying
            abs(np.linalg.norm(embedding) - 1.0) < 1e-5.
        """
        waveform = self.preprocess_audio(audio, sr=sr)
        raw_emb = self._extract_impl(waveform)
        v = raw_emb.flatten().astype(np.float32)

        if len(v) != self.embedding_dim:
            raise ValueError(
                f"Backbone '{self.name}' returned dimension {len(v)}, expected {self.embedding_dim}"
            )

        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            v_norm = v / norm
        else:
            # Handle near-zero / digital silence safely with uniform unit vector
            v_norm = np.ones_like(v, dtype=np.float32) / np.float32(np.sqrt(len(v)))

        # Enforce strict L2-normalization contract
        unit_norm = float(np.linalg.norm(v_norm))
        if abs(unit_norm - 1.0) >= 1e-5:
            raise ValueError(
                f"Strict L2 normalization failed: norm = {unit_norm:.8f}, expected 1.0 ± 1e-5"
            )

        return v_norm.astype(np.float32)
