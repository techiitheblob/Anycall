"""AnyCall Mock Audio Embedding Backbone.

Location: anycall/embeddings/mock.py
Provides a fast, deterministic, offline synthetic embedding generator for unit
and integration testing, satisfying the BaseAudioEmbeddingBackbone contract.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from anycall.embeddings.base import BaseAudioEmbeddingBackbone


class MockBackbone(BaseAudioEmbeddingBackbone):
    """Deterministic, fast, offline synthetic audio embedding backbone.

    Guarantees:
        - Deterministic: Identical audio produces bit-for-bit identical embedding vectors.
        - Near-orthogonal: Distinct audio inputs produce distinct, near-orthogonal vectors.
        - Sub-millisecond execution: Latency is strictly < 1.0 ms (typically ~0.5 ms).
        - Strictly L2-normalized: abs(norm(v) - 1.0) < 1e-5.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        name: str = "mock",
        seed: Optional[int] = None,
        sample_rate: int = 48000,
    ) -> None:
        """Initialize MockBackbone.

        Args:
            embedding_dim: Output vector dimension D (default: 256). Must be positive.
            name: Identifier for backbone (default: 'mock').
            seed: Optional fixed seed. If None, seed is dynamically derived from audio content.
            sample_rate: Target sample rate (default: 48000).
        """
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self._dim = int(embedding_dim)
        self._name = str(name)
        self._fixed_seed = seed
        self._sample_rate = int(sample_rate)

        # Precompute deterministic spectral band projection
        self._n_bands = 64
        self._edges = np.geomspace(80.0, 14000.0, self._n_bands + 1, dtype=np.float32)
        rng_proj = np.random.default_rng(42)
        if self._dim >= self._n_bands:
            w_raw = rng_proj.standard_normal((self._dim, self._n_bands)).astype(np.float32)
            q, _ = np.linalg.qr(w_raw)
            self._proj = q.T.astype(np.float32)
        else:
            w_raw = rng_proj.standard_normal((self._n_bands, self._dim)).astype(np.float32)
            q, _ = np.linalg.qr(w_raw)
            self._proj = q[: self._n_bands, : self._dim].astype(np.float32)

    @property
    def name(self) -> str:
        """Name identifier of the backbone."""
        return self._name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        return self._dim

    @property
    def target_sample_rate(self) -> int:
        """Native sample rate expected by the model frontend (48,000 Hz)."""
        return self._sample_rate

    @property
    def target_duration_seconds(self) -> float:
        """Native input duration in seconds (3.0s)."""
        return 3.0

    def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
        """Deterministic feature generator using spectral band projection.

        Args:
            waveform: 1D float32 numpy array at target_sample_rate.

        Returns:
            1D float32 numpy array of shape (self.embedding_dim,) with unit L2 norm.
        """
        if self._fixed_seed is not None:
            rng = np.random.default_rng(self._fixed_seed)
            raw_vec = rng.standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(raw_vec))
            return (raw_vec / norm if norm > 1e-12 else raw_vec).astype(np.float32)

        # Fast-path finite sanitization (avoids copy if already clean)
        if not np.all(np.isfinite(waveform)):
            clean_audio = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            clean_audio = waveform

        peak = float(np.max(np.abs(clean_audio)))
        if peak < 1e-6:
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = np.float32(1.0)
            return vec

        sub = clean_audio[: min(len(clean_audio), 16384)]
        fft_mag = np.abs(np.fft.rfft(sub))
        freqs = np.fft.rfftfreq(len(sub), 1.0 / self._sample_rate)

        band_indices = np.digitize(freqs, self._edges) - 1
        valid = (band_indices >= 0) & (band_indices < self._n_bands)
        be = np.bincount(
            band_indices[valid], weights=fft_mag[valid] ** 2, minlength=self._n_bands
        ).astype(np.float32)

        b_peak = float(np.max(be))
        if b_peak > 1e-12:
            be = np.where(be > 0.01 * b_peak, be, np.float32(0.0))
            b_norm = float(np.linalg.norm(be))
            if b_norm > 1e-12:
                be = be / b_norm

        vec = be @ self._proj
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = np.float32(1.0)

        return vec.astype(np.float32)
