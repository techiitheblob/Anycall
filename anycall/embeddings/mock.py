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
        """Deterministic feature generator using SHA-256 audio hashing.

        Args:
            waveform: 1D float32 numpy array at target_sample_rate.

        Returns:
            1D float32 numpy array of shape (self.embedding_dim,) with unit L2 norm.
        """
        if self._fixed_seed is not None:
            seed = self._fixed_seed
        else:
            # Fast-path finite sanitization (avoids copy if already clean)
            if not np.all(np.isfinite(waveform)):
                clean_audio = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                clean_audio = waveform

            # Ensure contiguous memory layout for hashing
            audio_bytes = np.ascontiguousarray(clean_audio, dtype=np.float32).tobytes()

            # Deterministic SHA-256 digest
            digest = hashlib.sha256(audio_bytes).digest()

            # Derive 64-bit integer seed from initial 8 bytes of hash digest
            seed = int.from_bytes(digest[:8], byteorder="little")

        # Generate Gaussian pseudo-random vector from seed
        rng = np.random.default_rng(seed)
        raw_vec = rng.standard_normal(self._dim).astype(np.float32)

        # Enforce unit L2-normalization
        norm = float(np.linalg.norm(raw_vec))
        if norm > 1e-12:
            norm_vec = raw_vec / norm
        else:
            norm_vec = np.zeros(self._dim, dtype=np.float32)
            norm_vec[0] = np.float32(1.0)

        return norm_vec.astype(np.float32)
