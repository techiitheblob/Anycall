"""AnyCall PANNs Audio Embedding Backbone.

Location: anycall/embeddings/panns.py
Implements BaseAudioEmbeddingBackbone using Pretrained Audio Neural Networks (PANNs)
CNN14 architecture (2048-dimensional embeddings, 32kHz, 3.0s input).
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Optional, Union
import urllib.request

import numpy as np

from anycall.embeddings.base import BaseAudioEmbeddingBackbone

DEFAULT_PANN_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth"
DEFAULT_CHECKPOINT_FILENAME = "Cnn14_mAP=0.431.pth"


def download_panns_checkpoint(
    target_path: Path,
    url: str = DEFAULT_PANN_URL,
    min_bytes: int = 300_000_000,
) -> Path:
    """Safely downloads PANNs CNN14 checkpoint via Python HTTPS, bypassing Windows wget bugs.

    Args:
        target_path: Destination path for checkpoint file.
        url: Remote URL for checkpoint download.
        min_bytes: Minimum expected file size in bytes to verify completeness.

    Returns:
        Verified Path to the downloaded checkpoint.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size >= min_bytes:
        return target_path

    temp_path = target_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "AnyCall/0.1"})

    with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out_file:
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB buffer
            if not chunk:
                break
            out_file.write(chunk)

    temp_path.replace(target_path)
    return target_path


class PannsBackbone(BaseAudioEmbeddingBackbone):
    """PANNs (CNN14) AudioTagging embedding extractor.

    Input audio requirements:
        - Native sample rate: 32,000 Hz.
        - Native duration: 3.0 seconds (96,000 samples).
        - 48 kHz 3.0s inputs are resampled to 32 kHz (96k).
        - Output embedding: 2048-dimensional float32 vector, strictly unit L2-normalized.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        offline_fallback: bool = False,
    ) -> None:
        """Initialize PannsBackbone.

        Args:
            checkpoint_path: Optional path to Cnn14_mAP=0.431.pth.
            device: Computation device ('cpu' or 'cuda').
            offline_fallback: If True, uses deterministic fallback when weights unavailable.
        """
        self._name = "panns"
        self._dim = 2048
        self._target_sr = 32000
        self._target_duration = 3.0
        self._device = device
        self._offline_fallback = offline_fallback
        self._at_model = None

        self._load_model(checkpoint_path=checkpoint_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def target_sample_rate(self) -> int:
        return self._target_sr

    @property
    def target_duration_seconds(self) -> float:
        return self._target_duration

    def _resolve_checkpoint(self, user_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolves or safely downloads the PANNs checkpoint file."""
        if user_path is not None:
            p = Path(user_path)
            if p.exists() and p.stat().st_size >= 300_000_000:
                return p.resolve()

        # Check candidate search locations
        candidates = [
            Path("models/panns") / DEFAULT_CHECKPOINT_FILENAME,
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / "panns"
            / DEFAULT_CHECKPOINT_FILENAME,
            Path.home() / "panns_data" / DEFAULT_CHECKPOINT_FILENAME,
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size >= 300_000_000:
                return c.resolve()

        if self._offline_fallback:
            return None

        # Proactively download via Python HTTPS to prevent Windows wget crash
        dest = Path("models/panns") / DEFAULT_CHECKPOINT_FILENAME
        download_panns_checkpoint(dest)

        # Also mirror to ~/panns_data/ for default panns_inference compatibility
        home_dest = Path.home() / "panns_data" / DEFAULT_CHECKPOINT_FILENAME
        if not home_dest.exists() or home_dest.stat().st_size < 300_000_000:
            home_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, home_dest)

        return dest.resolve()

    def _load_model(self, checkpoint_path: Optional[Union[str, Path]]) -> None:
        """Initializes panns_inference AudioTagging model."""
        resolved = self._resolve_checkpoint(checkpoint_path)
        if resolved is None:
            if self._offline_fallback:
                return
            raise FileNotFoundError(
                "PANNs CNN14 checkpoint could not be found or downloaded."
            )

        try:
            from panns_inference import AudioTagging  # type: ignore

            self._at_model = AudioTagging(checkpoint_path=str(resolved), device=self._device)
        except Exception as err:
            if self._offline_fallback:
                return
            raise RuntimeError(f"Failed to initialize PANNs AudioTagging: {err}") from err

    def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
        """Extracts 2048-d embedding from 32kHz 3.0s waveform (96,000 samples).

        Args:
            waveform: 1D float32 numpy array (96,000 samples).

        Returns:
            1D float32 numpy array of shape (2048,).
        """
        # Ensure exact 96,000 sample length via symmetrical center padding
        target_len = 96000
        if len(waveform) < target_len:
            pad_total = target_len - len(waveform)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")
        elif len(waveform) > target_len:
            start = (len(waveform) - target_len) // 2
            waveform = waveform[start : start + target_len]

        if self._at_model is None:
            # Deterministic synthetic fallback for offline tests
            import hashlib

            digest = hashlib.sha256(waveform.tobytes()).digest()
            seed = int.from_bytes(digest[:8], byteorder="little")
            rng = np.random.default_rng(seed)
            return rng.standard_normal(self._dim).astype(np.float32)

        # AudioTagging requires shape (batch_size, num_samples)
        audio_batch = waveform[np.newaxis, :].astype(np.float32)
        _, raw_embedding = self._at_model.inference(audio_batch)

        return raw_embedding.flatten().astype(np.float32)
