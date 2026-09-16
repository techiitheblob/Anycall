"""AnyCall Google Perch Audio Embedding Backbone.

Location: anycall/embeddings/perch.py
Implements BaseAudioEmbeddingBackbone using Google's bioacoustic EfficientNet
SavedModel (1280-dimensional embeddings, 32kHz, 5.0s input).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from anycall.embeddings.base import BaseAudioEmbeddingBackbone

DEFAULT_MODEL_PATHS = [
    Path("models/perch"),
    Path(__file__).resolve().parent.parent.parent / "models" / "perch",
    Path(os.environ.get("TEMP", "/tmp"))
    / "tfhub_modules"
    / "5dcbb82658655292c50ca88ce1e6f1073b17d0d9",
]

DEFAULT_HUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"


class PerchBackbone(BaseAudioEmbeddingBackbone):
    """Google Perch bioacoustic embedding extractor.

    Input audio requirements:
        - Native sample rate: 32,000 Hz.
        - Native duration: 5.0 seconds (160,000 samples).
        - 48 kHz 3.0s inputs are resampled to 32 kHz (96k) and center-padded to 5.0s (160k).
        - Output embedding: 1280-dimensional float32 vector, strictly unit L2-normalized.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        hub_url: str = DEFAULT_HUB_URL,
        offline_fallback: bool = False,
    ) -> None:
        """Initialize PerchBackbone.

        Args:
            model_path: Optional path to local SavedModel directory.
            hub_url: Fallback TensorFlow Hub model URL if local model not found.
            offline_fallback: If True, uses deterministic fallback when TF/weights unavailable.
        """
        self._name = "perch"
        self._dim = 1280
        self._target_sr = 32000
        self._target_duration = 5.0
        self._offline_fallback = offline_fallback
        self._infer_fn = None

        self._load_model(model_path=model_path, hub_url=hub_url)

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

    def _resolve_model_path(self, user_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolves existing SavedModel directory from user argument or default paths."""
        if user_path is not None:
            p = Path(user_path)
            if p.exists() and (p / "saved_model.pb").exists():
                return p.resolve()

        for candidate in DEFAULT_MODEL_PATHS:
            if candidate.exists() and (candidate / "saved_model.pb").exists():
                return candidate.resolve()
        return None

    def _load_model(self, model_path: Optional[Union[str, Path]], hub_url: str) -> None:
        """Loads SavedModel from local path or TF Hub."""
        resolved = self._resolve_model_path(model_path)
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as err:
            if self._offline_fallback:
                return
            raise ImportError(
                "TensorFlow is required for PerchBackbone. Install with `uv pip install tensorflow`."
            ) from err

        if resolved is not None:
            model = tf.saved_model.load(str(resolved))
            self._infer_fn = model.signatures["serving_default"]
        else:
            try:
                import tensorflow_hub as hub  # type: ignore

                model = hub.load(hub_url)
                self._infer_fn = model.signatures["serving_default"]
            except Exception as err:
                if self._offline_fallback:
                    return
                raise RuntimeError(
                    f"Failed to load Perch model from local paths or Hub ({hub_url}): {err}"
                ) from err

    def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
        """Extracts 1280-d embedding from 32kHz waveform padded to 160,000 samples.

        Args:
            waveform: 1D float32 numpy array.

        Returns:
            1D float32 numpy array of shape (1280,).
        """
        # Ensure exact 160,000 sample length via symmetrical center padding
        target_len = 160000
        if len(waveform) < target_len:
            pad_total = target_len - len(waveform)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")
        elif len(waveform) > target_len:
            start = (len(waveform) - target_len) // 2
            waveform = waveform[start : start + target_len]

        if self._infer_fn is None:
            # Offline deterministic fallback (seeded by audio contents)
            import hashlib

            digest = hashlib.sha256(waveform.tobytes()).digest()
            seed = int.from_bytes(digest[:8], byteorder="little")
            rng = np.random.default_rng(seed)
            return rng.standard_normal(self._dim).astype(np.float32)

        import tensorflow as tf  # type: ignore

        input_tensor = tf.constant(waveform[np.newaxis, :], dtype=tf.float32)
        outputs = self._infer_fn(inputs=input_tensor)

        # Structured output 'output_1' contains the 1280-d embedding
        if "output_1" in outputs:
            raw_emb = outputs["output_1"].numpy()
        elif "embedding" in outputs:
            raw_emb = outputs["embedding"].numpy()
        else:
            keys = list(outputs.keys())
            raw_emb = outputs[keys[1]].numpy() if len(keys) > 1 else outputs[keys[0]].numpy()

        return raw_emb.flatten().astype(np.float32)
