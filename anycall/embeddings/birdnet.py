"""AnyCall BirdNET Audio Embedding Backbone.

Location: anycall/embeddings/birdnet.py
Implements BaseAudioEmbeddingBackbone using BirdNET's EfficientNet-B0
TFLite architecture (1024-dimensional embeddings, 48kHz, 3.0s input).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

from anycall.embeddings.base import BaseAudioEmbeddingBackbone


def _resolve_default_birdnet_checkpoint() -> Optional[Path]:
    """Locates the bundled BirdNET V2.4 TFLite model checkpoint."""
    candidates = [
        Path("models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"),
        Path(__file__).resolve().parent.parent.parent
        / "models"
        / "birdnet"
        / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
    ]

    # Check birdnetlib package installation
    try:
        import birdnetlib

        candidates.append(
            Path(birdnetlib.__file__).parent
            / "models"
            / "analyzer"
            / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
        )
    except ImportError:
        pass

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    return None


class BirdNetBackbone(BaseAudioEmbeddingBackbone):
    """BirdNET feature extraction wrapper using EfficientNet-B0 backbone.

    Operates natively at 48,000 Hz on 3.0-second clips (144,000 samples).
    Extracts 1024-dimensional penultimate embeddings from the GlobalAvgPool layer.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_threads: int = 1,
        offline_fallback: bool = False,
    ) -> None:
        """Initialize BirdNetBackbone.

        Args:
            model_path: Optional path to BirdNET .tflite model file.
            num_threads: Number of CPU threads for inference.
            offline_fallback: If True, uses deterministic fallback when weights unavailable.
        """
        self._name = "birdnet"
        self._dim = 1024
        self._target_sr = 48000
        self._target_duration = 3.0
        self._num_threads = num_threads
        self._offline_fallback = offline_fallback

        resolved = Path(model_path) if model_path else _resolve_default_birdnet_checkpoint()
        if resolved is None or not resolved.is_file():
            if self._offline_fallback:
                self._interpreter = None
                self._input_index = None
                self._emb_output_index = None
                self._logits_output_index = None
                return
            raise FileNotFoundError(
                f"BirdNET model checkpoint not found at: {resolved or 'default search paths'}"
            )

        self._model_path = resolved
        self._interpreter, self._emb_output_index, self._logits_output_index = self._init_interpreter()
        input_details = self._interpreter.get_input_details()
        self._input_index = input_details[0]["index"]

    def _init_interpreter(self) -> Tuple[Interpreter, int, Optional[int]]:
        """Initializes the TFLite Interpreter.

        First attempts dynamic in-memory FlatBuffer patching to register tensor 545 (embedding)
        and tensor 546 (logits) as formal graph outputs.
        Falls back to experimental_preserve_all_tensors=True if FlatBuffer editing fails.
        """
        # Strategy 1: Dynamic FlatBuffer in-memory reconfiguration
        try:
            import flatbuffers
            from tensorflow.lite.python import schema_py_generated as schema_fb

            with open(self._model_path, "rb") as f:
                buf = bytearray(f.read())

            model = schema_fb.Model.GetRootAsModel(buf, 0)
            model_t = schema_fb.ModelT.InitFromObj(model)

            # Subgraph 0: designate tensor 545 (embedding) and 546 (logits) as outputs
            model_t.subgraphs[0].outputs = [545, 546]

            builder = flatbuffers.Builder(1024 * 1024 * 60)
            builder.Finish(model_t.Pack(builder), file_identifier=b"TFL3")
            patched_bytes = bytes(builder.Output())

            interp = Interpreter(
                model_content=patched_bytes,
                num_threads=self._num_threads,
            )
            interp.allocate_tensors()

            emb_idx = None
            logits_idx = None
            for out in interp.get_output_details():
                shape_tail = tuple(out["shape"][-1:])
                if shape_tail == (1024,) or out["index"] == 545:
                    emb_idx = out["index"]
                elif shape_tail == (6522,) or out["index"] == 546:
                    logits_idx = out["index"]

            if emb_idx is not None:
                return interp, emb_idx, logits_idx
        except Exception:
            pass

        # Strategy 2: Fallback with experimental_preserve_all_tensors=True
        interp = Interpreter(
            model_path=str(self._model_path),
            num_threads=self._num_threads,
            experimental_preserve_all_tensors=True,
        )
        interp.allocate_tensors()
        emb_idx = 545
        logits_idx = 546
        return interp, emb_idx, logits_idx

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

    def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
        """Feeds 144,000 float32 audio samples into BirdNET and returns the 1024-d embedding."""
        if self._interpreter is None:
            # Deterministic SHA-256 fallback for offline tests
            import hashlib

            digest = hashlib.sha256(waveform.tobytes()).digest()
            seed = int.from_bytes(digest[:8], byteorder="little")
            rng = np.random.default_rng(seed)
            return rng.standard_normal(self._dim).astype(np.float32)

        input_data = np.expand_dims(waveform, axis=0).astype(np.float32)
        self._interpreter.set_tensor(self._input_index, input_data)
        self._interpreter.invoke()
        raw_emb = self._interpreter.get_tensor(self._emb_output_index)
        return raw_emb.flatten().astype(np.float32)

    def extract_with_logits(
        self, audio: Union[str, Path, np.ndarray], sr: int = 48000
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Auxiliary method for Experiment 1 (Stock BirdNET failure evaluation).

        Returns both the L2-normalized embedding and raw stock BirdNET logits.
        """
        waveform = self.preprocess_audio(audio, sr=sr)
        if self._interpreter is None:
            import hashlib

            digest = hashlib.sha256(waveform.tobytes()).digest()
            seed = int.from_bytes(digest[:8], byteorder="little")
            rng = np.random.default_rng(seed)
            raw_emb = rng.standard_normal(self._dim).astype(np.float32)
            logits = rng.standard_normal(6522).astype(np.float32)
        else:
            input_data = np.expand_dims(waveform, axis=0).astype(np.float32)
            self._interpreter.set_tensor(self._input_index, input_data)
            self._interpreter.invoke()
            raw_emb = self._interpreter.get_tensor(self._emb_output_index).flatten().astype(np.float32)
            logits = None
            if self._logits_output_index is not None:
                logits = self._interpreter.get_tensor(self._logits_output_index).flatten().astype(np.float32)

        norm = float(np.linalg.norm(raw_emb))
        v_norm = raw_emb / norm if norm > 1e-12 else np.ones_like(raw_emb) / np.sqrt(len(raw_emb))
        return v_norm.astype(np.float32), logits


# Alias for compatibility with blueprints and prior naming
BirdNETEmbeddingBackbone = BirdNetBackbone
