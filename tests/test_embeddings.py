"""Unit and Contract Test Suite for AnyCall Embedding Backbones (Milestone 2).

Covers:
1. BaseAudioEmbeddingBackbone interface contract, duration padding/cropping,
   polyphase resampling, channel downmix, and strict unit L2-normalization.
2. MockBackbone determinism, orthogonality, dimension configurability,
   sub-millisecond latency budget, and synthetic audio fixture compatibility.
3. Real candidate backbones extraction on real Corvus splendens WAV recording.
4. Factory and registry functions (get_backbone, list_backbones).
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Type

import numpy as np
import pytest
import soundfile as sf

from anycall.embeddings import (
    BACKBONE_REGISTRY,
    BaseAudioEmbeddingBackbone,
    BirdNetBackbone,
    MockBackbone,
    PannsBackbone,
    PerchBackbone,
    get_backbone,
    list_backbones,
)
from tests.fixtures.synth_audio import (
    generate_animal_call,
    generate_broadband_noise,
    generate_harmonic_chirp,
    generate_pure_tone,
    generate_silence,
    generate_transient_pulse,
)


# ==============================================================================
# 1. BaseAudioEmbeddingBackbone Contract Verification
# ==============================================================================


class TestBaseAudioEmbeddingBackboneContract:
    """Verifies that BaseAudioEmbeddingBackbone enforces its contract."""

    def test_cannot_instantiate_abstract_base(self):
        """Verifies that BaseAudioEmbeddingBackbone cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseAudioEmbeddingBackbone()  # type: ignore

    def test_subclass_must_implement_properties(self):
        """Verifies that subclasses missing required abstract methods/properties fail instantiation."""

        class IncompleteBackbone(BaseAudioEmbeddingBackbone):
            pass

        with pytest.raises(TypeError):
            IncompleteBackbone()  # type: ignore

    def test_dummy_subclass_embed_contract(self, tmp_path):
        """Tests that a compliant subclass correctly executes the embed() template method."""

        class DummyBackbone(BaseAudioEmbeddingBackbone):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def embedding_dim(self) -> int:
                return 128

            @property
            def target_sample_rate(self) -> int:
                return 48000

            def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
                # Return arbitrary un-normalized vector based on waveform energy
                vec = np.ones(128, dtype=np.float32) * float(np.mean(waveform) + 1.0)
                return vec

        dummy = DummyBackbone()
        assert dummy.name == "dummy"
        assert dummy.embedding_dim == 128
        assert dummy.target_sample_rate == 48000
        assert dummy.target_duration_seconds == 3.0

        # 1. Test in-memory 1D float32 array
        audio = np.random.uniform(-0.5, 0.5, 144000).astype(np.float32)
        emb = dummy.embed(audio)
        assert emb.shape == (128,)
        assert emb.dtype == np.float32
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

        # 2. Test 2D stereo input (auto-downmix)
        stereo = np.random.uniform(-0.5, 0.5, (144000, 2)).astype(np.float32)
        emb_stereo = dummy.embed(stereo)
        assert emb_stereo.shape == (128,)
        assert abs(float(np.linalg.norm(emb_stereo)) - 1.0) < 1e-5

        # 3. Test short input zero-padding (< 3.0s)
        short_audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)  # 1.0s
        emb_short = dummy.embed(short_audio)
        assert emb_short.shape == (128,)
        assert abs(float(np.linalg.norm(emb_short)) - 1.0) < 1e-5

        # 4. Test long input cropping (> 3.0s)
        long_audio = np.random.uniform(-0.5, 0.5, 240000).astype(np.float32)  # 5.0s
        emb_long = dummy.embed(long_audio)
        assert emb_long.shape == (128,)
        assert abs(float(np.linalg.norm(emb_long)) - 1.0) < 1e-5

        # 5. Test audio file path input
        wav_file = tmp_path / "dummy_test.wav"
        sf.write(str(wav_file), audio, 48000)
        emb_file = dummy.embed(str(wav_file))
        assert emb_file.shape == (128,)
        assert abs(float(np.linalg.norm(emb_file)) - 1.0) < 1e-5

        # 6. Test digital silence zero division safety
        silence = np.zeros(144000, dtype=np.float32)
        emb_silence = dummy.embed(silence)
        assert emb_silence.shape == (128,)
        assert abs(float(np.linalg.norm(emb_silence)) - 1.0) < 1e-5
        assert not np.isnan(emb_silence).any()

        # 7. Test polyphase resampling when input rate differs
        audio_32k = np.random.uniform(-0.5, 0.5, 96000).astype(np.float32)  # 3.0s at 32kHz
        emb_32k = dummy.embed(audio_32k, sr=32000)
        assert emb_32k.shape == (128,)
        assert abs(float(np.linalg.norm(emb_32k)) - 1.0) < 1e-5

        # 8. Test non-existent file error
        with pytest.raises(FileNotFoundError):
            dummy.embed("non_existent_audio_file.wav")

        # 9. Test invalid audio input type error
        with pytest.raises(TypeError):
            dummy.embed(12345)  # type: ignore

    def test_dimension_mismatch_raises_error(self):
        """Verifies that an implementation returning wrong dimensions raises ValueError."""

        class BrokenDimBackbone(BaseAudioEmbeddingBackbone):
            @property
            def name(self) -> str:
                return "broken"

            @property
            def embedding_dim(self) -> int:
                return 256

            @property
            def target_sample_rate(self) -> int:
                return 48000

            def _extract_impl(self, waveform: np.ndarray) -> np.ndarray:
                return np.ones(128, dtype=np.float32)  # Returns 128 instead of 256

        b = BrokenDimBackbone()
        with pytest.raises(ValueError, match="returned dimension 128, expected 256"):
            b.embed(np.zeros(144000, dtype=np.float32))


# ==============================================================================
# 2. MockBackbone Unit & Boundary Tests
# ==============================================================================


class TestMockBackbone:
    """Verifies MockBackbone determinism, orthogonality, and performance."""

    def test_mock_default_properties(self):
        """Verifies default properties: name='mock', dim=256, target_sr=48000."""
        mock = MockBackbone()
        assert mock.name == "mock"
        assert mock.embedding_dim == 256
        assert mock.target_sample_rate == 48000
        assert mock.target_duration_seconds == 3.0

    def test_mock_custom_parameters(self):
        """Verifies parameterization: custom dimensions and names."""
        for dim in [320, 512, 1024, 1280, 2048]:
            mock = MockBackbone(embedding_dim=dim, name=f"mock_{dim}")
            assert mock.embedding_dim == dim
            assert mock.name == f"mock_{dim}"

            audio = np.random.randn(144000).astype(np.float32)
            emb = mock.embed(audio)
            assert emb.shape == (dim,)
            assert emb.dtype == np.float32
            assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_mock_invalid_dimension_raises_error(self):
        """Verifies that non-positive embedding dimensions raise ValueError."""
        with pytest.raises(ValueError):
            MockBackbone(embedding_dim=0)
        with pytest.raises(ValueError):
            MockBackbone(embedding_dim=-64)

    def test_mock_determinism_same_audio(self):
        """Verifies that identical audio inputs produce identical embeddings."""
        mock = MockBackbone(embedding_dim=256)
        audio = generate_harmonic_chirp(duration=3.0, sr=48000)

        emb1 = mock.embed(audio)
        emb2 = mock.embed(audio)

        assert np.array_equal(emb1, emb2), "Embeddings for identical audio must be identical"
        assert float(np.dot(emb1, emb2)) == pytest.approx(1.0, abs=1e-6)

    def test_mock_distinct_audio_orthogonality(self):
        """Verifies that distinct audio signals produce near-orthogonal vectors with high distance."""
        mock = MockBackbone(embedding_dim=256)

        audio1 = generate_pure_tone(freq=1000.0, duration=3.0, sr=48000)
        audio2 = generate_pure_tone(freq=3500.0, duration=3.0, sr=48000)
        audio3 = generate_broadband_noise(duration=3.0, sr=48000, noise_type="white")

        emb1 = mock.embed(audio1)
        emb2 = mock.embed(audio2)
        emb3 = mock.embed(audio3)

        # In D=256, dot product of random unit vectors has mean 0, std 1/sqrt(256)=0.0625.
        # Expect |dot(u, v)| < 0.25 (well within orthogonal regime)
        cos_12 = float(np.dot(emb1, emb2))
        cos_13 = float(np.dot(emb1, emb3))
        cos_23 = float(np.dot(emb2, emb3))

        assert abs(cos_12) < 0.25, f"Expected near-orthogonal embeddings, got cos={cos_12}"
        assert abs(cos_13) < 0.25, f"Expected near-orthogonal embeddings, got cos={cos_13}"
        assert abs(cos_23) < 0.25, f"Expected near-orthogonal embeddings, got cos={cos_23}"

        # Cosine distance = 1 - cos_sim > 0.75 (high distance)
        assert (1.0 - cos_12) > 0.75

    def test_mock_latency_budget_under_one_millisecond(self):
        """Strictly benchmarks execution time, asserting mean latency < 1.0 ms."""
        mock = MockBackbone(embedding_dim=256)
        audio = np.random.randn(144000).astype(np.float32)

        # Warmup
        mock.embed(audio)

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            mock.embed(audio)
        total_time = time.perf_counter() - start
        mean_ms = (total_time / iterations) * 1000.0

        assert mean_ms < 1.0, f"Execution time budget violated: {mean_ms:.3f} ms >= 1.0 ms"

    def test_mock_synthetic_fixtures_compatibility(self):
        """Verifies compatibility with all synthetic audio generator fixtures."""
        mock = MockBackbone(embedding_dim=256)
        generators = [
            generate_pure_tone(freq=1500.0, duration=3.0),
            generate_harmonic_chirp(start_freq=500.0, end_freq=4000.0, duration=3.0),
            generate_broadband_noise(noise_type="pink", duration=3.0),
            generate_transient_pulse(carrier_freq=6000.0, duration=3.0),
            generate_animal_call("aves", duration=3.0),
            generate_animal_call("insecta", duration=3.0),
            generate_animal_call("amphibia", duration=3.0),
            generate_animal_call("mammalia", duration=3.0),
            generate_silence(duration=3.0),
        ]

        for audio in generators:
            emb = mock.embed(audio)
            assert emb.shape == (256,)
            assert emb.dtype == np.float32
            assert np.all(np.isfinite(emb))
            assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_mock_real_corvus_splendens_file(self):
        """Verifies extraction on real Corvus splendens WAV recording."""
        wav_path = Path("data/processed/corvus_splendens/1009327_seg000.wav")
        if not wav_path.exists():
            pytest.skip(f"Sample WAV not found at {wav_path}")

        mock = MockBackbone(embedding_dim=256)
        emb_file = mock.embed(str(wav_path))

        assert emb_file.shape == (256,)
        assert emb_file.dtype == np.float32
        assert abs(float(np.linalg.norm(emb_file)) - 1.0) < 1e-5

        # Array extraction consistency
        audio, sr = sf.read(str(wav_path), dtype="float32")
        emb_arr = mock.embed(audio, sr=sr)
        assert np.allclose(emb_file, emb_arr, atol=1e-5)

    def test_mock_nan_inf_sanitization(self):
        """Verifies that waveforms containing NaN or Inf are sanitized safely."""
        mock = MockBackbone(embedding_dim=256)
        corrupted = np.ones(144000, dtype=np.float32)
        corrupted[10] = np.nan
        corrupted[20] = np.inf
        corrupted[30] = -np.inf

        emb = mock.embed(corrupted)
        assert emb.shape == (256,)
        assert np.all(np.isfinite(emb))
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


# ==============================================================================
# 3. Real Backbones Integration Tests
# ==============================================================================


class TestRealCandidateBackbones:
    """Verifies real deep learning backbones on real Corvus splendens WAV recording."""

    @pytest.fixture
    def real_wav_path(self) -> Path:
        p = Path("data/processed/corvus_splendens/1009327_seg000.wav")
        if not p.exists():
            pytest.skip(f"Test recording {p} not available")
        return p

    def test_birdnet_backbone_real_wav(self, real_wav_path):
        """Verifies BirdNetBackbone extraction, unit norm, shape, and logits."""
        backbone = BirdNetBackbone()
        assert backbone.name == "birdnet"
        assert backbone.embedding_dim == 1024
        assert backbone.target_sample_rate == 48000

        emb = backbone.embed(str(real_wav_path))
        assert emb.ndim == 1
        assert emb.shape == (1024,)
        assert emb.dtype == np.float32
        assert np.all(np.isfinite(emb))
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

        # Also test extract_with_logits
        emb_with_logits, logits = backbone.extract_with_logits(str(real_wav_path))
        assert emb_with_logits.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_with_logits)) - 1.0) < 1e-5
        assert logits is not None
        assert logits.shape == (6522,)

    def test_perch_backbone_real_wav(self, real_wav_path):
        """Verifies PerchBackbone extraction, unit norm, and shape (1280,)."""
        backbone = PerchBackbone()
        assert backbone.name == "perch"
        assert backbone.embedding_dim == 1280
        assert backbone.target_sample_rate == 32000

        emb = backbone.embed(str(real_wav_path))
        assert emb.ndim == 1
        assert emb.shape == (1280,)
        assert emb.dtype == np.float32
        assert np.all(np.isfinite(emb))
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_panns_backbone_real_wav(self, real_wav_path):
        """Verifies PannsBackbone extraction, unit norm, and shape (2048,)."""
        backbone = PannsBackbone()
        assert backbone.name == "panns"
        assert backbone.embedding_dim == 2048
        assert backbone.target_sample_rate == 32000

        emb = backbone.embed(str(real_wav_path))
        assert emb.ndim == 1
        assert emb.shape == (2048,)
        assert emb.dtype == np.float32
        assert np.all(np.isfinite(emb))
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_backbone_array_file_parity_real_audio(self, real_wav_path):
        """Verifies parity between file path and in-memory numpy array extraction."""
        audio, sr = sf.read(str(real_wav_path), dtype="float32")

        # Test BirdNET
        b_birdnet = BirdNetBackbone()
        emb_file = b_birdnet.embed(str(real_wav_path))
        emb_arr = b_birdnet.embed(audio, sr=sr)
        cos_sim = float(np.dot(emb_file, emb_arr))
        assert cos_sim >= 0.999, f"BirdNET file/array parity failed: cos_sim={cos_sim}"

        # Test Perch
        b_perch = PerchBackbone()
        emb_file_p = b_perch.embed(str(real_wav_path))
        emb_arr_p = b_perch.embed(audio, sr=sr)
        cos_sim_p = float(np.dot(emb_file_p, emb_arr_p))
        assert cos_sim_p >= 0.999, f"Perch file/array parity failed: cos_sim={cos_sim_p}"

        # Test PANNs
        b_panns = PannsBackbone()
        emb_file_pa = b_panns.embed(str(real_wav_path))
        emb_arr_pa = b_panns.embed(audio, sr=sr)
        cos_sim_pa = float(np.dot(emb_file_pa, emb_arr_pa))
        assert cos_sim_pa >= 0.999, f"PANNs file/array parity failed: cos_sim={cos_sim_pa}"


# ==============================================================================
# 4. Factory & Registry Tests
# ==============================================================================


class TestFactoryRegistry:
    """Verifies get_backbone and list_backbones registry behavior."""

    def test_registry_contains_all_backbones(self):
        """Verifies that BACKBONE_REGISTRY includes all four candidate backbones."""
        expected = {"mock", "birdnet", "perch", "panns"}
        assert expected.issubset(set(BACKBONE_REGISTRY.keys()))
        assert expected.issubset(set(list_backbones()))

    def test_get_backbone_mock(self):
        """Verifies get_backbone('mock') instantiation with kwargs."""
        b = get_backbone("mock", embedding_dim=512)
        assert isinstance(b, MockBackbone)
        assert b.embedding_dim == 512

    def test_get_backbone_case_insensitive(self):
        """Verifies case-insensitive backbone retrieval."""
        b1 = get_backbone("BirdNET")
        assert isinstance(b1, BirdNetBackbone)

        b2 = get_backbone("PERCH")
        assert isinstance(b2, PerchBackbone)

        b3 = get_backbone("pAnNs")
        assert isinstance(b3, PannsBackbone)

    def test_get_backbone_unknown_raises_value_error(self):
        """Verifies that unknown backbone name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone 'nonexistent'"):
            get_backbone("nonexistent")
