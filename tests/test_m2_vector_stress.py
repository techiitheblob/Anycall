"""Adversarial Vector Stress and Edge-Case Test Suite for AnyCall Milestone 2.

Location: tests/test_m2_vector_stress.py
Author: challenger_m2_1

Empirically challenges BaseAudioEmbeddingBackbone, MockBackbone, BirdNetBackbone,
PerchBackbone, and PannsBackbone across four adversarial test dimensions:
1. Extreme input vectors: digital silence, DC bias, near-zero, saturation, NaN/Inf.
2. MockBackbone determinism: 1,000 iterations on random waveforms with 0 variance.
3. Collision / orthogonality check: 500 distinct synthetic calls across taxa with max cosine < 0.35.
4. Universal strict L2 norm check: abs(norm - 1.0) < 1e-5 across all stress tests.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from anycall.embeddings import (
    BaseAudioEmbeddingBackbone,
    BirdNetBackbone,
    MockBackbone,
    PannsBackbone,
    PerchBackbone,
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
# Shared Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def mock_backbone() -> MockBackbone:
    return MockBackbone(embedding_dim=256)


@pytest.fixture(scope="module")
def birdnet_backbone() -> BirdNetBackbone:
    return BirdNetBackbone()


@pytest.fixture(scope="module")
def perch_backbone() -> PerchBackbone:
    return PerchBackbone()


@pytest.fixture(scope="module")
def panns_backbone() -> PannsBackbone:
    return PannsBackbone()


# ==============================================================================
# 1. Extreme Input Vectors Test Suite
# ==============================================================================

class TestExtremeInputVectors:
    """Adversarial challenge for extreme numerical vectors and boundary inputs."""

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_digital_silence_all_zeros(self, backbone_fixture, request):
        """Tests digital silence (all zeros) of various lengths."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        lengths = [
            100,      # Extremely short (padded)
            144000,   # Standard 3.0s at 48kHz
            288000,   # Long (cropped)
        ]
        for length in lengths:
            silence = np.zeros(length, dtype=np.float32)
            emb = backbone.embed(silence)

            assert emb.shape == (backbone.embedding_dim,)
            assert emb.dtype == np.float32
            assert np.all(np.isfinite(emb)), f"NaN/Inf detected in {backbone.name} silence embedding"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"{backbone.name} norm={norm:.8f} violates L2 contract"

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_constant_dc_bias(self, backbone_fixture, request):
        """Tests non-zero constant DC offset audio vectors."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        biases = [0.1, 0.5, 1.0, 10.0, -0.5, -5.0]
        for bias in biases:
            dc_audio = np.full(144000, bias, dtype=np.float32)
            emb = backbone.embed(dc_audio)

            assert emb.shape == (backbone.embedding_dim,)
            assert np.all(np.isfinite(emb)), f"Non-finite output for DC bias {bias} in {backbone.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"DC bias {bias} norm error: {abs(norm - 1.0):.2e}"

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_near_zero_amplitude(self, backbone_fixture, request):
        """Tests near-zero amplitude and subnormal floating-point audio vectors."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        scales = [1e-7, 1e-15, 1e-25, 1e-38]
        for scale in scales:
            audio = (np.sin(np.linspace(0, 100, 144000)) * scale).astype(np.float32)
            emb = backbone.embed(audio)

            assert emb.shape == (backbone.embedding_dim,)
            assert np.all(np.isfinite(emb)), f"Near-zero scale {scale} produced non-finite in {backbone.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Near-zero scale {scale} norm error: {abs(norm - 1.0):.2e}"

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_massive_saturation_and_clipping(self, backbone_fixture, request):
        """Tests heavily saturated and clipped waveforms far exceeding [-1.0, 1.0]."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        saturations = [
            np.full(144000, 100.0, dtype=np.float32),                                 # Constant saturation
            np.tile(np.array([100.0, -100.0], dtype=np.float32), 72000),             # Full-scale square wave
            (np.sin(np.linspace(0, 200, 144000)) * 1000.0).astype(np.float32),       # Megavolt sine
            np.full(144000, 1e5, dtype=np.float32),                                   # Extreme positive
        ]
        for sat in saturations:
            emb = backbone.embed(sat)

            assert emb.shape == (backbone.embedding_dim,)
            assert np.all(np.isfinite(emb)), f"Saturation produced non-finite in {backbone.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Saturation norm error: {abs(norm - 1.0):.2e}"

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_nan_and_inf_resilience(self, backbone_fixture, request):
        """Tests that corrupted arrays containing NaN and Inf are handled gracefully."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)

        # 1. Scattered NaNs and Infs
        scattered = np.sin(np.linspace(0, 100, 144000)).astype(np.float32)
        scattered[0] = np.nan
        scattered[500] = np.inf
        scattered[1000] = -np.inf
        scattered[-1] = np.nan

        # 2. All NaNs
        all_nan = np.full(144000, np.nan, dtype=np.float32)

        # 3. All +Inf
        all_posinf = np.full(144000, np.inf, dtype=np.float32)

        # 4. All -Inf
        all_neginf = np.full(144000, -np.inf, dtype=np.float32)

        # 5. Mixed NaNs and Infs
        mixed = np.empty(144000, dtype=np.float32)
        mixed[0::3] = np.nan
        mixed[1::3] = np.inf
        mixed[2::3] = -np.inf

        for corrupted in [scattered, all_nan, all_posinf, all_neginf, mixed]:
            emb = backbone.embed(corrupted)

            assert emb.shape == (backbone.embedding_dim,)
            assert np.all(np.isfinite(emb)), f"Corrupted input produced non-finite in {backbone.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Corrupted input norm error: {abs(norm - 1.0):.2e}"

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_multichannel_extreme_imbalance(self, backbone_fixture, request):
        """Tests multi-channel audio with extreme channel-to-channel discrepancies."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        # Channel 0: silence, Channel 1: massive saturation
        stereo = np.zeros((144000, 2), dtype=np.float32)
        stereo[:, 1] = 100.0

        emb = backbone.embed(stereo)
        assert emb.shape == (backbone.embedding_dim,)
        assert np.all(np.isfinite(emb))
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.parametrize(
        "backbone_fixture",
        ["mock_backbone", "birdnet_backbone", "perch_backbone", "panns_backbone"],
    )
    def test_sample_rate_extremes(self, backbone_fixture, request):
        """Tests polyphase resampling under non-standard and extreme sample rates."""
        backbone: BaseAudioEmbeddingBackbone = request.getfixturevalue(backbone_fixture)
        test_rates = [8000, 16000, 22050, 44100, 48000, 96000]

        for sr in test_rates:
            n_samples = int(sr * 3.0)
            t = np.linspace(0, 3.0, n_samples, endpoint=False)
            audio = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
            emb = backbone.embed(audio, sr=sr)

            assert emb.shape == (backbone.embedding_dim,)
            assert np.all(np.isfinite(emb))
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Sample rate {sr} norm error: {abs(norm - 1.0):.2e}"


# ==============================================================================
# 2. MockBackbone Determinism Test Suite (1,000 Iterations)
# ==============================================================================

class TestMockBackboneDeterminism1000Iterations:
    """Rigorous empirical verification of MockBackbone zero-variance determinism."""

    def test_determinism_across_1000_distinct_waveforms(self):
        """Tests 1,000 distinct random waveforms evaluated twice; asserts 0 variance."""
        mock = MockBackbone(embedding_dim=256)
        rng = np.random.default_rng(seed=1337)

        n_iterations = 1000
        waveforms = [
            rng.standard_normal(144000).astype(np.float32) for _ in range(n_iterations)
        ]

        # Pass 1
        embs_pass1 = np.array([mock.embed(w) for w in waveforms], dtype=np.float32)
        # Pass 2
        embs_pass2 = np.array([mock.embed(w) for w in waveforms], dtype=np.float32)

        # 1. Absolute difference must be identically 0.0 everywhere
        abs_diff = np.abs(embs_pass1 - embs_pass2)
        max_diff = float(np.max(abs_diff))
        assert max_diff == 0.0, f"Determinism violated: max absolute difference = {max_diff}"

        # 2. Component-wise variance across the two runs must be identically 0.0
        stacked = np.stack([embs_pass1, embs_pass2], axis=1)  # (1000, 2, 256)
        variances = np.var(stacked, axis=1)                  # (1000, 256)
        max_var = float(np.max(variances))
        assert max_var == 0.0, f"Determinism violated: max variance across repetitions = {max_var}"

        # 3. Strict L2 norm check across all 1,000 embeddings
        norms = np.linalg.norm(embs_pass1, axis=1)
        norm_errors = np.abs(norms - 1.0)
        max_norm_err = float(np.max(norm_errors))
        assert max_norm_err < 1e-5, f"L2 norm contract violated in 1000 waveforms: max err = {max_norm_err:.2e}"

    def test_determinism_1000_sequential_evaluations_same_waveform(self):
        """Evaluates a single fixed random waveform 1,000 consecutive times; asserts 0 variance."""
        mock = MockBackbone(embedding_dim=256)
        rng = np.random.default_rng(seed=9999)
        waveform = rng.standard_normal(144000).astype(np.float32)

        collected_embs = np.zeros((1000, 256), dtype=np.float32)
        for i in range(1000):
            collected_embs[i] = mock.embed(waveform)

        # Variance across all 1,000 evaluations
        col_var = np.var(collected_embs, axis=0)  # Shape (256,)
        max_col_var = float(np.max(col_var))
        assert max_col_var == 0.0, f"Sequential determinism violated: max variance = {max_col_var}"

        # All 1000 rows must be bit-for-bit identical to the first row
        first_row = collected_embs[0]
        assert np.array_equal(collected_embs, np.tile(first_row, (1000, 1)))

        # Unit norm check
        norm = float(np.linalg.norm(first_row))
        assert abs(norm - 1.0) < 1e-5


# ==============================================================================
# 3. Collision / Orthogonality Test Suite (500 Distinct Synthetic Calls)
# ==============================================================================

class TestCollisionAndOrthogonality500Calls:
    """Rigorous empirical verification of vector orthogonality and collision resistance."""

    def test_orthogonality_500_distinct_calls_across_taxa(self):
        """Generates 500 distinct synthetic calls across 4 taxa; asserts max cosine < 0.35."""
        mock = MockBackbone(embedding_dim=256)
        taxa = ["aves", "insecta", "amphibia", "mammalia"]

        n_calls = 500
        embeddings: List[np.ndarray] = []

        for i in range(n_calls):
            taxon = taxa[i % len(taxa)]
            call_audio = generate_animal_call(taxon=taxon, duration=3.0, sr=48000, seed=10000 + i)
            emb = mock.embed(call_audio)

            # Strict L2 norm check on each call
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Call {i} norm error: {abs(norm - 1.0):.2e}"

            embeddings.append(emb)

        emb_matrix = np.array(embeddings, dtype=np.float32)  # Shape: (500, 256)
        assert emb_matrix.shape == (500, 256)

        # Compute full 500x500 pairwise cosine similarity matrix via inner products
        # Note: Since vectors are strictly unit-normalized, cos(u, v) = u . v
        cos_matrix = np.matmul(emb_matrix, emb_matrix.T)  # Shape: (500, 500)

        # Self-similarity on diagonal must be 1.0
        diag_sims = np.diag(cos_matrix)
        assert np.allclose(diag_sims, 1.0, atol=1e-5), "Diagonal self-similarity must be 1.0"

        # Extract strictly off-diagonal elements (124,750 unique symmetric pairs)
        mask = ~np.eye(n_calls, dtype=bool)
        off_diag_sims = cos_matrix[mask]

        max_cos = float(np.max(off_diag_sims))
        min_cos = float(np.min(off_diag_sims))
        mean_cos = float(np.mean(off_diag_sims))
        std_cos = float(np.std(off_diag_sims))

        # Expected theoretical std in D=256 is 1 / sqrt(256) = 0.0625
        expected_std = 1.0 / math.sqrt(256)

        # Key Challenge Assertions:
        # 1. Max cosine similarity strictly < 0.35 across all 124,750 pairs
        assert max_cos < 0.35, (
            f"Collision detected or orthogonality failed: max cosine similarity = {max_cos:.5f} >= 0.35"
        )

        # 2. Min cosine similarity strictly > -0.35
        assert min_cos > -0.35, (
            f"Excessive negative alignment: min cosine similarity = {min_cos:.5f} <= -0.35"
        )

        # 3. Mean cosine similarity close to 0.0 (near-perfect zero-centering)
        assert abs(mean_cos) < 0.01, (
            f"Distribution bias detected: mean cosine similarity = {mean_cos:.5f} (expected ~0.0)"
        )

        # 4. Standard deviation matches theoretical random hypersphere projection (0.0625 ± 0.015)
        assert abs(std_cos - expected_std) < 0.015, (
            f"Empirical std {std_cos:.4f} deviates from theoretical {expected_std:.4f}"
        )


# ==============================================================================
# 4. Universal Strict L2 Norm Contract Verification
# ==============================================================================

class TestStrictL2NormUniversalEnforcement:
    """Exhaustive check ensuring abs(norm(emb) - 1.0) < 1e-5 holds universally."""

    @pytest.mark.parametrize(
        "dim", [128, 256, 320, 512, 1024, 1280, 2048]
    )
    def test_l2_norm_across_custom_dimensions(self, dim: int):
        """Verifies L2 norm across diverse embedding dimensions."""
        mock = MockBackbone(embedding_dim=dim)
        audio = generate_harmonic_chirp(duration=3.0, sr=48000)
        emb = mock.embed(audio)

        assert emb.shape == (dim,)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Dim {dim} norm = {norm:.8f}"

    def test_l2_norm_across_all_synthetic_fixtures(self, mock_backbone):
        """Verifies L2 norm across every synthetic audio generator."""
        generators = [
            generate_pure_tone(freq=440.0, duration=3.0),
            generate_pure_tone(freq=8000.0, duration=3.0),
            generate_harmonic_chirp(start_freq=300.0, end_freq=3000.0, duration=3.0),
            generate_broadband_noise(noise_type="white", duration=3.0),
            generate_broadband_noise(noise_type="pink", duration=3.0),
            generate_broadband_noise(noise_type="brown", duration=3.0),
            generate_transient_pulse(pulse_rate=50.0, duration=3.0),
            generate_silence(duration=3.0),
            generate_animal_call("aves", duration=3.0),
            generate_animal_call("insecta", duration=3.0),
            generate_animal_call("amphibia", duration=3.0),
            generate_animal_call("mammalia", duration=3.0),
        ]
        for audio in generators:
            emb = mock_backbone.embed(audio)
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Fixture norm violation: {norm:.8f}"

    def test_l2_norm_real_corvus_splendens_audio(
        self, birdnet_backbone, perch_backbone, panns_backbone, mock_backbone
    ):
        """Verifies L2 norm on real Corvus splendens audio file across all backbones."""
        real_wav = Path("data/processed/corvus_splendens/1009327_seg000.wav")
        if not real_wav.exists():
            pytest.skip(f"Real audio {real_wav} not present")

        for b in [mock_backbone, birdnet_backbone, perch_backbone, panns_backbone]:
            emb = b.embed(str(real_wav))
            norm = float(np.linalg.norm(emb))
            err = abs(norm - 1.0)
            assert err < 1e-5, f"{b.name} real audio norm error: {err:.2e}"
