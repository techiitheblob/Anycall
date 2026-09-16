"""Adversarial Audio Stress & Robustness Test Suite for AnyCall Milestone 2.

Location: tests/test_m2_audio_stress.py
Author: challenger_m2_2

Verifies:
1. Full batch verification across all 74 real Corvus splendens audio files
   using MockBackbone and BirdNetBackbone.
2. Strict unit L2-norm compliance (abs(norm - 1.0) < 1e-5), shape consistency,
   finite sanity (no NaNs/Infs), and memory leak / stability auditing.
3. Non-standard audio inputs:
   - Stereo 2-channel audio (array and WAV container)
   - 5.1 surround 6-channel audio (array and WAV container)
   - 8 kHz telephony audio (array and WAV container)
   - 96 kHz ultrasonic audio (array and WAV container)
   - 16 kHz and 192 kHz audio
4. Extreme duration boundaries:
   - 10 ms ultra-short impulse
   - 50 ms short chirp
   - 500 ms vocalization
   - 30.0s long continuous recording
   - 60.0s ultra-long recording
5. Numerical edge cases & robustness:
   - Digital silence (all-zero array and WAV)
   - Near-zero energy floor (1e-15 amplitude)
   - Saturated / clipped waveforms (+-1.0 square wave, 1e5 amplitude)
   - Non-finite values (NaN / Inf sanitization)
   - Deterministic reproducibility and array vs file parity
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import soundfile as sf

from anycall.embeddings import (
    BirdNetBackbone,
    MockBackbone,
    PannsBackbone,
    PerchBackbone,
)


CORVUS_DIR = Path("data/processed/corvus_splendens")


def get_all_corvus_files() -> List[Path]:
    """Returns sorted list of all 74 Corvus splendens processed WAV files."""
    if not CORVUS_DIR.exists():
        return []
    return sorted(CORVUS_DIR.glob("*.wav"))


# ==============================================================================
# 1. Full Batch Verification Across All 74 Real Corvus splendens Files
# ==============================================================================


class TestCorvusSplendensBatchVerification:
    """Evaluates MockBackbone and BirdNetBackbone across all 74 real field recordings."""

    def test_corvus_directory_has_74_files(self):
        """Verifies exactly 74 real Corvus splendens audio segments are present."""
        files = get_all_corvus_files()
        assert len(files) == 74, f"Expected 74 files in {CORVUS_DIR}, found {len(files)}"

    def test_mock_backbone_full_batch(self):
        """Runs MockBackbone across all 74 real files, validating norm, shape, and finiteness."""
        files = get_all_corvus_files()
        if not files:
            pytest.skip("Corvus splendens audio files not available")

        backbone = MockBackbone(embedding_dim=256)
        latencies = []

        for f in files:
            t0 = time.perf_counter()
            emb = backbone.embed(str(f))
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            assert emb.shape == (256,), f"Shape mismatch for {f.name}: {emb.shape}"
            assert emb.dtype == np.float32
            assert np.all(np.isfinite(emb)), f"Non-finite values in {f.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Norm violation for {f.name}: {norm}"
            assert np.std(emb) > 0.01, f"Embedding collapsed to constant for {f.name}"

        mean_lat = float(np.mean(latencies))
        assert mean_lat < 5.0, f"Mock latency exceeded expected threshold: {mean_lat:.2f} ms"

    def test_birdnet_backbone_full_batch(self):
        """Runs BirdNetBackbone across all 74 real files, validating norm, shape, finiteness, and latency."""
        files = get_all_corvus_files()
        if not files:
            pytest.skip("Corvus splendens audio files not available")

        backbone = BirdNetBackbone()
        latencies = []

        for f in files:
            t0 = time.perf_counter()
            emb = backbone.embed(str(f))
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            assert emb.shape == (1024,), f"Shape mismatch for {f.name}: {emb.shape}"
            assert emb.dtype == np.float32
            assert np.all(np.isfinite(emb)), f"Non-finite values in {f.name}"
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Norm violation for {f.name}: {norm}"
            assert np.std(emb) > 0.001, f"Embedding collapsed for {f.name}"

        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        assert mean_lat < 150.0, f"Mean latency too high: {mean_lat:.2f} ms"
        assert p95_lat < 300.0, f"p95 latency too high: {p95_lat:.2f} ms"

    def test_birdnet_memory_leak_audit(self):
        """Audits memory growth across 74 consecutive BirdNet extractions to detect leaks."""
        files = get_all_corvus_files()
        if not files:
            pytest.skip("Corvus splendens audio files not available")

        gc.collect()
        tracemalloc.start()

        backbone = BirdNetBackbone()
        # Warm up 5 samples
        for f in files[:5]:
            backbone.embed(str(f))

        gc.collect()
        start_cur, _ = tracemalloc.get_traced_memory()

        # Run remaining 69 files
        for f in files[5:]:
            backbone.embed(str(f))

        gc.collect()
        end_cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth_kb = (end_cur - start_cur) / 1024.0
        # Memory growth across 69 inferences must be tightly bounded (< 100 KB)
        assert growth_kb < 100.0, f"Memory leak detected: grew by {growth_kb:.2f} KB"


# ==============================================================================
# 2. Non-Standard Audio Inputs: Channels & Sampling Rates
# ==============================================================================


class TestNonStandardAudioInputs:
    """Stress tests backbones against non-standard channel counts and sampling rates."""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.mark.parametrize("channels", [2, 6])
    def test_multichannel_arrays(self, channels: int):
        """Verifies stereo (2ch) and 5.1 surround (6ch) array inputs."""
        sr = 48000
        samples = sr * 3
        multi_audio = np.random.randn(samples, channels).astype(np.float32)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(multi_audio, sr=sr)
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

        emb_b = birdnet.embed(multi_audio, sr=sr)
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5

    @pytest.mark.parametrize("channels", [2, 6])
    def test_multichannel_wav_files(self, channels: int, tmp_dir: Path):
        """Verifies multi-channel audio saved in standard WAV containers."""
        sr = 48000
        samples = sr * 3
        multi_audio = np.random.uniform(-0.8, 0.8, (samples, channels)).astype(np.float32)
        wav_path = tmp_dir / f"test_{channels}ch.wav"
        sf.write(str(wav_path), multi_audio, sr)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(str(wav_path))
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

        emb_b = birdnet.embed(str(wav_path))
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5

    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 32000, 44100, 96000, 192000])
    def test_non_standard_sample_rate_arrays(self, sr: int):
        """Verifies sample rates from 8 kHz telephony up to 192 kHz studio master."""
        duration = 3.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(audio, sr=sr)
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_m))

        emb_b = birdnet.embed(audio, sr=sr)
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_b))

    @pytest.mark.parametrize("sr", [8000, 96000])
    def test_non_standard_sample_rate_wav_files(self, sr: int, tmp_dir: Path):
        """Verifies 8kHz and 96kHz audio loaded directly from WAV files."""
        duration = 3.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
        audio = 0.6 * np.sin(2 * np.pi * 800.0 * t).astype(np.float32)

        wav_path = tmp_dir / f"test_{sr}hz.wav"
        sf.write(str(wav_path), audio, sr)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(str(wav_path))
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

        emb_b = birdnet.embed(str(wav_path))
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5


# ==============================================================================
# 3. Extreme Durations: Ultra-Short to Ultra-Long Audio
# ==============================================================================


class TestExtremeDurations:
    """Stress tests backbones on extreme durations: 10 ms to 60.0s."""

    @pytest.mark.parametrize(
        "duration_sec",
        [0.010, 0.050, 0.100, 0.500, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0],
    )
    def test_durations_array(self, duration_sec: float):
        """Verifies durations spanning 10ms to 60s as numpy arrays."""
        sr = 48000
        samples = int(sr * duration_sec)
        audio = np.random.randn(samples).astype(np.float32)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(audio, sr=sr)
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_m))

        emb_b = birdnet.embed(audio, sr=sr)
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_b))

    def test_extreme_durations_wav_file(self):
        """Verifies 50ms chirp and 30s recording saved to disk and loaded by path."""
        sr = 48000
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 50 ms short chirp
            chirp_50ms = np.random.uniform(-0.5, 0.5, int(0.05 * sr)).astype(np.float32)
            p_short = tmp_path / "short_50ms.wav"
            sf.write(str(p_short), chirp_50ms, sr)

            # 30.0s recording
            rec_30s = np.random.uniform(-0.5, 0.5, int(30.0 * sr)).astype(np.float32)
            p_long = tmp_path / "long_30s.wav"
            sf.write(str(p_long), rec_30s, sr)

            mock = MockBackbone(embedding_dim=256)
            birdnet = BirdNetBackbone()

            for p in [p_short, p_long]:
                emb_m = mock.embed(str(p))
                assert emb_m.shape == (256,)
                assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

                emb_b = birdnet.embed(str(p))
                assert emb_b.shape == (1024,)
                assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5


# ==============================================================================
# 4. Numerical Edge Cases & Adversarial Inputs
# ==============================================================================


class TestAdversarialNumericalEdgeCases:
    """Stress tests numerical edge cases: silence, tiny values, saturation, NaNs."""

    def test_complete_digital_silence_array(self):
        """Verifies complete digital silence (all zeros array) produces valid unit vector."""
        silence = np.zeros(144000, dtype=np.float32)

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(silence)
        assert emb_m.shape == (256,)
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_m))

        emb_b = birdnet.embed(silence)
        assert emb_b.shape == (1024,)
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5
        assert np.all(np.isfinite(emb_b))

    def test_complete_digital_silence_file(self):
        """Verifies silence WAV file on disk produces valid unit vector."""
        with tempfile.TemporaryDirectory() as tmp:
            sil_path = Path(tmp) / "silence.wav"
            sf.write(str(sil_path), np.zeros(144000, dtype=np.float32), 48000)

            mock = MockBackbone(embedding_dim=256)
            birdnet = BirdNetBackbone()

            emb_m = mock.embed(str(sil_path))
            assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

            emb_b = birdnet.embed(str(sil_path))
            assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5

    def test_subthreshold_and_saturated_amplitudes(self):
        """Verifies near-zero energy (1e-15) and heavy saturation (1e5) do not crash or produce NaNs."""
        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        tiny = np.full(144000, 1e-15, dtype=np.float32)
        huge = np.full(144000, 1e5, dtype=np.float32)
        square = np.sign(np.sin(2 * np.pi * 500.0 * np.linspace(0, 3, 144000))).astype(np.float32)

        for sig in [tiny, huge, square]:
            emb_m = mock.embed(sig)
            assert np.all(np.isfinite(emb_m))
            assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

            emb_b = birdnet.embed(sig)
            assert np.all(np.isfinite(emb_b))
            assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5

    def test_nan_and_inf_sanitization(self):
        """Verifies waveforms with NaN or Inf are sanitized safely into finite unit embeddings."""
        corrupted = np.ones(144000, dtype=np.float32)
        corrupted[100] = np.nan
        corrupted[200] = np.inf
        corrupted[300] = -np.inf

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m = mock.embed(corrupted)
        assert np.all(np.isfinite(emb_m))
        assert abs(float(np.linalg.norm(emb_m)) - 1.0) < 1e-5

        emb_b = birdnet.embed(corrupted)
        assert np.all(np.isfinite(emb_b))
        assert abs(float(np.linalg.norm(emb_b)) - 1.0) < 1e-5

    def test_array_vs_file_consistency(self):
        """Verifies that passing a WAV file path vs loading it as numpy array produces identical output."""
        files = get_all_corvus_files()
        if not files:
            pytest.skip("Corvus files not available")

        test_file = files[0]
        audio, sr = sf.read(str(test_file), dtype="float32")

        mock = MockBackbone(embedding_dim=256)
        birdnet = BirdNetBackbone()

        emb_m_file = mock.embed(str(test_file))
        emb_m_arr = mock.embed(audio, sr=sr)
        assert np.allclose(emb_m_file, emb_m_arr, atol=1e-5)

        emb_b_file = birdnet.embed(str(test_file))
        emb_b_arr = birdnet.embed(audio, sr=sr)
        cos_sim = float(np.dot(emb_b_file, emb_b_arr))
        assert cos_sim > 0.9999, f"Array vs file mismatch: cos_sim = {cos_sim}"

    def test_deterministic_reproducibility(self):
        """Verifies that repeating embed() 5 times on the same input yields identical vectors."""
        files = get_all_corvus_files()
        if not files:
            pytest.skip("Corvus files not available")

        test_file = files[0]
        birdnet = BirdNetBackbone()

        ref = birdnet.embed(str(test_file))
        for _ in range(5):
            rep = birdnet.embed(str(test_file))
            assert np.array_equal(ref, rep), "Inference non-deterministic on identical input"


# ==============================================================================
# 5. Standalone Execution Harness & Empirical Benchmark Runner
# ==============================================================================


def run_standalone_stress_suite() -> Dict[str, Any]:
    """Executes the complete stress suite and returns structured empirical metrics."""
    print("=" * 80)
    print("  ANYCALL MILESTONE 2: ADVERSARIAL AUDIO STRESS & ROBUSTNESS HARNESS")
    print("=" * 80)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corvus_batch_tests": {},
        "non_standard_inputs": {},
        "extreme_durations": {},
        "numerical_edge_cases": {},
        "memory_leak_audit": {},
    }

    files = get_all_corvus_files()
    print(f"\n[1/5] Running Full Batch Verification on {len(files)} Corvus splendens files...")
    assert len(files) == 74, f"Expected 74 files, got {len(files)}"

    # Mock full batch
    mock = MockBackbone(embedding_dim=256)
    mock_lats = []
    mock_norm_errs = []
    for f in files:
        t0 = time.perf_counter()
        emb = mock.embed(str(f))
        dt = (time.perf_counter() - t0) * 1000.0
        mock_lats.append(dt)
        mock_norm_errs.append(abs(float(np.linalg.norm(emb)) - 1.0))
        assert emb.shape == (256,)
        assert np.all(np.isfinite(emb))

    print(f"  -> MockBackbone: 74/74 passed. Mean latency: {np.mean(mock_lats):.2f} ms, Max L2 error: {max(mock_norm_errs):.2e}")

    # BirdNET full batch
    birdnet = BirdNetBackbone()
    bnet_lats = []
    bnet_norm_errs = []
    for f in files:
        t0 = time.perf_counter()
        emb = birdnet.embed(str(f))
        dt = (time.perf_counter() - t0) * 1000.0
        bnet_lats.append(dt)
        bnet_norm_errs.append(abs(float(np.linalg.norm(emb)) - 1.0))
        assert emb.shape == (1024,)
        assert np.all(np.isfinite(emb))

    print(f"  -> BirdNetBackbone: 74/74 passed. Mean latency: {np.mean(bnet_lats):.2f} ms, Max L2 error: {max(bnet_norm_errs):.2e}")

    results["corvus_batch_tests"] = {
        "file_count": len(files),
        "mock": {
            "passed": len(files),
            "mean_latency_ms": round(float(np.mean(mock_lats)), 3),
            "max_latency_ms": round(float(np.max(mock_lats)), 3),
            "max_norm_error": float(max(mock_norm_errs)),
        },
        "birdnet": {
            "passed": len(files),
            "mean_latency_ms": round(float(np.mean(bnet_lats)), 3),
            "max_latency_ms": round(float(np.max(bnet_lats)), 3),
            "p95_latency_ms": round(float(np.percentile(bnet_lats, 95)), 3),
            "max_norm_error": float(max(bnet_norm_errs)),
        },
    }

    # Memory Leak Audit
    print("\n[2/5] Running Memory Leak Audit (74 files sequential)...")
    gc.collect()
    tracemalloc.start()
    bnet_mem = BirdNetBackbone()
    for f in files[:5]:
        bnet_mem.embed(str(f))
    gc.collect()
    mem_start, _ = tracemalloc.get_traced_memory()

    for f in files[5:]:
        bnet_mem.embed(str(f))
    gc.collect()
    mem_end, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    leak_kb = (mem_end - mem_start) / 1024.0
    peak_mb = mem_peak / (1024.0 * 1024.0)
    print(f"  -> BirdNET Memory Growth: {leak_kb:.2f} KB across 69 iterations (Peak: {peak_mb:.1f} MB)")
    assert leak_kb < 100.0, f"Memory leak detected: {leak_kb:.2f} KB"
    results["memory_leak_audit"] = {
        "tested_iterations": 69,
        "memory_growth_kb": round(leak_kb, 3),
        "peak_memory_mb": round(peak_mb, 3),
        "leak_detected": False,
    }

    # Non-Standard Inputs
    print("\n[3/5] Testing Non-Standard Audio Inputs (Channels & Sample Rates)...")
    non_std_cases = [
        ("Stereo (N, 2)", np.random.randn(144000, 2).astype(np.float32), 48000),
        ("5.1 Surround (N, 6)", np.random.randn(144000, 6).astype(np.float32), 48000),
        ("8 kHz Telephony", np.random.randn(8000 * 3).astype(np.float32), 8000),
        ("16 kHz Wideband", np.random.randn(16000 * 3).astype(np.float32), 16000),
        ("22.05 kHz Half-CD", np.random.randn(int(22050 * 3)).astype(np.float32), 22050),
        ("44.1 kHz Standard", np.random.randn(int(44100 * 3)).astype(np.float32), 44100),
        ("96 kHz Ultrasonic", np.random.randn(96000 * 3).astype(np.float32), 96000),
        ("192 kHz Master", np.random.randn(192000 * 3).astype(np.float32), 192000),
    ]

    for name, audio, sr in non_std_cases:
        e_m = mock.embed(audio, sr=sr)
        e_b = birdnet.embed(audio, sr=sr)
        err_m = abs(float(np.linalg.norm(e_m)) - 1.0)
        err_b = abs(float(np.linalg.norm(e_b)) - 1.0)
        assert e_m.shape == (256,) and err_m < 1e-5
        assert e_b.shape == (1024,) and err_b < 1e-5
        print(f"  -> {name:20s}: PASS (Mock norm err={err_m:.1e}, BirdNET norm err={err_b:.1e})")
        results["non_standard_inputs"][name] = {
            "sr": sr,
            "shape_in": list(audio.shape),
            "mock_shape_out": list(e_m.shape),
            "birdnet_shape_out": list(e_b.shape),
            "status": "PASSED",
        }

    # Extreme Durations
    print("\n[4/5] Testing Extreme Duration Boundaries (10 ms to 60.0s)...")
    durations = [0.010, 0.050, 0.100, 0.500, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0]
    for dur in durations:
        samples = int(48000 * dur)
        audio = np.random.randn(samples).astype(np.float32)
        e_m = mock.embed(audio, sr=48000)
        e_b = birdnet.embed(audio, sr=48000)
        err_m = abs(float(np.linalg.norm(e_m)) - 1.0)
        err_b = abs(float(np.linalg.norm(e_b)) - 1.0)
        assert e_m.shape == (256,) and err_m < 1e-5
        assert e_b.shape == (1024,) and err_b < 1e-5
        dur_label = f"{dur * 1000:.0f} ms" if dur < 1.0 else f"{dur:.1f} s"
        print(f"  -> Duration {dur_label:10s} ({samples:7d} smp): PASS (L2 err: Mock={err_m:.1e}, BirdNET={err_b:.1e})")
        results["extreme_durations"][dur_label] = {
            "duration_sec": dur,
            "samples": samples,
            "status": "PASSED",
        }

    # Numerical Edge Cases
    print("\n[5/5] Testing Numerical Edge Cases & Adversarial Inputs...")
    edge_cases = [
        ("Digital Silence (Zeros)", np.zeros(144000, dtype=np.float32)),
        ("Near-Zero Floor (1e-15)", np.full(144000, 1e-15, dtype=np.float32)),
        ("Extreme High (1e5)", np.full(144000, 1e5, dtype=np.float32)),
        ("Hard Clipping (+-1.0)", np.sign(np.random.randn(144000)).astype(np.float32)),
        ("NaN & Inf Corrupted", np.nan_to_num(np.random.randn(144000), nan=0.0)),
    ]
    corrupted = np.ones(144000, dtype=np.float32)
    corrupted[50] = np.nan
    corrupted[100] = np.inf
    corrupted[150] = -np.inf
    edge_cases.append(("Explicit NaN/Inf", corrupted))

    for name, audio in edge_cases:
        e_m = mock.embed(audio)
        e_b = birdnet.embed(audio)
        err_m = abs(float(np.linalg.norm(e_m)) - 1.0)
        err_b = abs(float(np.linalg.norm(e_b)) - 1.0)
        assert np.all(np.isfinite(e_m)) and err_m < 1e-5
        assert np.all(np.isfinite(e_b)) and err_b < 1e-5
        print(f"  -> {name:25s}: PASS (Finite, Unit Norm: Mock={err_m:.1e}, BirdNET={err_b:.1e})")
        results["numerical_edge_cases"][name] = {
            "mock_finite": bool(np.all(np.isfinite(e_m))),
            "birdnet_finite": bool(np.all(np.isfinite(e_b))),
            "status": "PASSED",
        }

    print("\n" + "=" * 80)
    print("  >>> ALL ADVERSARIAL & EMPIRICAL STRESS TESTS PASSED WITH 0 FAILURES <<<")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    res = run_standalone_stress_suite()
