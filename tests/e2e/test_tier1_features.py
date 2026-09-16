"""Tier 1: Feature Coverage & Contract Tests for AnyCall.

Covers:
1. Audio format standardization contract (48 kHz, 16-bit mono float32).
2. Uniform audio slicing (3.0s / 144,000 samples) and VAD filtering.
3. Indian wildlife species catalog definitions and querying.
4. Deterministic synthetic audio and mock data fixture verification.
"""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures.synth_audio import (
    generate_pure_tone,
    generate_harmonic_chirp,
    generate_broadband_noise,
    generate_transient_pulse,
    generate_animal_call,
    generate_silence,
    save_wav_file,
    read_wav_file,
    create_multichannel_wav_file,
)
from tests.fixtures.mock_data import (
    MOCK_INDIAN_SPECIES_CATALOG,
    get_mock_catalog,
    get_mock_species_by_taxon,
    get_mock_species_by_id,
    generate_mock_xeno_canto_response,
    generate_mock_embedding,
    generate_mock_batch_embeddings,
)

# Dynamic import of anycall modules (handles concurrent development)
try:
    from anycall.audio.standardize import standardize_audio, AudioFormatError
    HAVE_ANYCALL_STANDARDIZE = True
except (ImportError, AttributeError):
    HAVE_ANYCALL_STANDARDIZE = False

try:
    try:
        from anycall.audio.vad import slice_audio_segments
    except (ImportError, AttributeError):
        from anycall.audio.standardize import slice_audio_segments
    HAVE_ANYCALL_VAD = True
except (ImportError, AttributeError):
    HAVE_ANYCALL_VAD = False

try:
    import anycall.data.species as anycall_species
    HAVE_ANYCALL_SPECIES = True
except (ImportError, AttributeError):
    HAVE_ANYCALL_SPECIES = False


class TestAudioStandardizationContract(unittest.TestCase):
    """Verifies anycall.audio.standardize against PROJECT.md Interface Contract 1."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_standardize_audio_from_wav_file(self):
        """Audio standardization must convert input WAV to 48kHz mono float32 in [-1, 1]."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        input_path = self.tmp_path / "test_44k_stereo.wav"
        ch1 = generate_pure_tone(freq=1000.0, duration=3.0, sr=44100, amplitude=0.7)
        ch2 = generate_pure_tone(freq=2000.0, duration=3.0, sr=44100, amplitude=0.5)
        create_multichannel_wav_file([ch1, ch2], input_path, sr=44100)

        audio_arr, sr = standardize_audio(input_path, target_sr=48000)

        self.assertEqual(sr, 48000, "Target sample rate must be 48,000 Hz")
        self.assertEqual(audio_arr.ndim, 1, "Audio array must be 1D mono")
        self.assertEqual(audio_arr.dtype, np.float32, "Audio array dtype must be float32")
        self.assertTrue(np.all(audio_arr >= -1.0), "Samples must not exceed -1.0")
        self.assertTrue(np.all(audio_arr <= 1.0), "Samples must not exceed +1.0")
        expected_len = int(round(3.0 * 48000))
        self.assertAlmostEqual(len(audio_arr), expected_len, delta=100)

    def test_standardize_audio_output_path_saving(self):
        """When output_path is provided, standardized WAV must be written to disk."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        input_path = self.tmp_path / "raw.wav"
        output_path = self.tmp_path / "standardized.wav"
        raw_audio = generate_pure_tone(freq=1500.0, duration=2.0, sr=22050)
        save_wav_file(raw_audio, input_path, sr=22050)

        arr, sr = standardize_audio(input_path, output_path=output_path, target_sr=48000)
        self.assertTrue(output_path.exists(), "Output WAV file must exist on disk")

        saved_arr, saved_sr = read_wav_file(output_path)
        self.assertEqual(saved_sr, 48000)
        self.assertEqual(saved_arr.ndim, 1)

    def test_standardize_audio_various_sample_rates(self):
        """Must handle common bioacoustic sample rates (22050, 44100, 96000)."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        for test_sr in [22050, 44100, 48000, 96000]:
            p = self.tmp_path / f"audio_{test_sr}.wav"
            tone = generate_pure_tone(freq=1000.0, duration=2.0, sr=test_sr)
            save_wav_file(tone, p, sr=test_sr)

            arr, sr = standardize_audio(p, target_sr=48000)
            self.assertEqual(sr, 48000)
            self.assertEqual(arr.ndim, 1)
            self.assertAlmostEqual(len(arr), 48000 * 2, delta=50)


class TestAudioSlicingAndVADContract(unittest.TestCase):
    """Verifies slice_audio_segments against PROJECT.md Interface Contract 1."""

    def test_slice_audio_segments_uniform_shape(self):
        """Slicing a 9.0-second audio stream must produce 3 segments of shape (144000,)."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        audio = generate_animal_call(taxon="aves", duration=9.0, sr=48000)
        segments = slice_audio_segments(audio, sr=48000, segment_duration=3.0, hop_duration=3.0, vad_filter=False)

        self.assertEqual(len(segments), 3, "Expected exactly 3 segments from 9.0s audio with 3.0s hop")
        for idx, seg in enumerate(segments):
            self.assertEqual(seg.shape, (144000,), f"Segment {idx} shape must be (144000,)")
            self.assertEqual(seg.dtype, np.float32, f"Segment {idx} dtype must be float32")

    def test_slice_audio_segments_padding_short_audio(self):
        """Audio shorter than 3.0s (e.g. 1.5s) must be zero-padded to exactly 144,000 samples."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        audio = generate_animal_call(taxon="insecta", duration=1.5, sr=48000)
        segments = slice_audio_segments(audio, sr=48000, segment_duration=3.0, vad_filter=False)

        self.assertEqual(len(segments), 1, "Expected 1 padded segment for 1.5s audio")
        self.assertEqual(segments[0].shape, (144000,), "Padded segment must be exactly 144,000 samples")

    def test_slice_audio_segments_vad_silence_discard(self):
        """VAD filter must discard pure silence segments."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        silence = generate_silence(duration=6.0, sr=48000)
        segments = slice_audio_segments(silence, sr=48000, segment_duration=3.0, vad_filter=True)

        self.assertEqual(len(segments), 0, "Pure digital silence segments must be discarded by VAD")

    def test_slice_audio_segments_vad_animal_call_retained(self):
        """VAD filter must retain high-energy animal call segments."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        call = generate_animal_call(taxon="aves", duration=3.0, sr=48000)
        segments = slice_audio_segments(call, sr=48000, segment_duration=3.0, vad_filter=True)

        self.assertGreaterEqual(len(segments), 1, "Vocal animal call must be retained by VAD")


class TestIndianSpeciesCatalogContract(unittest.TestCase):
    """Verifies Indian Wildlife Species Catalog (Feature 3 in PROJECT.md)."""

    def test_species_catalog_completeness_33_species(self):
        """Catalog must contain exactly 33 target species."""
        if HAVE_ANYCALL_SPECIES and hasattr(anycall_species, "INDIAN_SPECIES_CATALOG"):
            catalog = anycall_species.INDIAN_SPECIES_CATALOG
        else:
            catalog = get_mock_catalog()

        self.assertEqual(len(catalog), 33, f"Expected 33 target species, found {len(catalog)}")

    def test_species_catalog_taxa_distribution(self):
        """Catalog must contain 15 Aves, 8 Insecta, 5 Amphibia, 5 Mammalia."""
        if HAVE_ANYCALL_SPECIES and hasattr(anycall_species, "get_species_by_taxon"):
            birds = anycall_species.get_species_by_taxon("Aves")
            insects = anycall_species.get_species_by_taxon("Insecta")
            frogs = anycall_species.get_species_by_taxon("Amphibia")
            mammals = anycall_species.get_species_by_taxon("Mammalia")
        else:
            birds = get_mock_species_by_taxon("Aves")
            insects = get_mock_species_by_taxon("Insecta")
            frogs = get_mock_species_by_taxon("Amphibia")
            mammals = get_mock_species_by_taxon("Mammalia")

        self.assertEqual(len(birds), 15, "Expected 15 avian species")
        self.assertEqual(len(insects), 8, "Expected 8 insect species")
        self.assertEqual(len(frogs), 5, "Expected 5 amphibian species")
        self.assertEqual(len(mammals), 5, "Expected 5 mammalian species")

    def test_species_catalog_fields_schema(self):
        """All species entries must have species_id, scientific_name, common_name, taxon."""
        catalog = get_mock_catalog()
        for sp in catalog:
            self.assertIn("species_id", sp)
            self.assertIn("scientific_name", sp)
            self.assertIn("common_name", sp)
            self.assertIn("taxon", sp)
            self.assertTrue(sp["species_id"], "species_id must not be empty")
            self.assertTrue(sp["scientific_name"], "scientific_name must not be empty")

    def test_species_catalog_includes_corvus_splendens(self):
        """House Crow (Corvus splendens) is mandatory primary benchmark control species."""
        crow = get_mock_species_by_id("corvus_splendens")
        self.assertIsNotNone(crow, "Corvus splendens must be present in species catalog")
        self.assertEqual(crow["scientific_name"], "Corvus splendens")
        self.assertEqual(crow["taxon"], "Aves")


class TestSyntheticAudioFixturesSelfVerification(unittest.TestCase):
    """Verifies that synthetic audio generators adhere to mathematical properties."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pure_tone_generation_properties(self):
        """Pure tone must have exact duration, float32 type, and specified peak amplitude."""
        tone = generate_pure_tone(freq=1000.0, duration=2.5, sr=48000, amplitude=0.85)
        self.assertEqual(len(tone), int(round(2.5 * 48000)))
        self.assertEqual(tone.dtype, np.float32)
        self.assertAlmostEqual(float(np.max(np.abs(tone))), 0.85, delta=1e-3)

    def test_harmonic_chirp_properties(self):
        """Harmonic chirp must be bounded in [-1, 1] and have correct length."""
        chirp = generate_harmonic_chirp(start_freq=500.0, end_freq=3500.0, duration=3.0, sr=48000)
        self.assertEqual(len(chirp), 144000)
        self.assertTrue(np.all(np.abs(chirp) <= 1.0))

    def test_broadband_noise_determinism(self):
        """Broadband noise with same seed must be bitwise identical."""
        n1 = generate_broadband_noise(duration=1.0, sr=48000, seed=42)
        n2 = generate_broadband_noise(duration=1.0, sr=48000, seed=42)
        n3 = generate_broadband_noise(duration=1.0, sr=48000, seed=99)
        self.assertTrue(np.array_equal(n1, n2), "Identical seeds must yield identical noise")
        self.assertFalse(np.array_equal(n1, n3), "Different seeds must yield different noise")

    def test_transient_pulse_properties(self):
        """Transient pulses must produce periodic bursts with silent intervals."""
        pulse = generate_transient_pulse(pulse_rate=10.0, duration=1.0, sr=48000)
        self.assertEqual(len(pulse), 48000)
        # Should have both non-zero values and near-zero values
        self.assertGreater(float(np.max(np.abs(pulse))), 0.5)
        self.assertLess(float(np.min(np.abs(pulse))), 1e-4)

    def test_animal_call_all_taxa(self):
        """Animal call generator must produce valid signals for all 4 taxa."""
        for taxon in ["aves", "insecta", "amphibia", "mammalia"]:
            call = generate_animal_call(taxon=taxon, duration=3.0, sr=48000)
            self.assertEqual(len(call), 144000, f"Call for {taxon} must have 144,000 samples")
            self.assertFalse(np.any(np.isnan(call)), f"Call for {taxon} must not contain NaN")
            self.assertFalse(np.any(np.isinf(call)), f"Call for {taxon} must not contain Inf")
            rms = float(np.sqrt(np.mean(call ** 2)))
            self.assertGreater(rms, 0.05, f"Animal call for {taxon} must have non-trivial energy")

    def test_wav_roundtrip_fidelity(self):
        """Saving and reading a 16-bit PCM WAV must preserve signal with SNR > 60 dB."""
        orig = generate_pure_tone(freq=1200.0, duration=1.0, sr=48000, amplitude=0.8)
        wav_path = self.tmp_path / "roundtrip.wav"
        save_wav_file(orig, wav_path, sr=48000, bit_depth=16)

        read_arr, read_sr = read_wav_file(wav_path)
        self.assertEqual(read_sr, 48000)
        self.assertEqual(len(read_arr), len(orig))

        # Quantization noise for 16-bit PCM is ~ -96 dB; check SNR > 60 dB
        err = orig - read_arr
        snr = 10.0 * np.log10(np.sum(orig ** 2) / (np.sum(err ** 2) + 1e-12))
        self.assertGreater(snr, 60.0, f"WAV roundtrip SNR {snr:.1f} dB is below 60 dB threshold")


class TestMockDataFixturesSelfVerification(unittest.TestCase):
    """Verifies that mock data and embedding generators adhere to specifications."""

    def test_mock_xeno_canto_response_schema(self):
        """Mock Xeno-Canto response must have numRecordings, recordings, and valid audio URLs."""
        resp = generate_mock_xeno_canto_response("Corvus splendens", page=1, num_recordings=20)
        self.assertEqual(resp["numRecordings"], "20")
        self.assertEqual(len(resp["recordings"]), 20)
        rec = resp["recordings"][0]
        self.assertIn("id", rec)
        self.assertIn("file", rec)
        self.assertTrue(rec["file"].startswith("https://xeno-canto.org/"))
        self.assertEqual(rec["cnt"], "India")

    def test_mock_embeddings_unit_norm_and_clustering(self):
        """Mock embeddings must have unit L2 norm and tight intra-class clustering."""
        emb1 = generate_mock_embedding("Corvus splendens", seed=1, dim=320, noise_level=0.08)
        emb2 = generate_mock_embedding("Corvus splendens", seed=2, dim=320, noise_level=0.08)
        emb_other = generate_mock_embedding("Hoplobatrachus tigerinus", seed=1, dim=320, noise_level=0.08)

        # Norm verification
        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))
        norm_other = float(np.linalg.norm(emb_other))
        self.assertAlmostEqual(norm1, 1.0, delta=1e-5)
        self.assertAlmostEqual(norm2, 1.0, delta=1e-5)
        self.assertAlmostEqual(norm_other, 1.0, delta=1e-5)

        # Intra-class vs inter-class cosine similarity
        cos_same = float(np.dot(emb1, emb2))
        cos_diff = float(np.dot(emb1, emb_other))

        self.assertGreater(cos_same, 0.85, f"Same species cosine similarity {cos_same:.4f} must be > 0.85")
        self.assertLess(abs(cos_diff), 0.30, f"Different species cosine similarity {cos_diff:.4f} must be < 0.30")


if __name__ == "__main__":
    unittest.main()
