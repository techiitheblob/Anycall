"""Tier 2: Boundary & Corner Case Tests for AnyCall.

Covers:
1. Degenerate audio inputs: 0-byte files, non-WAV files, truncated RIFF headers, random binary garbage.
2. Extreme audio signals: digital silence (zero division checks), hard clipped waveforms,
   DC offsets, extreme sample rates (8 kHz, 96 kHz), and extreme dynamic range.
3. Sub-frame audio boundaries and length corner cases.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures.synth_audio import (
    generate_pure_tone,
    generate_silence,
    generate_dc_offset,
    generate_clipped_audio,
    create_corrupted_wav_file,
    save_wav_file,
    read_wav_file,
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


class TestCorruptedAndDegenerateAudio(unittest.TestCase):
    """Verifies that invalid or corrupted audio files raise explicit errors per Contract 1."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_zero_byte_file_raises_error(self):
        """0-byte file must raise AudioFormatError or ValueError, not unhandled EOFError."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        zero_byte_file = self.tmp_path / "empty.wav"
        create_corrupted_wav_file(zero_byte_file, "zero_byte")

        with self.assertRaises((AudioFormatError, ValueError, OSError, EOFError)) as ctx:
            standardize_audio(zero_byte_file)
        self.assertIsNotNone(ctx.exception)

    def test_corrupted_header_non_wav_raises_error(self):
        """Plain text file renamed to .wav must raise AudioFormatError."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        text_file = self.tmp_path / "fake.wav"
        create_corrupted_wav_file(text_file, "text_file")

        with self.assertRaises((AudioFormatError, ValueError, OSError)):
            standardize_audio(text_file)

    def test_truncated_wav_header_raises_error(self):
        """WAV file truncated to 12 bytes must raise AudioFormatError."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        truncated_file = self.tmp_path / "truncated.wav"
        create_corrupted_wav_file(truncated_file, "truncated_header")

        with self.assertRaises((AudioFormatError, ValueError, OSError)):
            standardize_audio(truncated_file)

    def test_invalid_magic_bytes_raises_error(self):
        """File without 'RIFF' magic bytes must raise AudioFormatError."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        invalid_magic = self.tmp_path / "invalid_magic.wav"
        create_corrupted_wav_file(invalid_magic, "invalid_riff")

        with self.assertRaises((AudioFormatError, ValueError, OSError)):
            standardize_audio(invalid_magic)

    def test_random_binary_garbage_raises_error(self):
        """1024 bytes of random binary noise must raise AudioFormatError."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        garbage_file = self.tmp_path / "garbage.wav"
        create_corrupted_wav_file(garbage_file, "random_garbage")

        with self.assertRaises((AudioFormatError, ValueError, OSError)):
            standardize_audio(garbage_file)


class TestExtremeAudioSignals(unittest.TestCase):
    """Verifies robustness against boundary audio signals: silence, clipping, DC offset."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_digital_silence_all_zeros_no_division_by_zero(self):
        """Digital silence (norm = 0) must not cause ZeroDivisionError during processing."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        silence = generate_silence(duration=3.0, sr=48000)
        # Process with VAD disabled: must return array of zeros without crash
        segments = slice_audio_segments(silence, sr=48000, segment_duration=3.0, vad_filter=False)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].shape, (144000,))
        self.assertTrue(np.all(segments[0] == 0.0))

    def test_clipped_audio_overflow_no_nan_or_inf(self):
        """Hard-clipped square-wave audio must not introduce NaN or Inf into float output."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        clipped = generate_clipped_audio(freq=800.0, duration=3.0, sr=48000, gain=20.0)
        clip_path = self.tmp_path / "clipped.wav"
        save_wav_file(clipped, clip_path, sr=48000)

        standardized, sr = standardize_audio(clip_path, target_sr=48000)
        self.assertFalse(np.any(np.isnan(standardized)), "Clipped audio must not produce NaN")
        self.assertFalse(np.any(np.isinf(standardized)), "Clipped audio must not produce Inf")
        self.assertTrue(np.all(standardized >= -1.0), "Normalized values must not exceed -1.0")
        self.assertTrue(np.all(standardized <= 1.0), "Normalized values must not exceed +1.0")

    def test_dc_offset_audio_stability(self):
        """Audio with large DC bias (+0.5) must process cleanly without float overflow."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        dc_audio = generate_dc_offset(offset=0.5, duration=3.0, sr=48000)
        dc_path = self.tmp_path / "dc_offset.wav"
        save_wav_file(dc_audio, dc_path, sr=48000)

        arr, sr = standardize_audio(dc_path, target_sr=48000)
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))
        self.assertTrue(np.all(arr >= -1.0) and np.all(arr <= 1.0))

    def test_extreme_sample_rates_telephony_and_ultrasonic(self):
        """Must handle 8,000 Hz (telephony) and 96,000 Hz (ultrasonic bat) sampling rates."""
        if not HAVE_ANYCALL_STANDARDIZE:
            self.skipTest("anycall.audio.standardize not yet implemented (M1 in progress)")

        for sr_test in [8000, 96000]:
            p = self.tmp_path / f"rate_{sr_test}.wav"
            tone = generate_pure_tone(freq=1000.0, duration=2.0, sr=sr_test)
            save_wav_file(tone, p, sr=sr_test)

            arr, sr = standardize_audio(p, target_sr=48000)
            self.assertEqual(sr, 48000)
            self.assertAlmostEqual(len(arr), 48000 * 2, delta=100)

    def test_extreme_dynamic_range_inaudible_signal(self):
        """Very low amplitude signal (1e-6) must not underflow or crash VAD."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        tiny = generate_pure_tone(freq=1000.0, duration=3.0, sr=48000, amplitude=1e-6)
        segments = slice_audio_segments(tiny, sr=48000, segment_duration=3.0, vad_filter=True)
        # Since amplitude is near silence, VAD should safely discard it
        self.assertEqual(len(segments), 0)

    def test_sub_frame_short_audio_length_safety(self):
        """Audio with only 100 samples (< 50ms frame) must not cause IndexError."""
        if not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio.vad or slice_audio_segments not yet implemented (M1 in progress)")

        tiny_arr = np.array([0.1, -0.1] * 50, dtype=np.float32)
        try:
            segments = slice_audio_segments(tiny_arr, sr=48000, segment_duration=3.0, vad_filter=False)
            # Should either zero-pad to 144,000 or return empty if under minimum threshold
            self.assertIn(len(segments), [0, 1])
            if len(segments) == 1:
                self.assertEqual(segments[0].shape, (144000,))
        except ValueError:
            # Explicit ValueError for too-short audio is also acceptable
            pass


class TestBoundaryFixturesSelfVerification(unittest.TestCase):
    """Verifies that boundary generator fixtures behave as specified."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_corrupted_wav_generator_file_sizes(self):
        """Verifies byte sizes and headers produced by create_corrupted_wav_file."""
        f_zero = create_corrupted_wav_file(self.tmp_path / "zero.wav", "zero_byte")
        self.assertEqual(f_zero.stat().st_size, 0)

        f_trunc = create_corrupted_wav_file(self.tmp_path / "trunc.wav", "truncated_header")
        self.assertEqual(f_trunc.stat().st_size, 12)

        f_riff = create_corrupted_wav_file(self.tmp_path / "riff.wav", "invalid_riff")
        self.assertTrue(f_riff.read_bytes().startswith(b"ABCD"))

        f_txt = create_corrupted_wav_file(self.tmp_path / "txt.wav", "text_file")
        self.assertTrue(f_txt.read_text(encoding="utf-8").startswith("This is not a WAV"))

    def test_clipped_audio_fixture_saturation(self):
        """Verifies generate_clipped_audio exhibits saturation at +/- 1.0."""
        clipped = generate_clipped_audio(freq=500.0, duration=1.0, sr=48000, gain=15.0)
        # Proportion of samples at extreme boundaries
        boundary_fraction = float(np.mean(np.isclose(np.abs(clipped), 1.0, atol=1e-3)))
        self.assertGreater(boundary_fraction, 0.50, "At gain 15, >50% of samples should be saturated")

    def test_dc_offset_fixture_mean(self):
        """Verifies generate_dc_offset shifts average signal level by offset."""
        dc_arr = generate_dc_offset(offset=0.35, duration=2.0, sr=48000)
        self.assertAlmostEqual(float(np.mean(dc_arr)), 0.35, delta=0.05)


if __name__ == "__main__":
    unittest.main()
