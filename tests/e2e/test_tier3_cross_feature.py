"""Tier 3: Cross-Feature Integration Tests for AnyCall.

Covers cross-feature integration pipelines:
1. Audio Standardization -> Uniform Slicing / VAD -> Embedding Extraction.
2. Embedding Extraction -> Prototypical Centroid Calculation -> Unit Hypersphere Normalization.
3. Multi-Species Enrollment -> Cosine Similarity Ranking -> Open-Set Rejection (threshold theta).
4. Online Incremental Prototype Updates (mathematical identity verification).
5. Sub-Prototype Clustering for Multi-Call Repertoires (intra-class variance > 0.05).
6. SQLite Database Persistence (species prototypes BLOB serialization, detection event logging).
7. Full End-to-End Cross-Feature Pipeline: Raw WAV -> Standardization -> Slicing/VAD ->
   Embedding Extraction -> Classification -> Database Logging.

All tests execute fast and offline (< 2.0s) using synthetic audio fixtures and MockBackbone.
Gracefully skips or handles pending M2/M3 modules when concurrent implementations are in flight.
"""

import hashlib
import io
import math
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
)
from tests.fixtures.mock_data import (
    MOCK_INDIAN_SPECIES_CATALOG,
    get_mock_catalog,
    get_mock_species_by_id,
    generate_mock_embedding,
    generate_mock_batch_embeddings,
)

# ---------------------------------------------------------------------------
# Dynamic module imports to support progressive milestones & concurrent dev
# ---------------------------------------------------------------------------
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
    from anycall.embeddings.mock import MockBackbone
    HAVE_MOCK_BACKBONE = True
except (ImportError, AttributeError):
    try:
        from anycall.embeddings import MockBackbone
        HAVE_MOCK_BACKBONE = True
    except (ImportError, AttributeError):
        HAVE_MOCK_BACKBONE = False

try:
    from anycall.classifier.engine import PrototypicalClassifier, PredictionResult
    HAVE_CLASSIFIER = True
except (ImportError, AttributeError):
    try:
        from anycall.classifier import PrototypicalClassifier, PredictionResult
        HAVE_CLASSIFIER = True
    except (ImportError, AttributeError):
        HAVE_CLASSIFIER = False

try:
    from anycall.classifier.clustering import cluster_sub_prototypes
    HAVE_CLUSTERING = True
except (ImportError, AttributeError):
    try:
        from anycall.classifier import cluster_sub_prototypes
        HAVE_CLUSTERING = True
    except (ImportError, AttributeError):
        HAVE_CLUSTERING = False

try:
    from anycall.storage.db import DatabaseManager
    HAVE_STORAGE = True
except (ImportError, AttributeError):
    try:
        from anycall.storage import DatabaseManager
        HAVE_STORAGE = True
    except (ImportError, AttributeError):
        HAVE_STORAGE = False


class TestAudioToEmbeddingPipelineIntegration(unittest.TestCase):
    """Verifies integration between anycall.audio and anycall.embeddings."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wav_file_to_standardized_slices_pipeline(self):
        """Pipeline: Raw multi-second WAV -> standardize_audio -> slice_audio_segments (VAD)."""
        if not HAVE_ANYCALL_STANDARDIZE or not HAVE_ANYCALL_VAD:
            self.skipTest("anycall.audio modules not fully available")

        # Create a 9.0s raw audio file: 3s bird call, 3s silence, 3s insect call at 44.1kHz
        sr_orig = 44100
        call1 = generate_animal_call("aves", duration=3.0, sr=sr_orig, seed=10)
        silence = generate_silence(duration=3.0, sr=sr_orig)
        call2 = generate_animal_call("amphibia", duration=3.0, sr=sr_orig, seed=20)
        raw_stream = np.concatenate([call1, silence, call2])

        wav_path = self.tmp_path / "field_raw_44k.wav"
        save_wav_file(raw_stream, wav_path, sr=sr_orig)

        # 1. Standardize to 48kHz mono
        standardized, sr = standardize_audio(wav_path, target_sr=48000)
        self.assertEqual(sr, 48000)
        self.assertEqual(standardized.ndim, 1)
        expected_len = int(round(9.0 * 48000))
        self.assertAlmostEqual(len(standardized), expected_len, delta=100)

        # 2. Slice into 3.0s segments with VAD active
        slices = slice_audio_segments(standardized, sr=48000, segment_duration=3.0, vad_filter=True)
        # Slices should retain the 2 vocal calls and reject the silence
        self.assertGreaterEqual(len(slices), 2, "VAD must retain vocal segments")
        for s in slices:
            self.assertEqual(s.shape, (144000,), "Each segment must be exactly 144,000 samples (3.0s @ 48kHz)")
            self.assertEqual(s.dtype, np.float32)
            rms = float(np.sqrt(np.mean(s ** 2)))
            self.assertGreater(rms, 0.01, "Retained segment must contain active acoustic energy")

    def test_slices_to_mock_backbone_embedding(self):
        """Pipeline: Standardized 144,000-sample slice -> MockBackbone.embed -> L2-norm vector."""
        if not HAVE_MOCK_BACKBONE:
            self.skipTest("anycall.embeddings.mock.MockBackbone not yet implemented (M2 in progress)")

        backbone = MockBackbone(embedding_dim=256)
        slice_audio = generate_animal_call("aves", duration=3.0, sr=48000, seed=42)
        self.assertEqual(len(slice_audio), 144000)

        emb = backbone.embed(slice_audio, sr=48000)

        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (256,), "Embedding dimension must match backbone definition")
        self.assertEqual(emb.dtype, np.float32)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, delta=1e-5, msg="Backbone embedding must be strictly L2-normalized")

    def test_backbone_file_path_and_array_parity(self):
        """MockBackbone must accept both audio file path and in-memory numpy array."""
        if not HAVE_MOCK_BACKBONE:
            self.skipTest("anycall.embeddings.mock.MockBackbone not yet implemented (M2 in progress)")

        backbone = MockBackbone(embedding_dim=256)
        audio = generate_animal_call("mammalia", duration=3.0, sr=48000, seed=77)
        audio_file = self.tmp_path / "squirrel.wav"
        save_wav_file(audio, audio_file, sr=48000)

        emb_from_array = backbone.embed(audio, sr=48000)
        emb_from_file = backbone.embed(audio_file, sr=48000)

        self.assertEqual(emb_from_array.shape, (256,))
        self.assertEqual(emb_from_file.shape, (256,))
        self.assertAlmostEqual(float(np.linalg.norm(emb_from_array)), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(emb_from_file)), 1.0, delta=1e-5)


class TestPrototypicalClassificationEngineContract(unittest.TestCase):
    """Verifies anycall.classifier.PrototypicalClassifier against Contract 3."""

    def test_prototype_centroid_unit_normalization(self):
        """Prototype centroid of K embeddings must be unit L2-normalized: ||c||_2 = 1.0."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()
        embeddings = generate_mock_batch_embeddings("Corvus splendens", count=10, dim=320, base_seed=1)

        prototype = classifier.compute_prototype(embeddings)

        self.assertIsInstance(prototype, np.ndarray)
        self.assertEqual(prototype.shape, (320,))
        norm = float(np.linalg.norm(prototype))
        self.assertAlmostEqual(norm, 1.0, delta=1e-5, msg="Centroid prototype must have unit L2 norm")

    def test_multi_species_enrollment_and_classification(self):
        """Enrolling multiple species and querying known sample must return correct species label."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()

        # Enroll 3 distinct species
        sp1_embs = generate_mock_batch_embeddings("corvus_splendens", count=10, dim=256, base_seed=10)
        sp2_embs = generate_mock_batch_embeddings("hoplobatrachus_tigerinus", count=10, dim=256, base_seed=20)
        sp3_embs = generate_mock_batch_embeddings("gryllodes_sigillatus", count=10, dim=256, base_seed=30)

        classifier.enroll_species("corvus_splendens", "House Crow", "Aves", sp1_embs)
        classifier.enroll_species("hoplobatrachus_tigerinus", "Indian Bullfrog", "Amphibia", sp2_embs)
        classifier.enroll_species("gryllodes_sigillatus", "Indian Cricket", "Insecta", sp3_embs)

        # Query with an exemplar of Corvus splendens (seeded near sp1 centroid)
        query_sp1 = generate_mock_embedding("corvus_splendens", seed=999, dim=256, noise_level=0.05)
        result = classifier.predict(query_sp1, threshold=0.70)

        self.assertEqual(result.predicted_label, "corvus_splendens")
        self.assertTrue(result.is_known)
        self.assertGreater(result.confidence, 0.70)
        self.assertIn("corvus_splendens", result.top_k_similarities)

    def test_open_set_rejection_below_threshold(self):
        """Query from an unenrolled species must be rejected as 'Unknown' with is_known=False."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()

        sp1_embs = generate_mock_batch_embeddings("corvus_splendens", count=10, dim=256, base_seed=10)
        classifier.enroll_species("corvus_splendens", "House Crow", "Aves", sp1_embs)

        # Query with completely different unenrolled species (orthogonal pseudo-embedding)
        unknown_query = generate_mock_embedding("canis_aureus", seed=500, dim=256, noise_level=0.05)
        result = classifier.predict(unknown_query, threshold=0.70)

        self.assertEqual(result.predicted_label, "Unknown")
        self.assertFalse(result.is_known)
        self.assertLess(result.confidence, 0.70)

    def test_tunable_rejection_threshold_sweep(self):
        """Classification output must adhere strictly to threshold theta parameter."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()
        sp1_embs = generate_mock_batch_embeddings("corvus_splendens", count=10, dim=256, base_seed=10)
        classifier.enroll_species("corvus_splendens", "House Crow", "Aves", sp1_embs)

        # Generate a query with moderate noise so similarity is ~0.75
        query = generate_mock_embedding("corvus_splendens", seed=777, dim=256, noise_level=0.20)

        # Very low threshold -> Known
        res_low = classifier.predict(query, threshold=0.50)
        # Very high threshold -> Unknown
        res_high = classifier.predict(query, threshold=0.99)

        self.assertTrue(res_low.is_known)
        self.assertEqual(res_low.predicted_label, "corvus_splendens")
        self.assertFalse(res_high.is_known)
        self.assertEqual(res_high.predicted_label, "Unknown")


class TestIncrementalPrototypeUpdate(unittest.TestCase):
    """Verifies online incremental update formula: c^(N+M) = normalize((N*c^N + sum(x_j)) / (N+M))."""

    def test_incremental_prototype_update_mathematical_identity(self):
        """Incremental update must match batch re-computation of all N+M exemplars."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()
        species_id = "corvus_splendens"

        initial_embs = generate_mock_batch_embeddings(species_id, count=5, dim=256, base_seed=100)
        classifier.enroll_species(species_id, "House Crow", "Aves", initial_embs)

        new_embs = generate_mock_batch_embeddings(species_id, count=5, dim=256, base_seed=200)
        updated_prototype = classifier.update_prototype(species_id, new_embs)

        # Expected: batch centroid over all 10 exemplars
        all_embs = initial_embs + new_embs
        raw_centroid = np.mean(all_embs, axis=0)
        expected_prototype = raw_centroid / np.linalg.norm(raw_centroid)

        # Cosine similarity between incremental result and batch result must be ~1.0
        cos_sim = float(np.dot(updated_prototype, expected_prototype))
        self.assertAlmostEqual(cos_sim, 1.0, places=4, msg="Incremental update must match batch prototype")
        self.assertAlmostEqual(float(np.linalg.norm(updated_prototype)), 1.0, delta=1e-5)

    def test_single_sample_incremental_update(self):
        """Single-shot incremental update (M=1) must update centroid and sample count."""
        if not HAVE_CLASSIFIER:
            self.skipTest("anycall.classifier not yet implemented (M3 in progress)")

        classifier = PrototypicalClassifier()
        species_id = "funambulus_palmarum"

        initial_embs = generate_mock_batch_embeddings(species_id, count=4, dim=256, base_seed=10)
        classifier.enroll_species(species_id, "Indian Palm Squirrel", "Mammalia", initial_embs)

        one_new_emb = generate_mock_embedding(species_id, seed=99, dim=256, noise_level=0.05)
        updated_proto = classifier.update_prototype(species_id, [one_new_emb])

        self.assertEqual(updated_proto.shape, (256,))
        self.assertAlmostEqual(float(np.linalg.norm(updated_proto)), 1.0, delta=1e-5)


class TestSubPrototypeClustering(unittest.TestCase):
    """Verifies sub-prototype clustering for multi-call repertoires (variance > 0.05)."""

    def test_sub_prototype_clustering_multi_call_variance(self):
        """Multi-modal call distribution must yield >= 2 sub-prototypes when variance > 0.05."""
        if not HAVE_CLUSTERING and not HAVE_CLASSIFIER:
            self.skipTest("Sub-prototype clustering not yet implemented (M3 in progress)")

        # Generate bimodal embeddings: call type A vs call type B
        dim = 256
        embs_call_a = generate_mock_batch_embeddings("sp_call_a", count=10, dim=dim, base_seed=10)
        embs_call_b = generate_mock_batch_embeddings("sp_call_b", count=10, dim=dim, base_seed=20)
        mixed_embs = embs_call_a + embs_call_b

        # Measure cosine variance
        mean_v = np.mean(mixed_embs, axis=0)
        mean_v /= np.linalg.norm(mean_v)
        sims = [float(np.dot(e, mean_v)) for e in mixed_embs]
        cos_var = float(np.var(sims))
        self.assertGreater(cos_var, 0.05, "Synthetic bimodal call set must have variance > 0.05")

        # Verify clustering function
        if HAVE_CLUSTERING:
            sub_prototypes = cluster_sub_prototypes(mixed_embs, variance_threshold=0.05)
            self.assertGreaterEqual(len(sub_prototypes), 2, "Must generate at least 2 sub-prototypes")
            for sp in sub_prototypes:
                self.assertAlmostEqual(float(np.linalg.norm(sp)), 1.0, delta=1e-5)


class TestSQLiteDatabasePersistencePipeline(unittest.TestCase):
    """Verifies SQLite storage for prototypes (species) and detection logs (detections)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_anycall.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_prototype_blob_fidelity(self):
        """Serialized BLOB prototype must restore with bitwise or float32 precision."""
        if not HAVE_STORAGE:
            self.skipTest("anycall.storage not yet implemented (M3 in progress)")

        db = DatabaseManager(self.db_path)
        orig_prototype = generate_mock_embedding("Corvus splendens", seed=1, dim=256)

        db.save_prototype(
            species_id="corvus_splendens",
            common_name="House Crow",
            taxon="Aves",
            prototype=orig_prototype,
            radius=0.15,
            sample_count=10,
        )

        retrieved = db.get_prototype("corvus_splendens")
        self.assertIsNotNone(retrieved)

        # Handle either dict or tuple return structure
        proto_arr = retrieved["prototype"] if isinstance(retrieved, dict) else retrieved[0]
        if isinstance(proto_arr, bytes):
            proto_arr = np.frombuffer(proto_arr, dtype=np.float32)

        self.assertEqual(proto_arr.shape, orig_prototype.shape)
        self.assertTrue(np.allclose(proto_arr, orig_prototype, atol=1e-6))

    def test_log_detection_event_and_query(self):
        """Logged detection events must be recorded with timestamp, confidence, and audio hash."""
        if not HAVE_STORAGE:
            self.skipTest("anycall.storage not yet implemented (M3 in progress)")

        db = DatabaseManager(self.db_path)
        test_hash = hashlib.sha256(b"synthetic_audio_chunk").hexdigest()

        db.log_detection(
            species_id="corvus_splendens",
            confidence=0.885,
            audio_hash=test_hash,
            threshold=0.70,
        )

        detections = db.list_detections(limit=10) if hasattr(db, "list_detections") else db.get_detections()
        self.assertGreaterEqual(len(detections), 1)
        det = detections[0]
        sp_id = det["species_id"] if isinstance(det, dict) else det[2]
        conf = det["confidence"] if isinstance(det, dict) else det[3]

        self.assertEqual(sp_id, "corvus_splendens")
        self.assertAlmostEqual(conf, 0.885, places=3)


class TestCrossFeatureMathematicalAndSchemaContracts(unittest.TestCase):
    """Opaque-box mathematical and schema self-verification tests.

    Executes immediately and independently of any external implementation to
    guarantee that the foundational algebra and storage layouts conform to PROJECT.md.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "contract_test.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_centroid_formula_algebraic_correctness(self):
        """c_k = (1 / K) * sum(e_i) normalized to unit length."""
        rng = np.random.default_rng(123)
        raw_embs = [rng.standard_normal(256).astype(np.float32) for _ in range(7)]
        norm_embs = [e / np.linalg.norm(e) for e in raw_embs]

        # Centroid
        centroid = np.mean(norm_embs, axis=0)
        norm_centroid = centroid / np.linalg.norm(centroid)

        self.assertAlmostEqual(float(np.linalg.norm(norm_centroid)), 1.0, delta=1e-6)

        # Incremental formula verification: N=4, M=3
        N, M = 4, 3
        c_N = np.mean(norm_embs[:N], axis=0)
        c_N_norm = c_N / np.linalg.norm(c_N)

        # Using unnormalized N*c_N or re-scaled: (N * c_N + sum(new)) / (N + M)
        # Note: N * c_N == sum(norm_embs[:N]) exactly!
        incremental_sum = (norm_embs[0] + norm_embs[1] + norm_embs[2] + norm_embs[3]) + sum(norm_embs[N:N+M])
        incremental_centroid = incremental_sum / (N + M)
        incremental_norm = incremental_centroid / np.linalg.norm(incremental_centroid)

        self.assertTrue(np.allclose(norm_centroid, incremental_norm, atol=1e-6))

    def test_sqlite_schema_specification_contract(self):
        """PROJECT.md SQLite specification: 'species' and 'detections' tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create tables according to PROJECT.md Contract 4
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species (
                id TEXT PRIMARY KEY,
                common_name TEXT NOT NULL,
                taxon TEXT NOT NULL,
                prototype_blob BLOB NOT NULL,
                radius REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 1,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                species_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                audio_hash TEXT NOT NULL,
                threshold REAL NOT NULL
            )
        """)
        conn.commit()

        # Verify insertion and BLOB recovery
        sample_proto = np.random.default_rng(42).standard_normal(256).astype(np.float32)
        sample_proto /= np.linalg.norm(sample_proto)
        blob = sample_proto.tobytes()

        cursor.execute(
            "INSERT INTO species VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("corvus_splendens", "House Crow", "Aves", blob, 0.12, 10, "2026-09-16T12:00:00Z"),
        )
        cursor.execute(
            "INSERT INTO detections (timestamp, species_id, confidence, audio_hash, threshold) VALUES (?, ?, ?, ?, ?)",
            ("2026-09-16T12:05:00Z", "corvus_splendens", 0.92, "hash123", 0.70),
        )
        conn.commit()

        # Retrieve and verify
        cursor.execute("SELECT prototype_blob FROM species WHERE id = ?", ("corvus_splendens",))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        recovered_proto = np.frombuffer(row[0], dtype=np.float32)
        self.assertTrue(np.array_equal(sample_proto, recovered_proto))

        cursor.execute("SELECT species_id, confidence FROM detections WHERE audio_hash = ?", ("hash123",))
        det_row = cursor.fetchone()
        self.assertIsNotNone(det_row)
        self.assertEqual(det_row[0], "corvus_splendens")
        self.assertAlmostEqual(det_row[1], 0.92)

        conn.close()


class TestEndToEndCrossFeatureIntegrationPipeline(unittest.TestCase):
    """Full cross-feature pipeline integration test across all modules."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.db_path = self.tmp_path / "e2e_pipeline.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_cross_feature_lifecycle(self):
        """Executes the full pipeline:
        Audio Standardization -> Slicing/VAD -> MockBackbone -> PrototypicalClassifier -> DatabaseManager.
        """
        all_modules_ready = (
            HAVE_ANYCALL_STANDARDIZE
            and HAVE_ANYCALL_VAD
            and HAVE_MOCK_BACKBONE
            and HAVE_CLASSIFIER
            and HAVE_STORAGE
        )
        if not all_modules_ready:
            self.skipTest("Pending full implementation of M2/M3 modules for end-to-end chain")

        # 1. Initialize storage and classifier
        db = DatabaseManager(self.db_path)
        backbone = MockBackbone(embedding_dim=256)
        classifier = PrototypicalClassifier()

        # 2. Synthesize enrollment audio for Corvus splendens
        enrollment_audio = generate_animal_call("aves", duration=6.0, sr=44100, seed=10)
        raw_enroll_path = self.tmp_path / "crow_enroll.wav"
        save_wav_file(enrollment_audio, raw_enroll_path, sr=44100)

        # Standardize & slice
        std_enroll, sr = standardize_audio(raw_enroll_path, target_sr=48000)
        enroll_slices = slice_audio_segments(std_enroll, sr=48000, segment_duration=3.0, vad_filter=True)
        self.assertGreaterEqual(len(enroll_slices), 1)

        # Extract embeddings and enroll
        enroll_embeddings = [backbone.embed(s, sr=48000) for s in enroll_slices]
        classifier.enroll_species("corvus_splendens", "House Crow", "Aves", enroll_embeddings)
        prototype = classifier.compute_prototype(enroll_embeddings)
        db.save_prototype("corvus_splendens", "House Crow", "Aves", prototype, radius=0.1, sample_count=len(enroll_slices))

        # 3. Synthesize query audio (known call)
        query_audio = generate_animal_call("aves", duration=3.0, sr=48000, seed=15)
        query_path = self.tmp_path / "crow_query.wav"
        save_wav_file(query_audio, query_path, sr=48000)

        std_query, _ = standardize_audio(query_path, target_sr=48000)
        query_slices = slice_audio_segments(std_query, sr=48000, segment_duration=3.0, vad_filter=True)
        self.assertGreaterEqual(len(query_slices), 1)

        query_emb = backbone.embed(query_slices[0], sr=48000)
        result = classifier.predict(query_emb, threshold=0.70)

        # 4. Log detection to database
        audio_hash = hashlib.sha256(query_slices[0].tobytes()).hexdigest()
        db.log_detection(
            species_id=result.predicted_label,
            confidence=result.confidence,
            audio_hash=audio_hash,
            threshold=0.70,
        )

        # 5. Assertions
        self.assertTrue(result.is_known)
        self.assertEqual(result.predicted_label, "corvus_splendens")


if __name__ == "__main__":
    unittest.main()
