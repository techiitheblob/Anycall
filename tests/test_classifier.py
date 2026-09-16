"""Unit tests for anycall.classifier.engine and anycall.storage.db."""
from __future__ import annotations

import numpy as np
import pytest

from anycall.classifier.engine import PrototypicalClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rand_unit(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_cluster(centroid: np.ndarray, n: int = 8, noise: float = 0.05, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n):
        noisy = centroid + (rng.standard_normal(centroid.shape) * noise).astype(np.float32)
        norm = np.linalg.norm(noisy)
        vecs.append((noisy / norm).astype(np.float32))
    return vecs


DIM = 128
C1 = _rand_unit(DIM, seed=10)
C2 = _rand_unit(DIM, seed=20)
C3 = _rand_unit(DIM, seed=30)


# ---------------------------------------------------------------------------
# PrototypicalClassifier — basic tests
# ---------------------------------------------------------------------------

class TestPrototypicalClassifier:

    def test_enroll_and_predict_correct_species(self):
        clf = PrototypicalClassifier(threshold=0.5)
        clf.enroll("species_A", _make_cluster(C1, seed=1))
        clf.enroll("species_B", _make_cluster(C2, seed=2))
        # Query very close to C1
        query = _make_cluster(C1, n=1, noise=0.01, seed=99)[0]
        label, score = clf.predict(query)
        assert label == "species_A", f"Expected species_A, got {label}"
        assert score > 0.5

    def test_predict_unknown_below_threshold(self):
        clf = PrototypicalClassifier(threshold=0.999)  # near-impossible threshold
        clf.enroll("species_A", _make_cluster(C1, seed=1))
        query = _make_cluster(C2, n=1, noise=0.01, seed=99)[0]  # far from C1
        label, score = clf.predict(query)
        assert label == "Unknown"

    def test_empty_bank_returns_unknown(self):
        clf = PrototypicalClassifier(threshold=0.5)
        label, score = clf.predict(C1)
        assert label == "Unknown"
        assert score == 0.0

    def test_enroll_empty_raises(self):
        clf = PrototypicalClassifier(threshold=0.5)
        with pytest.raises(ValueError):
            clf.enroll("species_A", [])

    def test_incremental_update(self):
        clf = PrototypicalClassifier(threshold=0.0)
        clf.enroll("species_A", _make_cluster(C1, n=5, seed=1))
        proto_before = clf.get_prototype("species_A").copy()

        clf.update("species_A", _make_cluster(C1, n=1, noise=0.01, seed=77)[0])
        proto_after = clf.get_prototype("species_A")

        # Prototype should shift slightly but remain unit-normalized
        assert proto_after is not None
        assert abs(np.linalg.norm(proto_after) - 1.0) < 1e-5
        # After update n_support should be 6
        assert clf._bank["species_A"].n_support == 6

    def test_update_new_species_enrolls(self):
        clf = PrototypicalClassifier(threshold=0.0)
        clf.update("new_species", C1)
        assert "new_species" in clf.enrolled_species

    def test_l2_normalization_of_prototype(self):
        clf = PrototypicalClassifier(threshold=0.0)
        clf.enroll("sp", _make_cluster(C1, n=10, noise=0.1, seed=5))
        proto = clf.get_prototype("sp")
        assert abs(np.linalg.norm(proto) - 1.0) < 1e-5

    def test_threshold_setter_validation(self):
        clf = PrototypicalClassifier(threshold=0.5)
        clf.threshold = 0.8
        assert clf.threshold == 0.8
        with pytest.raises(ValueError):
            clf.threshold = 1.5

    def test_remove_species(self):
        clf = PrototypicalClassifier(threshold=0.0)
        clf.enroll("sp_A", _make_cluster(C1))
        clf.enroll("sp_B", _make_cluster(C2))
        clf.remove("sp_A")
        assert "sp_A" not in clf.enrolled_species
        assert "sp_B" in clf.enrolled_species

    def test_clear(self):
        clf = PrototypicalClassifier(threshold=0.0)
        clf.enroll("sp_A", _make_cluster(C1))
        clf.clear()
        assert clf.enrolled_species == []

    def test_batch_predict(self):
        clf = PrototypicalClassifier(threshold=0.5)
        clf.enroll("sp_A", _make_cluster(C1, seed=1))
        clf.enroll("sp_B", _make_cluster(C2, seed=2))
        queries = np.stack(_make_cluster(C1, n=3, noise=0.02, seed=5))
        results = clf.predict_batch(queries)
        assert len(results) == 3
        for label, score in results:
            assert label == "sp_A"

    def test_subprototypes(self):
        clf = PrototypicalClassifier(threshold=0.0, n_subprototypes=2)
        # Two clusters far apart for the same species (polyphonic)
        mixed = _make_cluster(C1, n=6, seed=1) + _make_cluster(C2, n=6, seed=2)
        clf.enroll("poly_sp", mixed)
        assert len(clf._bank["poly_sp"].sub_prototypes) == 2

    def test_predict_three_species(self):
        clf = PrototypicalClassifier(threshold=0.3)
        clf.enroll("sp_A", _make_cluster(C1, seed=1))
        clf.enroll("sp_B", _make_cluster(C2, seed=2))
        clf.enroll("sp_C", _make_cluster(C3, seed=3))

        for centroid, expected in [(C1, "sp_A"), (C2, "sp_B"), (C3, "sp_C")]:
            q = _make_cluster(centroid, n=1, noise=0.02, seed=42)[0]
            label, _ = clf.predict(q)
            assert label == expected, f"Expected {expected}, got {label}"


# ---------------------------------------------------------------------------
# PrototypeStore — basic tests
# ---------------------------------------------------------------------------

class TestPrototypeStore:

    def test_save_and_load(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            centroid = C1.copy()
            store.save_prototype("sp_A", centroid, n_support=5, backbone="mock")
            result = store.load_prototype("sp_A")
            assert result is not None
            loaded_centroid, n_support = result
            assert n_support == 5
            assert np.allclose(loaded_centroid, centroid, atol=1e-5)

    def test_list_labels(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            store.save_prototype("sp_A", C1)
            store.save_prototype("sp_B", C2)
            labels = store.list_labels()
            assert set(labels) == {"sp_A", "sp_B"}

    def test_delete_prototype(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            store.save_prototype("sp_A", C1)
            store.delete_prototype("sp_A")
            assert store.load_prototype("sp_A") is None

    def test_load_nonexistent_returns_none(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            assert store.load_prototype("ghost") is None

    def test_load_all_prototypes(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            store.save_prototype("sp_A", C1, n_support=3)
            store.save_prototype("sp_B", C2, n_support=7)
            all_protos = store.load_all_prototypes()
            assert len(all_protos) == 2
            assert "sp_A" in all_protos
            assert all_protos["sp_A"][1] == 3

    def test_log_and_query_detection(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            store.log_detection("sp_A", 0.87, backbone="birdnet", audio_path="/tmp/a.wav")
            store.log_detection("sp_B", 0.72, backbone="perch")
            detections = store.query_detections()
            assert len(detections) == 2
            filtered = store.query_detections(label="sp_A")
            assert len(filtered) == 1
            assert filtered[0]["score"] == pytest.approx(0.87)

    def test_upsert_prototype(self):
        from anycall.storage.db import PrototypeStore
        with PrototypeStore(":memory:") as store:
            store.save_prototype("sp_A", C1, n_support=5)
            store.save_prototype("sp_A", C2, n_support=10)  # overwrite
            result = store.load_prototype("sp_A")
            assert result[1] == 10
            assert np.allclose(result[0], C2, atol=1e-5)
