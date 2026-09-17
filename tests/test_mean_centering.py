import unittest
import numpy as np
from anycall.classifier.engine import PrototypicalClassifier
from anycall.classifier.engine import UnidentifiedSoundBank

class TestMeanCenteringClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = PrototypicalClassifier()
        
    def test_prototype_enrollment_raw(self):
        """Prototypes must be saved raw, NOT mean-centered."""
        embs = [
            np.array([1.0, 0.1, 0.0], dtype=np.float32),
            np.array([1.0, 0.2, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.1], dtype=np.float32)
        ]
        # Normalize mock embeddings
        embs = [e / np.linalg.norm(e) for e in embs]
        self.classifier.enroll_species("bird_a", "Bird A", "Aves", embs)
        
        # Centroid should be mean of raw, then L2 normalized
        expected_raw_mean = np.mean(embs, axis=0)
        expected_centroid = expected_raw_mean / np.linalg.norm(expected_raw_mean)
        
        saved_centroid = self.classifier._bank["bird_a"].centroid
        np.testing.assert_array_almost_equal(saved_centroid, expected_centroid, decimal=5)

    def test_predict_applies_mean_centering(self):
        """Prediction must apply mean centering to both query and prototypes."""
        # Create a strong bias in dimension 0
        embs_a = [np.array([1.0, 0.5, 0.0], dtype=np.float32)]
        embs_b = [np.array([1.0, -0.5, 0.0], dtype=np.float32)]
        
        embs_a = [e / np.linalg.norm(e) for e in embs_a]
        embs_b = [e / np.linalg.norm(e) for e in embs_b]
        
        self.classifier.enroll_species("bird_a", "Bird A", "Aves", embs_a)
        self.classifier.enroll_species("bird_b", "Bird B", "Aves", embs_b)
        
        # Test query that is biased but points more to A
        query = np.array([1.0, 0.4, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)
        
        res = self.classifier.predict(query, threshold=0.10)
        self.assertEqual(res.predicted_label, "bird_a")
        self.assertTrue(res.is_known)

    def test_discover_clusters_applies_mean_centering(self):
        """Clustering must center the bank before DBSCAN."""
        bank = UnidentifiedSoundBank(cluster_similarity=0.50)
        
        # High global bias across all vectors
        vec1 = np.array([10.0, 1.0, 0.0], dtype=np.float32); vec1 /= np.linalg.norm(vec1)
        vec2 = np.array([10.0, 1.1, 0.0], dtype=np.float32); vec2 /= np.linalg.norm(vec2)
        vec3 = np.array([10.0, -1.0, 0.0], dtype=np.float32); vec3 /= np.linalg.norm(vec3)
        vec4 = np.array([10.0, -1.1, 0.0], dtype=np.float32); vec4 /= np.linalg.norm(vec4)
        
        bank.add(vec1, "hash1")
        bank.add(vec2, "hash2")
        bank.add(vec3, "hash3")
        bank.add(vec4, "hash4")
        
        clusters = bank.discover_clusters(min_cluster_size=2)
        # Should find 2 distinct clusters despite high cosine similarity in raw space!
        self.assertEqual(len(clusters), 2)
        
        # Check centroids are raw
        for c in clusters:
            # Centroid vector must have positive first dimension since all raw vectors do
            self.assertTrue(c.centroid[0] > 0)

if __name__ == '__main__':
    unittest.main()
