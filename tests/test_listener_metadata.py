import unittest
from unittest.mock import MagicMock
import numpy as np
from anycall.stream.listener import ContinuousMicrophoneListener
from anycall.classifier.engine import PredictionResult

class TestListenerMetadata(unittest.TestCase):
    def test_listener_queries_metadata(self):
        mock_db = MagicMock()
        mock_db.get_prototype.return_value = {
            "common_name": "House Crow",
            "taxon": "Aves"
        }
        
        mock_classifier = MagicMock()
        mock_classifier.threshold = 0.50
        mock_classifier.predict.return_value = PredictionResult(
            predicted_label="corvus_splendens",
            confidence=0.8,
            is_known=True,
            scores={"corvus_splendens": 0.8}
        )
        
        mock_backbone = MagicMock()
        mock_backbone.embed.return_value = np.zeros(256, dtype=np.float32)
        
        listener = ContinuousMicrophoneListener(
            classifier=mock_classifier,
            backbone=mock_backbone,
            db_mgr=mock_db
        )
        
        # Give it a dummy audio clip
        dummy_audio = np.zeros(48000 * 3, dtype=np.float32)
        listener._process_active_vocalization(dummy_audio)
        
        # Verify DB was queried
        mock_db.get_prototype.assert_called_with("corvus_splendens")
        
        # Verify payload contains metadata
        payload = listener._last_detection
        self.assertIsNotNone(payload)
        self.assertEqual(payload["common_name"], "House Crow")
        self.assertEqual(payload["taxon"], "Aves")
        
    def test_listener_handles_missing_metadata(self):
        mock_db = MagicMock()
        mock_db.get_prototype.return_value = None
        
        mock_classifier = MagicMock()
        mock_classifier.threshold = 0.50
        mock_classifier.predict.return_value = PredictionResult(
            predicted_label="corvus_splendens",
            confidence=0.8,
            is_known=True,
            scores={"corvus_splendens": 0.8}
        )
        
        mock_backbone = MagicMock()
        mock_backbone.embed.return_value = np.zeros(256, dtype=np.float32)
        
        listener = ContinuousMicrophoneListener(
            classifier=mock_classifier,
            backbone=mock_backbone,
            db_mgr=mock_db
        )
        
        dummy_audio = np.zeros(48000 * 3, dtype=np.float32)
        listener._process_active_vocalization(dummy_audio)
        
        payload = listener._last_detection
        self.assertIsNotNone(payload)
        self.assertEqual(payload["common_name"], "")
        self.assertEqual(payload["taxon"], "")

if __name__ == '__main__':
    unittest.main()
