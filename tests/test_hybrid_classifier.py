"""Unit test suite for ParallelHybridClassifier."""

import numpy as np
import pytest

from anycall.classifier.hybrid import ParallelHybridClassifier, HybridPredictionResult


def test_hybrid_classifier_initialization():
    hybrid = ParallelHybridClassifier(theta_bird=0.71, theta_non_avian=0.89, offline_fallback=True)
    assert hybrid.theta_bird == 0.71
    assert hybrid.theta_non_avian == 0.89


def test_hybrid_classifier_enroll_and_predict():
    hybrid = ParallelHybridClassifier(theta_bird=0.70, theta_non_avian=0.80, offline_fallback=True)

    # Enroll mock bird (1024-dim BirdNET embeddings)
    bird_emb = np.random.randn(1024).astype(np.float32)
    bird_emb /= np.linalg.norm(bird_emb)
    hybrid.enroll_species("corvus_splendens", "Aves", [bird_emb])

    # Enroll mock frog (2048-dim PANNs embeddings)
    frog_emb = np.random.randn(2048).astype(np.float32)
    frog_emb /= np.linalg.norm(frog_emb)
    hybrid.enroll_species("fejervarya_limnocharis", "Amphibia", [frog_emb])

    # Test dummy 48kHz audio buffer (3 seconds = 144,000 samples)
    synth_audio = np.random.randn(144000).astype(np.float32)
    synth_audio /= np.max(np.abs(synth_audio))

    result = hybrid.predict(synth_audio, sr=48000)
    assert isinstance(result, HybridPredictionResult)
    assert hasattr(result, "predicted_label")
    assert hasattr(result, "confidence")
    assert hasattr(result, "taxon")
    assert hasattr(result, "backbone_used")
