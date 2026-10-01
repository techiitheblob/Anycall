"""AnyCall Parallel Hybrid Prototypical Classifier.

Location: anycall/classifier/hybrid.py
Combines BirdNET (for Aves/Birds) and PANNs (for Amphibia, Mammalia, Insecta) backbones
in parallel, running simultaneous feature extraction and cosine prototype scoring
with normalized confidence margin conflict resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from anycall.classifier.engine import PredictionResult, PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.embeddings.panns import PANNsBackbone


@dataclass
class HybridPredictionResult:
    """Result of hybrid dual-backbone prediction containing primary & optional secondary predictions."""

    predicted_label: str
    confidence: float
    is_known: bool
    taxon: str
    backbone_used: str
    primary_prediction: PredictionResult
    secondary_prediction: Optional[PredictionResult] = None
    all_scores: Dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        yield self.predicted_label
        yield self.confidence


class ParallelHybridClassifier:
    """Parallel Dual-Backbone Prototypical Classifier.

    Executes BirdNET (Aves) and PANNs (Non-Avian) concurrently on each incoming 3.0s audio clip:
    1. BirdNET branch extracts 1024-d vector -> scores against Aves prototypes (threshold theta_bird).
    2. PANNs branch extracts 2048-d vector -> scores against Non-Avian prototypes (threshold theta_non_avian).
    3. Resolves dual detections using Normalized Confidence Margin (Delta):
       Delta = (sim - theta) / (1.0 - theta)
    """

    def __init__(
        self,
        theta_bird: float = 0.71,
        theta_non_avian: float = 0.89,
        birdnet_model_path: Optional[str] = None,
        panns_model_path: Optional[str] = None,
        offline_fallback: bool = True,
    ) -> None:
        self.theta_bird = theta_bird
        self.theta_non_avian = theta_non_avian

        # Initialize backbones
        self.birdnet = BirdNetBackbone(model_path=birdnet_model_path, offline_fallback=offline_fallback)
        self.panns = PANNsBackbone(checkpoint_path=panns_model_path, offline_fallback=offline_fallback)


        # Initialize prototype classifier banks
        self.bird_classifier = PrototypicalClassifier(threshold=theta_bird)
        self.non_avian_classifier = PrototypicalClassifier(threshold=theta_non_avian)

        self._species_taxa: Dict[str, str] = {}

    def enroll_species(
        self,
        species_id: str,
        taxon: str,
        embeddings: List[np.ndarray],
    ) -> None:
        """Enroll species into either bird bank or non-avian bank depending on taxon."""
        taxon_lower = taxon.lower()
        self._species_taxa[species_id] = taxon

        if "aves" in taxon_lower or "bird" in taxon_lower:
            self.bird_classifier.enroll(species_id, embeddings)
        else:
            self.non_avian_classifier.enroll(species_id, embeddings)

    def predict(
        self,
        audio: Union[str, np.ndarray],
        sr: int = 48000,
    ) -> HybridPredictionResult:
        """Runs parallel feature extraction and scores query audio across both backbones."""
        # Path 1: BirdNET Branch (Aves)
        emb_bird, _ = self.birdnet.extract_with_logits(audio, sr=sr)
        res_bird = self.bird_classifier.predict(emb_bird, threshold=self.theta_bird)

        # Path 2: PANNs Branch (Non-Avian)
        emb_panns, _ = self.panns.extract_with_logits(audio, sr=sr)
        res_panns = self.non_avian_classifier.predict(emb_panns, threshold=self.theta_non_avian)

        # Combined scores dictionary
        all_scores = {}
        all_scores.update(res_bird.scores)
        all_scores.update(res_panns.scores)

        bird_known = res_bird.is_known
        non_avian_known = res_panns.is_known

        # Case 1: Both models pass threshold -> Conflict Resolution via Normalized Confidence Margin Delta
        if bird_known and non_avian_known:
            delta_bird = (res_bird.confidence - self.theta_bird) / max(1e-6, (1.0 - self.theta_bird))
            delta_panns = (res_panns.confidence - self.theta_non_avian) / max(1e-6, (1.0 - self.theta_non_avian))

            if delta_bird >= delta_panns:
                primary = res_bird
                secondary = res_panns
                primary_taxon = self._species_taxa.get(res_bird.predicted_label, "Aves")
                primary_bb = "birdnet"
            else:
                primary = res_panns
                secondary = res_bird
                primary_taxon = self._species_taxa.get(res_panns.predicted_label, "Non-Avian")
                primary_bb = "panns"

            return HybridPredictionResult(
                predicted_label=primary.predicted_label,
                confidence=primary.confidence,
                is_known=True,
                taxon=primary_taxon,
                backbone_used=primary_bb,
                primary_prediction=primary,
                secondary_prediction=secondary,
                all_scores=all_scores,
            )

        # Case 2: Only BirdNET passes threshold
        elif bird_known:
            taxon = self._species_taxa.get(res_bird.predicted_label, "Aves")
            return HybridPredictionResult(
                predicted_label=res_bird.predicted_label,
                confidence=res_bird.confidence,
                is_known=True,
                taxon=taxon,
                backbone_used="birdnet",
                primary_prediction=res_bird,
                secondary_prediction=None,
                all_scores=all_scores,
            )

        # Case 3: Only PANNs passes threshold
        elif non_avian_known:
            taxon = self._species_taxa.get(res_panns.predicted_label, "Non-Avian")
            return HybridPredictionResult(
                predicted_label=res_panns.predicted_label,
                confidence=res_panns.confidence,
                is_known=True,
                taxon=taxon,
                backbone_used="panns",
                primary_prediction=res_panns,
                secondary_prediction=None,
                all_scores=all_scores,
            )

        # Case 4: Neither model passes threshold -> Reject as Unknown
        else:
            best_conf = max(res_bird.confidence, res_panns.confidence)
            return HybridPredictionResult(
                predicted_label="Unknown",
                confidence=best_conf,
                is_known=False,
                taxon="Unknown",
                backbone_used="none",
                primary_prediction=PredictionResult("Unknown", best_conf, is_known=False),
                secondary_prediction=None,
                all_scores=all_scores,
            )
