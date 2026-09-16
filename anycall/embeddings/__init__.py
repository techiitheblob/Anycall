"""AnyCall Unified Acoustic Embedding Extraction Subsystem.

Provides unified abstract base class, candidate bioacoustic backbones
(BirdNET, Google Perch, PANNs), deterministic mock backbone, and a registry/factory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

from anycall.embeddings.base import BaseAudioEmbeddingBackbone
from anycall.embeddings.birdnet import BirdNetBackbone, BirdNETEmbeddingBackbone
from anycall.embeddings.mock import MockBackbone
from anycall.embeddings.panns import PannsBackbone
from anycall.embeddings.perch import PerchBackbone

BACKBONE_REGISTRY: Dict[str, Type[BaseAudioEmbeddingBackbone]] = {
    "birdnet": BirdNetBackbone,
    "perch": PerchBackbone,
    "panns": PannsBackbone,
    "mock": MockBackbone,
}


def get_backbone(name: str, **kwargs: Any) -> BaseAudioEmbeddingBackbone:
    """Factory function to retrieve and instantiate an audio embedding backbone by name.

    Args:
        name: Backbone identifier ('birdnet', 'perch', 'panns', 'mock').
        **kwargs: Optional keyword arguments forwarded to the backbone constructor.

    Returns:
        Instantiated BaseAudioEmbeddingBackbone subclass.

    Raises:
        ValueError: If backbone name is not registered.
    """
    key = name.strip().lower()
    if key not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone '{name}'. Available backbones: {available}")

    backbone_cls = BACKBONE_REGISTRY[key]
    return backbone_cls(**kwargs)


def list_backbones() -> List[str]:
    """Returns list of registered backbone names."""
    return list(BACKBONE_REGISTRY.keys())


__all__ = [
    "BaseAudioEmbeddingBackbone",
    "BirdNetBackbone",
    "BirdNETEmbeddingBackbone",
    "PerchBackbone",
    "PannsBackbone",
    "MockBackbone",
    "BACKBONE_REGISTRY",
    "get_backbone",
    "list_backbones",
]
