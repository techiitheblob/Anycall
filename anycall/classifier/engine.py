"""AnyCall Prototypical Classification Engine.

Implements nearest-centroid prototypical classification with:
- Cosine similarity matching
- Tunable open-set rejection threshold θ
- Online / incremental prototype updates
- Optional K-Means sub-prototype splitting for polyphonic species
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _Prototype:
    """Internal representation of a species prototype (centroid)."""

    label: str
    centroid: np.ndarray          # L2-normalized, shape (D,)
    n_support: int = 0            # number of enrollment embeddings used
    sub_prototypes: List[np.ndarray] = field(default_factory=list)


class PrototypicalClassifier:
    """Nearest-centroid prototypical classifier with open-set rejection.

    Usage
    -----
    clf = PrototypicalClassifier(threshold=0.65)
    clf.enroll("Corvus splendens", embeddings_list)
    label, score = clf.predict(query_embedding)

    Parameters
    ----------
    threshold : float
        Cosine-similarity rejection threshold θ ∈ [0, 1].
        Queries whose best score < θ are classified as "Unknown".
    n_subprototypes : int
        If > 1, K-Means is used to split each species enrollment set into
        sub-prototypes (useful for species with diverse call repertoires).
        Default 1 disables sub-prototype splitting.
    unknown_label : str
        Label returned for rejected queries. Default "Unknown".
    """

    UNKNOWN = "Unknown"

    def __init__(
        self,
        threshold: float = 0.65,
        n_subprototypes: int = 1,
        unknown_label: str = "Unknown",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold
        self._n_sub = n_subprototypes
        self._unknown_label = unknown_label
        self._bank: Dict[str, _Prototype] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self._threshold = value

    @property
    def enrolled_species(self) -> List[str]:
        return list(self._bank.keys())

    def enroll(self, label: str, embeddings: List[np.ndarray]) -> None:
        """Compute and store a species prototype from enrollment embeddings.

        Parameters
        ----------
        label : str
            Species label (e.g. "Corvus splendens").
        embeddings : list of np.ndarray
            List of L2-normalized float32 vectors, each of shape (D,).
        """
        if not embeddings:
            raise ValueError(f"Cannot enroll '{label}' with empty embedding list.")

        stack = np.stack([self._l2(e) for e in embeddings], axis=0)  # (N, D)
        centroid = self._l2(stack.mean(axis=0))

        proto = _Prototype(label=label, centroid=centroid, n_support=len(embeddings))

        if self._n_sub > 1 and len(embeddings) >= self._n_sub:
            proto.sub_prototypes = self._kmeans_subprototypes(stack, self._n_sub)

        self._bank[label] = proto

    def update(self, label: str, new_embedding: np.ndarray) -> None:
        """Incrementally update an existing prototype with one new embedding.

        Uses the running-mean formula:
            c_new = L2( (n * c_old + e_new) / (n + 1) )

        If the species does not exist in the bank, it is enrolled fresh.
        """
        e = self._l2(new_embedding)
        if label not in self._bank:
            self.enroll(label, [e])
            return

        proto = self._bank[label]
        n = proto.n_support
        updated = self._l2((n * proto.centroid + e) / (n + 1))
        proto.centroid = updated
        proto.n_support = n + 1

    def predict(
        self, query: np.ndarray
    ) -> Tuple[str, float]:
        """Classify a query embedding using cosine nearest-centroid matching.

        Parameters
        ----------
        query : np.ndarray
            L2-normalized float32 vector of shape (D,).

        Returns
        -------
        (label, score) where label is the predicted species (or "Unknown")
        and score is the best cosine similarity found.
        """
        if not self._bank:
            return self._unknown_label, 0.0

        q = self._l2(query)
        best_label = self._unknown_label
        best_score = -1.0

        for proto in self._bank.values():
            score = self._max_cosine(q, proto)
            if score > best_score:
                best_score = score
                best_label = proto.label

        if best_score < self._threshold:
            return self._unknown_label, float(best_score)
        return best_label, float(best_score)

    def predict_batch(
        self, queries: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Vectorized batch prediction over a (N, D) query matrix."""
        return [self.predict(q) for q in queries]

    def clear(self) -> None:
        """Remove all enrolled prototypes."""
        self._bank.clear()

    def remove(self, label: str) -> None:
        """Remove a single species from the prototype bank."""
        self._bank.pop(label, None)

    def get_prototype(self, label: str) -> Optional[np.ndarray]:
        """Return the centroid vector for a given label, or None."""
        proto = self._bank.get(label)
        return proto.centroid.copy() if proto else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        """Return L2-normalized copy of v. Handles near-zero vectors safely."""
        v = v.flatten().astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            return np.ones_like(v) / float(np.sqrt(len(v)))
        return (v / norm).astype(np.float32)

    def _max_cosine(self, q: np.ndarray, proto: _Prototype) -> float:
        """Return the maximum cosine similarity between q and the prototype.

        When sub-prototypes exist, returns the maximum score across all
        sub-centroids (handles species with diverse repertoires).
        """
        if proto.sub_prototypes:
            scores = [float(np.dot(q, sp)) for sp in proto.sub_prototypes]
            return max(scores)
        return float(np.dot(q, proto.centroid))

    @staticmethod
    def _kmeans_subprototypes(
        stack: np.ndarray, k: int, max_iter: int = 50, seed: int = 42
    ) -> List[np.ndarray]:
        """Lightweight L2-normalized K-Means for sub-prototype computation.

        Parameters
        ----------
        stack : np.ndarray
            Shape (N, D), L2-normalized enrollment embeddings.
        k : int
            Number of sub-prototypes / clusters.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(stack), size=k, replace=False)
        centroids = stack[indices].copy()  # (k, D)

        for _ in range(max_iter):
            # Assignment step: cosine similarity = dot product for L2-normed vecs
            sims = stack @ centroids.T          # (N, k)
            assignments = np.argmax(sims, axis=1)  # (N,)

            new_centroids = np.zeros_like(centroids)
            changed = False
            for ci in range(k):
                members = stack[assignments == ci]
                if len(members) == 0:
                    new_centroids[ci] = centroids[ci]
                else:
                    mean_v = members.mean(axis=0)
                    norm = np.linalg.norm(mean_v)
                    nc = mean_v / norm if norm > 1e-12 else centroids[ci]
                    if not np.allclose(nc, centroids[ci], atol=1e-6):
                        changed = True
                    new_centroids[ci] = nc

            centroids = new_centroids
            if not changed:
                break

        return [centroids[i].astype(np.float32) for i in range(k)]
