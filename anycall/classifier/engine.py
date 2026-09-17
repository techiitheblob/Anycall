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
    sum_vector: Optional[np.ndarray] = None


class PredictionResult(tuple):
    """Result of prototypical prediction supporting both tuple unpacking and attribute access.

    Can be unpacked as `label, score = result` or accessed via `result.predicted_label`,
    `result.confidence`, and `result.is_known`.
    """

    def __new__(
        cls,
        predicted_label: str,
        confidence: float,
        is_known: bool = True,
        scores: Optional[Dict[str, float]] = None,
    ):
        return super().__new__(cls, (predicted_label, float(confidence)))

    def __init__(
        self,
        predicted_label: str,
        confidence: float,
        is_known: bool = True,
        scores: Optional[Dict[str, float]] = None,
    ):
        self.predicted_label = predicted_label
        self.confidence = float(confidence)
        self.is_known = is_known
        self.scores = scores or {}
        self.top_k_similarities = self.scores


class PrototypicalClassifier:
    """Nearest-centroid classifier on the unit hypersphere with cosine-distance rejection.

    Features
    --------
    - Enroll species from few-shot embeddings: computes L2-normalized centroid.
    - Classify query embeddings via cosine similarity (dot product on unit vectors).
    - Open-set rejection: returns "Unknown" if max similarity < threshold (θ).
    - Incremental online updates: running mean formula without retraining.
    - Multi-call sub-prototypes: optional K-Means clustering for diverse vocal repertoires.

    Parameters
    ----------
    threshold : float
        Rejection threshold θ ∈ [0.0, 1.0]. Default 0.65.
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

    def compute_prototype(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        """Compute the L2-normalized centroid prototype vector from a set of embeddings."""
        if not embeddings:
            raise ValueError("Cannot compute prototype from empty embeddings.")
        stack = np.stack([self._l2(e) for e in embeddings], axis=0)
        return self._l2(stack.mean(axis=0))

    def enroll_species(
        self,
        species_id: str,
        common_name: str = "",
        taxon: str = "",
        embeddings: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """Enroll a species into the prototype bank."""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError(f"Cannot enroll '{species_id}' with empty embeddings.")
        self.enroll(species_id, list(embeddings))

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
        sum_vec = stack.sum(axis=0)
        centroid = self._l2(sum_vec)

        proto = _Prototype(
            label=label,
            centroid=centroid,
            n_support=len(embeddings),
            sum_vector=sum_vec,
        )

        if self._n_sub > 1 and len(embeddings) >= self._n_sub:
            proto.sub_prototypes = self._kmeans_subprototypes(stack, self._n_sub)

        self._bank[label] = proto

    def update(self, label: str, new_embedding: np.ndarray) -> None:
        """Incrementally update an existing prototype with one new embedding.

        Uses the running-sum formula:
            c_new = L2( sum_old + e_new )

        If the species does not exist in the bank, it is enrolled fresh.
        """
        e = self._l2(new_embedding)
        if label not in self._bank:
            self.enroll(label, [e])
            return

        proto = self._bank[label]
        if proto.sum_vector is None:
            proto.sum_vector = proto.centroid * float(proto.n_support)
        proto.sum_vector = proto.sum_vector + e
        proto.n_support += 1
        proto.centroid = self._l2(proto.sum_vector)

    def update_prototype(
        self, label: str, new_embeddings: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Incrementally update an existing prototype with a batch of new embeddings."""
        for e in new_embeddings:
            self.update(label, e)
        return self._bank[label].centroid

    def predict(
        self, query: np.ndarray, threshold: Optional[float] = None, use_mean_centering: bool = True
    ) -> PredictionResult:
        """Classify a query embedding using cosine nearest-centroid matching.

        Parameters
        ----------
        query : np.ndarray
            L2-normalized float32 vector of shape (D,).
        threshold : float, optional
            Optional threshold override for this prediction.
        use_mean_centering : bool
            If True, applies dynamic mean-centering to mathematically untangle
            cross-taxa biases in frozen embedding spaces.

        Returns
        -------
        PredictionResult where index 0 is label, index 1 is confidence, and attributes
        `predicted_label`, `confidence`, and `is_known` are accessible.
        """
        effective_threshold = self._threshold if threshold is None else threshold
        if not self._bank:
            return PredictionResult(self._unknown_label, 0.0, is_known=False)

        q = self._l2(query)
        
        # Apply dynamic mean-centering if enabled
        global_mean = None
        if use_mean_centering and len(self._bank) > 1:
            all_protos = np.stack([p.centroid for p in self._bank.values()], axis=0)
            global_mean = all_protos.mean(axis=0)
            q = self._l2(q - global_mean)

        best_label = self._unknown_label
        best_score = -1.0
        scores: Dict[str, float] = {}

        for proto in self._bank.values():
            if global_mean is not None:
                centered_proto_centroid = self._l2(proto.centroid - global_mean)
                score = float(np.dot(q, centered_proto_centroid))
            else:
                score = self._max_cosine(q, proto)
            
            scores[proto.label] = float(score)
            if score > best_score:
                best_score = score
                best_label = proto.label

        if best_score < effective_threshold:
            return PredictionResult(
                self._unknown_label, float(best_score), is_known=False, scores=scores
            )
        return PredictionResult(
            best_label, float(best_score), is_known=True, scores=scores
        )

    def predict_batch(
        self, queries: np.ndarray
    ) -> List[PredictionResult]:
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


@dataclass
class NovelCluster:
    """Represents a recurring group of unidentified sounds forming a candidate new species."""
    cluster_id: str
    centroid: np.ndarray             # L2-normalized prototype of the novel sound
    sample_indices: List[int]        # Indices of member samples in the bank
    audio_paths: List[str]           # File paths of member audio clips
    n_samples: int                   # Number of times this unknown sound was heard
    cohesion: float                  # Mean intra-cluster cosine similarity


class UnidentifiedSoundBank:
    """Buffers out-of-bank sounds and groups recurring unidentified calls into candidate novel species.

    Enables field discovery of unknown species:
    1. Audio rejected by PrototypicalClassifier (< θ) is added here.
    2. Recurring calls are clustered using cosine similarity.
    3. When a cluster reaches min_samples (e.g. 5 occurrences), it can be promoted
       directly to an enrolled species in PrototypicalClassifier with zero retraining.
    """

    def __init__(self, cluster_similarity: float = 0.70) -> None:
        self.cluster_similarity = cluster_similarity
        self._embeddings: List[np.ndarray] = []
        self._paths: List[str] = []
        self._timestamps: List[str] = []
        self._clusters: Dict[str, NovelCluster] = {}

    def add(
        self,
        embedding: np.ndarray,
        audio_path: str = "",
        timestamp: str = "",
    ) -> int:
        """Add an unidentified sound embedding to the bank."""
        v = PrototypicalClassifier._l2(embedding)
        idx = len(self._embeddings)
        self._embeddings.append(v)
        self._paths.append(audio_path)
        self._timestamps.append(timestamp)
        return idx

    @property
    def total_unidentified(self) -> int:
        return len(self._embeddings)

    def clear(self) -> None:
        """Clear all buffered unidentified sounds and discovered clusters."""
        self._embeddings.clear()
        self._paths.clear()
        self._timestamps.clear()
        self._clusters.clear()

    def discover_clusters(self, min_cluster_size: int = 3) -> List[NovelCluster]:
        """Leader-clustering over unidentified embeddings to group recurring calls.

        Groups recordings whose mutual cosine similarity exceeds cluster_similarity.
        """
        if len(self._embeddings) < min_cluster_size:
            return []

        raw_stack = np.stack(self._embeddings, axis=0)  # (N, D)
        stack = raw_stack.copy()
        
        # Apply mean centering for robust cross-taxa clustering
        if len(stack) > 1:
            global_mean = stack.mean(axis=0)
            stack = stack - global_mean
            norms = np.linalg.norm(stack, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            stack = stack / norms
            effective_threshold = 0.50 # User requested at least 0.50
        else:
            effective_threshold = self.cluster_similarity

        assigned = np.full(len(stack), -1, dtype=int)
        clusters: List[NovelCluster] = []
        cluster_idx = 0

        for i in range(len(stack)):
            if assigned[i] != -1:
                continue

            # Compute similarity to all unassigned samples
            sims = stack @ stack[i]  # cosine similarity
            members = np.where((sims >= effective_threshold) & (assigned == -1))[0]

            if len(members) >= min_cluster_size:
                assigned[members] = cluster_idx
                # MUST use raw embeddings to calculate centroid for downstream promotion!
                member_embs = raw_stack[members]
                centroid = PrototypicalClassifier._l2(member_embs.mean(axis=0))
                cohesion = float((member_embs @ centroid).mean())

                cid = f"Novel_Cluster_{cluster_idx + 1:02d}"
                c = NovelCluster(
                    cluster_id=cid,
                    centroid=centroid,
                    sample_indices=members.tolist(),
                    audio_paths=[self._paths[m] for m in members if self._paths[m]],
                    n_samples=len(members),
                    cohesion=round(cohesion, 4),
                )
                clusters.append(c)
                self._clusters[cid] = c
                cluster_idx += 1

        return clusters

    def promote_to_species(
        self,
        cluster_id: str,
        species_label: str,
        classifier: PrototypicalClassifier,
    ) -> None:
        """Promote a discovered novel cluster to a named species in the active classifier."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found in bank.")
        cluster = self._clusters[cluster_id]
        member_embs = [self._embeddings[i] for i in cluster.sample_indices]
        classifier.enroll(species_label, member_embs)
