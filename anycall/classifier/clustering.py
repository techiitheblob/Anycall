"""AnyCall Sub-Prototype Clustering for Multi-Call Repertoires."""
from __future__ import annotations
from typing import List, Sequence, Union
import numpy as np


def cluster_sub_prototypes(
    embeddings: Sequence[np.ndarray],
    variance_threshold: float = 0.05,
    k_clusters: int = 2,
    max_iter: int = 50,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generates sub-prototypes when intra-class cosine variance exceeds threshold.

    Parameters
    ----------
    embeddings : Sequence[np.ndarray]
        List or array of L2-normalized float32 vectors.
    variance_threshold : float
        Minimum cosine variance to trigger sub-prototype generation.
    k_clusters : int
        Number of clusters to generate if variance exceeds threshold.
    max_iter : int
        Max K-Means iterations.
    seed : int
        Random seed for centroid initialization.

    Returns
    -------
    List[np.ndarray]
        List of L2-normalized sub-prototype vectors (or single centroid if variance <= threshold).
    """
    if not embeddings:
        raise ValueError("Cannot cluster empty embeddings list.")

    stack = np.stack(embeddings, axis=0).astype(np.float32)
    # Ensure L2-normalized
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    stack = stack / norms

    # Compute mean centroid
    mean_v = stack.mean(axis=0)
    norm = np.linalg.norm(mean_v)
    mean_v = mean_v / norm if norm > 1e-12 else mean_v

    # Calculate cosine variance
    sims = stack @ mean_v
    cos_var = float(np.var(sims))

    if cos_var < variance_threshold or len(stack) < k_clusters:
        return [mean_v.astype(np.float32)]

    # K-Means clustering
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(stack), size=k_clusters, replace=False)
    centroids = stack[indices].copy()

    for _ in range(max_iter):
        sim_mat = stack @ centroids.T  # (N, k)
        assignments = np.argmax(sim_mat, axis=1)

        new_centroids = np.zeros_like(centroids)
        changed = False
        for ci in range(k_clusters):
            members = stack[assignments == ci]
            if len(members) == 0:
                new_centroids[ci] = centroids[ci]
            else:
                m_vec = members.mean(axis=0)
                m_norm = np.linalg.norm(m_vec)
                nc = m_vec / m_norm if m_norm > 1e-12 else centroids[ci]
                if not np.allclose(nc, centroids[ci], atol=1e-5):
                    changed = True
                new_centroids[ci] = nc

        centroids = new_centroids
        if not changed:
            break

    return [c.astype(np.float32) for c in centroids]
