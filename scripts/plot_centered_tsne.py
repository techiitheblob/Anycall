import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

from anycall.embeddings.birdnet import BirdNetBackbone

processed_dir = Path("data/processed")
if not processed_dir.exists():
    print("No data/processed directory found.")
    sys.exit(1)

print("Initializing BirdNET...")
backbone = BirdNetBackbone()

embeddings_all = []
labels_all = []

species_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and "corrupted" not in d.name]
print(f"Extracting embeddings for {len(species_dirs)} clean species...")

for d in species_dirs:
    species_name = d.name.replace("_", " ").title()
    wavs = list(d.glob("*.wav"))
    for w in wavs:
        audio, sr = sf.read(str(w))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        emb = backbone.embed(audio, sr=sr)
        embeddings_all.append(emb)
        labels_all.append(species_name)

embeddings_all = np.stack(embeddings_all)
labels_all = np.array(labels_all)

# MEAN CENTERING: Subtract the global mean!
global_mean = np.mean(embeddings_all, axis=0)
centered_embeddings = embeddings_all - global_mean
# L2 Normalize again after centering
norms = np.linalg.norm(centered_embeddings, axis=1, keepdims=True)
centered_embeddings = np.where(norms > 1e-10, centered_embeddings / norms, centered_embeddings)

print("Computing t-SNE on Centered Embeddings...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords = tsne.fit_transform(centered_embeddings)

# Plotting
plt.figure(figsize=(16, 12))
unique_labels = list(set(labels_all))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for idx, label in enumerate(unique_labels):
    mask = labels_all == label
    subset = coords[mask]
    
    plt.scatter(subset[:, 0], subset[:, 1], color=colors[idx], alpha=0.6, s=30)
    
    # Compute centroid in t-SNE space for labeling
    centroid_tsne = np.mean(subset, axis=0)
    plt.scatter(centroid_tsne[0], centroid_tsne[1], color=colors[idx], 
                edgecolor='black', marker='X', s=250, label=label)
    
    plt.annotate(label, (centroid_tsne[0], centroid_tsne[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=8, 
                 fontweight='bold', alpha=0.9,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[idx], alpha=0.6))

plt.title(f't-SNE of {len(embeddings_all)} Audio Clips (Mean Centered)', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1, title="Species")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

out_path = "centered_tsne_scatter.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved scatter plot to {out_path}")
