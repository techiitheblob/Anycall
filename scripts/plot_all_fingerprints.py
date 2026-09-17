import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import soundfile as sf
import warnings

# Suppress annoying warnings
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
centroids = {}

species_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
print(f"Extracting embeddings for {len(species_dirs)} species. This may take a minute...")

for d in species_dirs:
    species_name = d.name.replace("_", " ").title()
    wavs = list(d.glob("*.wav"))
    
    if not wavs:
        continue
        
    sp_embs = []
    for w in wavs:
        audio, sr = sf.read(str(w))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        emb = backbone.embed(audio, sr=sr)
        sp_embs.append(emb)
        embeddings_all.append(emb)
        labels_all.append(species_name)
    
    if sp_embs:
        centroids[species_name] = np.mean(np.stack(sp_embs), axis=0)
    print(f"Processed {len(sp_embs)} clips for {species_name}")

embeddings_all = np.stack(embeddings_all)
labels_all = np.array(labels_all)

print("Computing PCA...")
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings_all)

# Plotting
plt.figure(figsize=(16, 12))
unique_labels = list(set(labels_all))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for idx, label in enumerate(unique_labels):
    mask = labels_all == label
    subset = coords[mask]
    
    # Plot individual clips
    plt.scatter(subset[:, 0], subset[:, 1], color=colors[idx], alpha=0.4, s=20)
    
    # Compute centroid in PCA space (or just use mean of coords)
    centroid_pca = np.mean(subset, axis=0)
    # Plot centroid
    plt.scatter(centroid_pca[0], centroid_pca[1], color=colors[idx], 
                edgecolor='black', marker='X', s=200, label=label)
    
    # Label the centroid
    plt.annotate(label, (centroid_pca[0], centroid_pca[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=8, 
                 fontweight='bold', alpha=0.9,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[idx], alpha=0.6))

plt.title(f'PCA of {len(embeddings_all)} Audio Clips (All Individual Fingerprints)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1, title="Species Centroids")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

out_path = "all_fingerprints_scatter.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved scatter plot to {out_path}")
