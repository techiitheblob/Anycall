import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

db_path = "anycall.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT species_id, common_name, taxon, prototype FROM species")
rows = c.fetchall()
conn.close()

if not rows:
    print("Database is empty!")
    sys.exit(1)

species_ids = []
names = []
taxa = []
embeddings = []

for sp_id, c_name, taxon, blob in rows:
    if blob is None:
        continue
    emb = np.frombuffer(blob, dtype=np.float32)
    if emb.shape[0] != 1024:
        print(f"Skipping {c_name}, bad dimension {emb.shape}")
        continue
    embeddings.append(emb)
    names.append(c_name)
    taxa.append(taxon)
    species_ids.append(sp_id)

embeddings = np.stack(embeddings)

pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.figure(figsize=(14, 12))
unique_taxa = list(set(taxa))
# Use tab10 for colors safely
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_taxa)))

for t_idx, taxon in enumerate(unique_taxa):
    mask = [t == taxon for t in taxa]
    subset_coords = coords[mask]
    plt.scatter(subset_coords[:, 0], subset_coords[:, 1], label=taxon, color=colors[t_idx], s=100)
    
    subset_names = [n for m, n in zip(mask, names) if m]
    for i, name in enumerate(subset_names):
        plt.annotate(name, (subset_coords[i, 0], subset_coords[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

plt.title('PCA of Enrolled Species Prototypes (BirdNET Embeddings)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.legend(title='Taxon', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

out_path = "prototype_scatter.png"
plt.savefig(out_path, dpi=300)
print(f"Scatter plot saved to {out_path}")
