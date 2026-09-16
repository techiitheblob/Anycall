import random
from pathlib import Path
from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings import get_backbone
from anycall.data.species import SPECIES_CATALOG

backbone = get_backbone("birdnet")
proc = Path("data/processed")
bird_species = [s for s, r in SPECIES_CATALOG.items() if r.taxon.value == "aves"]

embs = {}
for s in bird_species:
    wavs = sorted(list((proc / s).glob("*.wav")))
    if len(wavs) >= 10:
        embs[s] = [backbone.embed(w) for w in wavs]

thresholds = [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
results = {th: {"correct": 0, "wrong_bird": 0, "unknown": 0, "total": 0} for th in thresholds}

random.seed(42)
for fold in range(5):
    # Split indices
    split_indices = {}
    for s, vec_list in embs.items():
        idx = list(range(len(vec_list)))
        random.shuffle(idx)
        split_indices[s] = idx

    for th in thresholds:
        clf = PrototypicalClassifier(threshold=th)
        for s, vec_list in embs.items():
            idx = split_indices[s]
            clf.enroll(s, [vec_list[i] for i in idx[:5]])
        
        for s, vec_list in embs.items():
            idx = split_indices[s]
            for i in idx[5:]:
                pred, score = clf.predict(vec_list[i])
                results[th]["total"] += 1
                if pred == s:
                    results[th]["correct"] += 1
                elif pred == "Unknown":
                    results[th]["unknown"] += 1
                else:
                    results[th]["wrong_bird"] += 1

print("=" * 85)
print("  REJECTION THRESHOLD ANALYSIS ON INDIAN BIRDS (5-Shot BirdNET)")
print("=" * 85)
print(f"{'Threshold (θ)':15s} | {'Correct Bird':14s} | {'Wrong Bird (False ID)':22s} | {'Said Unknown':14s} | {'Precision':10s}")
print("-" * 85)
for th in thresholds:
    c = results[th]["correct"]
    w = results[th]["wrong_bird"]
    u = results[th]["unknown"]
    tot = results[th]["total"]
    pct_c = c / tot * 100
    pct_w = w / tot * 100
    pct_u = u / tot * 100
    precision = (c / (c + w) * 100) if (c + w) > 0 else 0
    print(f"θ = {th:4.2f}          | {pct_c:5.1f}%        | {pct_w:5.1f}%                | {pct_u:5.1f}%        | {precision:5.1f}%")
print("=" * 85)
