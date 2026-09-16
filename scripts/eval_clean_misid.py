import random
from pathlib import Path
from collections import defaultdict
from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings import get_backbone

backbone = get_backbone("birdnet")
proc = Path("data/processed")

# Load all 33 species on the clean dataset
species_dirs = [d.name for d in proc.iterdir() if d.is_dir() and len(list(d.glob("*.wav"))) >= 5]

embs = {}
for s in sorted(species_dirs):
    wavs = sorted(list((proc / s).glob("*.wav")))
    embs[s] = [backbone.embed(w) for w in wavs]

thresholds = [0.0, 0.60, 0.65, 0.70, 0.75, 0.80]
results = {th: {"correct": 0, "wrong": 0, "unknown": 0, "total": 0} for th in thresholds}
confusions = defaultdict(int)

random.seed(42)
for fold in range(5):
    # Split: enroll 5, test rest
    split_idx = {}
    for s, vec_list in embs.items():
        idx = list(range(len(vec_list)))
        random.shuffle(idx)
        split_idx[s] = idx

    for th in thresholds:
        clf = PrototypicalClassifier(threshold=th)
        for s, vec_list in embs.items():
            idx = split_idx[s]
            clf.enroll(s, [vec_list[i] for i in idx[:5]])

        for s, vec_list in embs.items():
            idx = split_idx[s]
            for i in idx[5:]:
                pred, score = clf.predict(vec_list[i])
                results[th]["total"] += 1
                if pred == s:
                    results[th]["correct"] += 1
                elif pred == "Unknown":
                    results[th]["unknown"] += 1
                else:
                    results[th]["wrong"] += 1
                    if th == 0.0:
                        confusions[(s, pred)] += 1

print("=" * 90)
print("  UPDATED MISIDENTIFICATION RATES ON CLEAN DATASET (33 SPECIES, 5-SHOT BIRDNET)")
print("=" * 90)
print(f"{'Threshold (theta)':17s} | {'Correct':10s} | {'Misidentified (Wrong)':24s} | {'Said Unknown':14s} | {'Precision':10s}")
print("-" * 90)
for th in thresholds:
    c = results[th]["correct"]
    w = results[th]["wrong"]
    u = results[th]["unknown"]
    tot = results[th]["total"]
    pct_c = c / tot * 100
    pct_w = w / tot * 100
    pct_u = u / tot * 100
    prec = (c / (c + w) * 100) if (c + w) > 0 else 0
    print(f"theta = {th:4.2f}          | {pct_c:5.1f}%    | {pct_w:5.1f}% ({w:4d}/{tot})      | {pct_u:5.1f}%        | {prec:5.1f}%")
print("=" * 90)

print("\nTop 8 Remaining Misidentifications (when forced to guess theta=0.0):")
for (true_sp, pred_sp), count in sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:8]:
    pct = count / results[0.0]["total"] * 100
    print(f"  {true_sp:25s} --> {pred_sp:25s} ({count} times, {pct:.2f}% of all queries)")
