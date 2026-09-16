import numpy as np
from pathlib import Path
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.data.species import SPECIES_CATALOG

lbl_file = Path(r"C:\Users\kahaa\teamwork_projects\anycall\.venv\Lib\site-packages\birdnet_analyzer\labels\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
with open(lbl_file, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

bn = BirdNetBackbone()
proc = Path("data/processed")

# Only evaluate birds present in BirdNET's vocabulary
bird_species = []
for s, r in SPECIES_CATALOG.items():
    if r.taxon.value == "aves":
        matches = [i for i, l in enumerate(labels) if r.scientific_name.lower() in l.lower()]
        if matches:
            bird_species.append((s, r.common_name, r.scientific_name, matches[0]))

print(f"Evaluating stock BirdNET on {len(bird_species)} bird species present in its vocabulary:\n")

conf_thresholds = [0.0, 0.10, 0.25, 0.50, 0.70]
stats = {ct: {"correct": 0, "wrong": 0, "no_det": 0, "total": 0} for ct in conf_thresholds}

def flat_sigmoid(x, sensitivity=1.0):
    return 1.0 / (1.0 + np.exp(-sensitivity * np.clip(x, -20, 20)))

per_species_res = {}

for sp_id, comm, sci, target_idx in bird_species:
    wavs = list((proc / sp_id).glob("*.wav"))
    if not wavs:
        continue
    sp_correct = 0
    sp_wrong = 0
    
    for w in wavs:
        _, logits = bn.extract_with_logits(w)
        probs = flat_sigmoid(logits)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        is_right = (top_idx == target_idx)
        
        for ct in conf_thresholds:
            stats[ct]["total"] += 1
            if top_prob < ct:
                stats[ct]["no_det"] += 1
            elif is_right:
                stats[ct]["correct"] += 1
            else:
                stats[ct]["wrong"] += 1
                
        if is_right:
            sp_correct += 1
        else:
            sp_wrong += 1
        
    per_species_res[comm] = (sp_correct, len(wavs), sp_correct / len(wavs) * 100)

print("=" * 80)
print(f"{'Common Name':28s} | {'Clips':6s} | {'Correct':8s} | {'Wrong Bird':12s} | {'Accuracy':10s}")
print("-" * 80)
for comm, (c, tot, acc) in sorted(per_species_res.items(), key=lambda x: x[1][2], reverse=True):
    w = tot - c
    print(f"{comm:28s} | {tot:6d} | {c:8d} | {w:12d} | {acc:5.1f}%")
print("=" * 80)

print("\nStock BirdNET Misidentification vs Confidence Threshold (for the 14 birds in its vocab):")
print("-" * 90)
print(f"{'Min Conf Threshold':20s} | {'Correct Bird':14s} | {'Wrong Bird (MisID)':20s} | {'No Detection (Below Conf)':26s}")
print("-" * 90)
for ct in conf_thresholds:
    tot = stats[ct]["total"]
    c = stats[ct]["correct"]
    w = stats[ct]["wrong"]
    nd = stats[ct]["no_det"]
    print(f"Conf >= {ct:4.2f}          | {c/tot*100:5.1f}% ({c:3d})    | {w/tot*100:5.1f}% ({w:3d})      | {nd/tot*100:5.1f}% ({nd:3d})")
print("=" * 90)
