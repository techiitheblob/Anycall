import numpy as np
from pathlib import Path
from anycall.embeddings.birdnet import BirdNetBackbone
from anycall.data.species import SPECIES_CATALOG

lbl_file = Path(r"C:\Users\kahaa\teamwork_projects\anycall\.venv\Lib\site-packages\birdnet_analyzer\labels\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt")
with open(lbl_file, encoding="utf-8") as f:
    labels = [l.strip() for l in f if l.strip()]

bn = BirdNetBackbone()
proc = Path("data/processed")

bird_species = [(s, r.scientific_name, r.common_name) for s, r in SPECIES_CATALOG.items() if r.taxon.value == "aves"]

print("=" * 105)
print("  EVALUATION OF STOCK BIRDNET ON 15 INDIAN RESIDENT BIRD SPECIES")
print("=" * 105)
print(f"{'Common Name':28s} | {'Scientific Name':25s} | {'In BirdNET?':11s} | {'Clips':6s} | {'Stock Correct':14s} | {'Accuracy':9s}")
print("-" * 105)

total_all_clips = 0
total_all_correct = 0
total_in_vocab_clips = 0
total_in_vocab_correct = 0

for sp_id, sci, comm in bird_species:
    wavs = list((proc / sp_id).glob("*.wav"))
    matches = [i for i, l in enumerate(labels) if sci.lower() in l.lower()]
    
    in_vocab = len(matches) > 0
    target_idx = matches[0] if in_vocab else -1
    
    correct = 0
    for w in wavs:
        _, logits = bn.extract_with_logits(w)
        top_idx = int(np.argmax(logits))
        if in_vocab and top_idx == target_idx:
            correct += 1
            
    total_all_clips += len(wavs)
    total_all_correct += correct
    
    if in_vocab:
        total_in_vocab_clips += len(wavs)
        total_in_vocab_correct += correct
        
    acc_str = f"{correct/len(wavs):.1%}" if wavs else "N/A"
    in_str = "YES" if in_vocab else "NO (0%)"
    print(f"{comm:28s} | {sci:25s} | {in_str:11s} | {len(wavs):6d} | {correct:6d} / {len(wavs):<5d} | {acc_str:9s}")

print("-" * 105)
print(f"Overall Stock BirdNET on ALL 15 Indian Birds:          {total_all_correct} / {total_all_clips} ({total_all_correct/total_all_clips:.1%})")
print(f"Stock BirdNET on the 14 Birds PRESENT in BirdNET vocab: {total_in_vocab_correct} / {total_in_vocab_clips} ({total_in_vocab_correct/total_in_vocab_clips:.1%})")
print("=" * 105)
