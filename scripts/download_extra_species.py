#!/usr/bin/env python3
"""
Download extra Indian resident species for Insecta, Amphibia, and Mammalia
using the existing XenoCantoHarvester (API v3 + fallback).

Constructs SpeciesRecord objects for each extra species so the harvester
works without modifying the original species.py catalog.
Writes data/supplementary_taxa.json for the benchmark taxon lookup.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from anycall.data.harvester import XenoCantoHarvester
from anycall.data.species import SpeciesRecord, TaxonGroup
from anycall.audio.standardize import standardize_audio, slice_audio_segments

PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")
SUPP_JSON     = Path("data/supplementary_taxa.json")
TARGET        = 15
DL_LIMIT      = 40

harvester = XenoCantoHarvester(
    raw_dir=RAW_DIR,
    rate_limit=1.5,
    force_fallback=False,   # use API v3 Mode A (no key = falls back to search)
)

# ── Extra species definitions ─────────────────────────────────────────────
# All confirmed Indian resident species. Recordings from anywhere globally.
EXTRA_SPECIES: list[SpeciesRecord] = [
    # ── Insecta (+6 → ~14 total) ──
    SpeciesRecord(
        species_id="hieroglyphus_banian", scientific_name="Hieroglyphus banian",
        common_name="Paddy Grasshopper", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(2000, 8000), characteristic_features="stridulation",
        search_query="Hieroglyphus banian", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="homorocoryphus_nitidulus", scientific_name="Homorocoryphus nitidulus",
        common_name="Tropical Katydid", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(5000, 15000), characteristic_features="stridulation",
        search_query="Homorocoryphus nitidulus", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="conocephalus_maculatus", scientific_name="Conocephalus maculatus",
        common_name="Spotted Bush Cricket", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(8000, 20000), characteristic_features="stridulation",
        search_query="Conocephalus maculatus", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="acheta_domesticus", scientific_name="Acheta domesticus",
        common_name="House Cricket", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(3000, 6000), characteristic_features="chirping",
        search_query="Acheta domesticus", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="valanga_irregularis", scientific_name="Valanga irregularis",
        common_name="Giant Grasshopper", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(2000, 10000), characteristic_features="stridulation",
        search_query="Valanga irregularis", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="schistocerca_gregaria", scientific_name="Schistocerca gregaria",
        common_name="Desert Locust", taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(2000, 12000), characteristic_features="stridulation",
        search_query="Schistocerca gregaria", target_recordings=DL_LIMIT,
    ),

    # ── Amphibia (+6 → ~11 total) ──
    SpeciesRecord(
        species_id="microhyla_ornata", scientific_name="Microhyla ornata",
        common_name="Ornate Narrow-mouthed Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(1000, 5000), characteristic_features="advertisement call",
        search_query="Microhyla ornata", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="fejervarya_limnocharis", scientific_name="Fejervarya limnocharis",
        common_name="Asian Grass Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(500, 3000), characteristic_features="advertisement call",
        search_query="Fejervarya limnocharis", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="kaloula_taprobanica", scientific_name="Kaloula taprobanica",
        common_name="Sri Lanka Bull Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(500, 4000), characteristic_features="loud mooing call",
        search_query="Kaloula taprobanica", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="sphaerotheca_breviceps", scientific_name="Sphaerotheca breviceps",
        common_name="Indian Burrowing Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(500, 3500), characteristic_features="advertisement call",
        search_query="Sphaerotheca breviceps", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="uperodon_systoma", scientific_name="Uperodon systoma",
        common_name="Marbled Balloon Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(1000, 5000), characteristic_features="balloon-like call",
        search_query="Uperodon systoma", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="nyctibatrachus_major", scientific_name="Nyctibatrachus major",
        common_name="Large Torrent Frog", taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(1000, 6000), characteristic_features="torrent call",
        search_query="Nyctibatrachus major", target_recordings=DL_LIMIT,
    ),

    # ── Mammalia (+6 → ~11 total) ──
    SpeciesRecord(
        species_id="elephas_maximus", scientific_name="Elephas maximus",
        common_name="Asian Elephant", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(14, 8000), characteristic_features="trumpeting and rumbles",
        search_query="Elephas maximus", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="panthera_tigris", scientific_name="Panthera tigris",
        common_name="Bengal Tiger", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(100, 4000), characteristic_features="roar and prusten",
        search_query="Panthera tigris", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="axis_axis", scientific_name="Axis axis",
        common_name="Spotted Deer", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(500, 4000), characteristic_features="alarm bark",
        search_query="Axis axis", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="vulpes_bengalensis", scientific_name="Vulpes bengalensis",
        common_name="Indian Fox", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(500, 6000), characteristic_features="bark and whine",
        search_query="Vulpes bengalensis", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="sus_scrofa", scientific_name="Sus scrofa",
        common_name="Wild Boar", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(100, 4000), characteristic_features="grunt and squeal",
        search_query="Sus scrofa", target_recordings=DL_LIMIT,
    ),
    SpeciesRecord(
        species_id="melursus_ursinus", scientific_name="Melursus ursinus",
        common_name="Sloth Bear", taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(100, 5000), characteristic_features="huffing and woofing",
        search_query="Melursus ursinus", target_recordings=DL_LIMIT,
    ),
]


def process_raws(raw_dir: Path, processed_dir: Path, use_vad: bool = True) -> int:
    processed_dir.mkdir(parents=True, exist_ok=True)
    existing = set(p.name for p in processed_dir.glob("*.wav"))
    n_new = 0
    for raw_path in sorted(raw_dir.glob("*.mp3")):
        if len(list(processed_dir.glob("*.wav"))) >= TARGET:
            break
        try:
            waveform, sr = standardize_audio(raw_path)
            segments = slice_audio_segments(waveform, sr=sr, vad_filter=use_vad)
            for i, seg in enumerate(segments):
                out_name = f"{raw_path.stem}_seg{i:03d}.wav"
                if out_name not in existing:
                    sf.write(str(processed_dir / out_name), seg, sr, subtype="PCM_16")
                    existing.add(out_name)
                    n_new += 1
        except Exception:
            continue
    return n_new


# ── Load existing supplementary taxa ──────────────────────────────────────
supp_taxa: dict = {}
if SUPP_JSON.exists():
    with open(SUPP_JSON) as f:
        supp_taxa = json.load(f)

print(f"[Extra] Downloading {len(EXTRA_SPECIES)} extra Indian resident species.\n")

for sp in EXTRA_SPECIES:
    proc_dir = PROCESSED_DIR / sp.species_id
    existing_n = len(list(proc_dir.glob("*.wav"))) if proc_dir.exists() else 0

    # Register taxon regardless
    supp_taxa[sp.species_id] = sp.taxon.value

    if existing_n >= TARGET:
        print(f"  [SKIP] {sp.common_name} — already has {existing_n} segments")
        continue

    print(f"  [>>>] {sp.common_name} ({sp.scientific_name})  [{sp.taxon.value}]")

    # Harvest using the existing working harvester (no country filter)
    try:
        summary = harvester.harvest_species(sp, limit=DL_LIMIT, country=None, force=True)
        raw_files = summary.valid_file_paths
        print(f"         Got {len(raw_files)} raw recordings")
    except Exception as exc:
        print(f"         [ERROR] {exc}")
        continue

    if not raw_files:
        print(f"         [SKIP] No recordings available")
        continue

    # Process: VAD on first
    raw_dir = RAW_DIR / sp.species_id
    n_new = process_raws(raw_dir, proc_dir, use_vad=True)
    total = len(list(proc_dir.glob("*.wav")))
    print(f"         +{n_new} (VAD on)  -> {total} total")

    # Fallback: VAD off
    if total < TARGET:
        n_new2 = process_raws(raw_dir, proc_dir, use_vad=False)
        total = len(list(proc_dir.glob("*.wav")))
        print(f"         +{n_new2} (VAD off) -> {total} total")

    print(f"         [{'OK' if total >= TARGET else 'STILL LOW'}]")

# Save supplementary taxa JSON
SUPP_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(SUPP_JSON, "w") as f:
    json.dump(supp_taxa, f, indent=2)

from collections import Counter
print(f"\n[Extra] Supplementary taxa saved: {dict(Counter(supp_taxa.values()))}")
print(f"[Extra] Done.")
