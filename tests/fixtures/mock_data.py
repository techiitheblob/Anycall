"""Mock Data and Metadata Fixtures for AnyCall E2E Testing.

Provides:
- The curated 33 target Indian wildlife species catalog across 4 taxa.
- Mock Xeno-Canto API response generators.
- Deterministic mock embedding generators with controllable cosine clustering.
"""

import hashlib
from typing import Any, Dict, List, Optional
import numpy as np


# ==============================================================================
# Curated 33 Target Indian Wildlife Species Catalog
# ==============================================================================

MOCK_INDIAN_SPECIES_CATALOG: List[Dict[str, Any]] = [
    # Aves (15 species)
    {
        "species_id": "corvus_splendens",
        "scientific_name": "Corvus splendens",
        "common_name": "House Crow",
        "taxon": "Aves",
        "freq_range_hz": (1000, 4500),
        "typical_call": "harsh nasal caw",
    },
    {
        "species_id": "centropus_sinensis",
        "scientific_name": "Centropus sinensis",
        "common_name": "Greater Coucal",
        "taxon": "Aves",
        "freq_range_hz": (300, 900),
        "typical_call": "deep resonant coop-coop-coop",
    },
    {
        "species_id": "pycnonotus_cafer",
        "scientific_name": "Pycnonotus cafer",
        "common_name": "Red-vented Bulbul",
        "taxon": "Aves",
        "freq_range_hz": (1500, 6000),
        "typical_call": "cheerful brief chatter",
    },
    {
        "species_id": "psittacula_krameri",
        "scientific_name": "Psittacula krameri",
        "common_name": "Rose-ringed Parakeet",
        "taxon": "Aves",
        "freq_range_hz": (1800, 7500),
        "typical_call": "shrill squawking screech",
    },
    {
        "species_id": "dicrurus_macrocercus",
        "scientific_name": "Dicrurus macrocercus",
        "common_name": "Black Drongo",
        "taxon": "Aves",
        "freq_range_hz": (1200, 5500),
        "typical_call": "metallic screeching whistle",
    },
    {
        "species_id": "eudynamys_scolopaceus",
        "scientific_name": "Eudynamys scolopaceus",
        "common_name": "Asian Koel",
        "taxon": "Aves",
        "freq_range_hz": (800, 3000),
        "typical_call": "rising koo-ooo whistle",
    },
    {
        "species_id": "orthotomus_sutorius",
        "scientific_name": "Orthotomus sutorius",
        "common_name": "Common Tailorbird",
        "taxon": "Aves",
        "freq_range_hz": (2000, 8000),
        "typical_call": "loud persistent towit-towit",
    },
    {
        "species_id": "copsychus_saularis",
        "scientific_name": "Copsychus saularis",
        "common_name": "Oriental Magpie-Robin",
        "taxon": "Aves",
        "freq_range_hz": (1500, 6500),
        "typical_call": "melodic whistling phrases",
    },
    {
        "species_id": "merops_orientalis",
        "scientific_name": "Merops orientalis",
        "common_name": "Asian Green Bee-eater",
        "taxon": "Aves",
        "freq_range_hz": (2500, 7000),
        "typical_call": "soft rolling trill",
    },
    {
        "species_id": "halcyon_smyrnensis",
        "scientific_name": "Halcyon smyrnensis",
        "common_name": "White-throated Kingfisher",
        "taxon": "Aves",
        "freq_range_hz": (1400, 5000),
        "typical_call": "loud descending cackling laugh",
    },
    {
        "species_id": "alcedo_atthis",
        "scientific_name": "Alcedo atthis",
        "common_name": "Common Kingfisher",
        "taxon": "Aves",
        "freq_range_hz": (3000, 8500),
        "typical_call": "high-pitched whistling peep",
    },
    {
        "species_id": "spilopelia_chinensis",
        "scientific_name": "Spilopelia chinensis",
        "common_name": "Spotted Dove",
        "taxon": "Aves",
        "freq_range_hz": (400, 1200),
        "typical_call": "gentle rhythmic coo-croo-coo",
    },
    {
        "species_id": "columba_livia",
        "scientific_name": "Columba livia",
        "common_name": "Rock Dove",
        "taxon": "Aves",
        "freq_range_hz": (200, 800),
        "typical_call": "throaty guttural cooing",
    },
    {
        "species_id": "argya_striata",
        "scientific_name": "Argya striata",
        "common_name": "Jungle Babbler",
        "taxon": "Aves",
        "freq_range_hz": (1200, 4500),
        "typical_call": "harsh synchronous squeaking chatter",
    },
    {
        "species_id": "acridotheres_tristis",
        "scientific_name": "Acridotheres tristis",
        "common_name": "Common Myna",
        "taxon": "Aves",
        "freq_range_hz": (1000, 5000),
        "typical_call": "varied chirps, croaks, and squawks",
    },

    # Insecta (8 species)
    {
        "species_id": "gryllodes_sigillatus",
        "scientific_name": "Gryllodes sigillatus",
        "common_name": "Indian Field Cricket",
        "taxon": "Insecta",
        "freq_range_hz": (5500, 7500),
        "typical_call": "continuous high-frequency chirping trill",
    },
    {
        "species_id": "acheta_domesticus",
        "scientific_name": "Acheta domesticus",
        "common_name": "House Cricket",
        "taxon": "Insecta",
        "freq_range_hz": (4500, 6000),
        "typical_call": "repetitive sharp chirp pulse",
    },
    {
        "species_id": "schizodactylus_monstrosus",
        "scientific_name": "Schizodactylus monstrosus",
        "common_name": "Dune Cricket",
        "taxon": "Insecta",
        "freq_range_hz": (3000, 6500),
        "typical_call": "harsh low rasping clicks",
    },
    {
        "species_id": "teleogryllus_mitratus",
        "scientific_name": "Teleogryllus mitratus",
        "common_name": "Mitered Cricket",
        "taxon": "Insecta",
        "freq_range_hz": (4000, 7000),
        "typical_call": "rhythmic burst chirping",
    },
    {
        "species_id": "tibicen_plebejus",
        "scientific_name": "Tibicen plebejus",
        "common_name": "Common Cicada",
        "taxon": "Insecta",
        "freq_range_hz": (5000, 12000),
        "typical_call": "intense sustained high-frequency whine",
    },
    {
        "species_id": "platypleura_octoguttata",
        "scientific_name": "Platypleura octoguttata",
        "common_name": "Spotted Cicada",
        "taxon": "Insecta",
        "freq_range_hz": (4000, 10000),
        "typical_call": "pulsing metallic buzzing drone",
    },
    {
        "species_id": "chremistica_mixta",
        "scientific_name": "Chremistica mixta",
        "common_name": "Bark Cicada",
        "taxon": "Insecta",
        "freq_range_hz": (6000, 14000),
        "typical_call": "piercing resonant stridulation",
    },
    {
        "species_id": "gryllus_bimaculatus",
        "scientific_name": "Gryllus bimaculatus",
        "common_name": "Two-spotted Cricket",
        "taxon": "Insecta",
        "freq_range_hz": (4500, 6500),
        "typical_call": "loud rhythmic calling song",
    },

    # Amphibia (5 species)
    {
        "species_id": "hoplobatrachus_tigerinus",
        "scientific_name": "Hoplobatrachus tigerinus",
        "common_name": "Indian Bullfrog",
        "taxon": "Amphibia",
        "freq_range_hz": (300, 1200),
        "typical_call": "deep explosive guttural croak",
    },
    {
        "species_id": "duttaphrynus_melanostictus",
        "scientific_name": "Duttaphrynus melanostictus",
        "common_name": "Asian Common Toad",
        "taxon": "Amphibia",
        "freq_range_hz": (800, 1800),
        "typical_call": "continuous metallic trilling pulse",
    },
    {
        "species_id": "euphlyctis_cyanophlyctis",
        "scientific_name": "Euphlyctis cyanophlyctis",
        "common_name": "Skittering Frog",
        "taxon": "Amphibia",
        "freq_range_hz": (1200, 2600),
        "typical_call": "sharp repeated clicking quacks",
    },
    {
        "species_id": "fejervarya_limnocharis",
        "scientific_name": "Fejervarya limnocharis",
        "common_name": "Paddy Frog",
        "taxon": "Amphibia",
        "freq_range_hz": (1500, 3200),
        "typical_call": "rapid rattling chuckle",
    },
    {
        "species_id": "polypedates_maculatus",
        "scientific_name": "Polypedates maculatus",
        "common_name": "Indian Tree Frog",
        "taxon": "Amphibia",
        "freq_range_hz": (600, 1600),
        "typical_call": "low resonant tonk-tonk pulses",
    },

    # Mammalia (5 species)
    {
        "species_id": "funambulus_palmarum",
        "scientific_name": "Funambulus palmarum",
        "common_name": "Indian Palm Squirrel",
        "taxon": "Mammalia",
        "freq_range_hz": (1800, 6000),
        "typical_call": "repetitive sharp bird-like alarm chatter",
    },
    {
        "species_id": "semnopithecus_entellus",
        "scientific_name": "Semnopithecus entellus",
        "common_name": "Gray Langur",
        "taxon": "Mammalia",
        "freq_range_hz": (250, 1500),
        "typical_call": "deep whooping call and guttural barks",
    },
    {
        "species_id": "canis_aureus",
        "scientific_name": "Canis aureus",
        "common_name": "Golden Jackal",
        "taxon": "Mammalia",
        "freq_range_hz": (400, 2200),
        "typical_call": "drawn-out undulating pack howl",
    },
    {
        "species_id": "macaca_mulatta",
        "scientific_name": "Macaca mulatta",
        "common_name": "Rhesus Macaque",
        "taxon": "Mammalia",
        "freq_range_hz": (300, 3500),
        "typical_call": "cooing contact calls and aggressive screams",
    },
    {
        "species_id": "herpestes_edwardsi",
        "scientific_name": "Herpestes edwardsi",
        "common_name": "Indian Grey Mongoose",
        "taxon": "Mammalia",
        "freq_range_hz": (800, 4000),
        "typical_call": "high-pitched clucking and defensive hisses",
    },
]


def get_mock_catalog() -> List[Dict[str, Any]]:
    """Return a fresh copy of the complete 33 target species catalog."""
    return [dict(s) for s in MOCK_INDIAN_SPECIES_CATALOG]


def get_mock_species_by_taxon(taxon: str) -> List[Dict[str, Any]]:
    """Filter catalog by taxonomic group (case-insensitive)."""
    tax = taxon.lower().strip()
    return [s for s in MOCK_INDIAN_SPECIES_CATALOG if s["taxon"].lower() == tax]


def get_mock_species_by_id(species_id: str) -> Optional[Dict[str, Any]]:
    """Look up species by ID."""
    for s in MOCK_INDIAN_SPECIES_CATALOG:
        if s["species_id"] == species_id:
            return dict(s)
    return None


# ==============================================================================
# Mock Xeno-Canto API Response Generator
# ==============================================================================

def generate_mock_xeno_canto_response(
    species_name: str,
    page: int = 1,
    num_recordings: int = 25,
    country: str = "India",
) -> Dict[str, Any]:
    """Generate a realistic mock Xeno-Canto API JSON response.

    Args:
        species_name: Scientific name (e.g. 'Corvus splendens').
        page: Page number.
        num_recordings: Total recordings returned.
        country: Country filter string.

    Returns:
        Dictionary adhering to Xeno-Canto API v2/v3 JSON schema.
    """
    parts = species_name.split()
    gen = parts[0] if parts else "Corvus"
    sp = parts[1] if len(parts) > 1 else "splendens"

    recordings = []
    base_id = int(hashlib.md5(species_name.encode()).hexdigest()[:6], 16) % 900000 + 100000

    for i in range(num_recordings):
        rec_id = str(base_id + i)
        recordings.append({
            "id": rec_id,
            "gen": gen,
            "sp": sp,
            "ssp": "",
            "en": species_name,
            "rec": f"Field Recordist {i + 1}",
            "cnt": country,
            "loc": f"Protected Reserve {i + 1}",
            "lat": f"19.{1000 + i}",
            "lng": f"72.{8000 + i}",
            "alt": "50",
            "type": "call",
            "url": f"https://xeno-canto.org/{rec_id}",
            "file": f"https://xeno-canto.org/{rec_id}/download",
            "file-name": f"XC{rec_id}-{gen}_{sp}.mp3",
            "license": "//creativecommons.org/licenses/by-nc-sa/4.0/",
            "q": "A" if i % 3 == 0 else "B",
            "length": "0:30",
            "time": "06:30",
            "date": "2026-03-15",
            "uploaded": "2026-03-16",
            "also": [],
            "rmk": "Clean recording",
            "bird-seen": "yes",
            "playback-used": "no",
        })

    return {
        "numRecordings": str(num_recordings),
        "numSpecies": "1",
        "page": page,
        "numPages": 1,
        "recordings": recordings,
    }


# ==============================================================================
# Deterministic Mock Embedding Generators
# ==============================================================================

def generate_mock_embedding(
    species_name: str,
    seed: Optional[int] = None,
    dim: int = 320,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate a deterministic L2-normalized pseudo-embedding for a species.

    Employs an MD5 hash of species_name as a basis seed so that:
    1. Embeddings of the SAME species cluster tightly (cosine similarity > 0.85).
    2. Embeddings of DIFFERENT species are near-orthogonal (cosine similarity < 0.20).
    3. Output vector has exactly unit L2 norm (|v|_2 = 1.0).

    Args:
        species_name: Identifier or scientific name of the species.
        seed: Optional integer seed for exemplar variation.
        dim: Dimensionality of embedding (default: 320 for BirdNET; 1280 for Perch).
        noise_level: Magnitude of perturbation around the class centroid.

    Returns:
        1D float32 numpy array of shape (dim,) with unit norm.
    """
    # Deterministic base vector for this species
    hash_int = int(hashlib.sha256(species_name.encode("utf-8")).hexdigest()[:8], 16)
    base_rng = np.random.default_rng(hash_int)
    base_vector = base_rng.standard_normal(dim).astype(np.float64)
    base_vector /= np.linalg.norm(base_vector)

    if seed is not None:
        exemplar_rng = np.random.default_rng(seed)
        perturb = exemplar_rng.standard_normal(dim).astype(np.float64)
        p_norm = np.linalg.norm(perturb)
        if p_norm > 1e-9:
            perturb = (perturb / p_norm) * float(noise_level)
        vector = base_vector + perturb
    else:
        vector = base_vector

    norm = np.linalg.norm(vector)
    if norm > 1e-9:
        vector = vector / norm
    else:
        vector = base_vector

    return vector.astype(np.float32)


def generate_mock_batch_embeddings(
    species_name: str,
    count: int = 10,
    dim: int = 320,
    base_seed: int = 100,
    noise_level: float = 0.05,
) -> List[np.ndarray]:
    """Generate a batch of mock embeddings for enrollment or query evaluation.

    Args:
        species_name: Species identifier.
        count: Number of exemplar vectors.
        dim: Embedding dimension.
        base_seed: Starting seed.
        noise_level: Intra-class perturbation level.

    Returns:
        List of count unit-normalized float32 numpy arrays.
    """
    return [
        generate_mock_embedding(
            species_name=species_name,
            seed=base_seed + i,
            dim=dim,
            noise_level=noise_level,
        )
        for i in range(count)
    ]
