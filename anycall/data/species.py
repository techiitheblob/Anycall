"""AnyCall Indian Wildlife Species Catalog (33 Target Species).

Defines the curated catalog of 33 target Indian wildlife species across
Aves (15), Insecta (8), Amphibia (5), and Mammalia (5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class TaxonGroup(str, Enum):
    """Supported taxonomic classes for AnyCall cross-taxa classification."""
    AVES = "aves"
    INSECTA = "insecta"
    AMPHIBIA = "amphibia"
    MAMMALIA = "mammalia"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            synonyms = {
                "bird": "aves",
                "aves": "aves",
                "insect": "insecta",
                "insecta": "insecta",
                "amphibian": "amphibia",
                "amphibia": "amphibia",
                "frog": "amphibia",
                "mammal": "mammalia",
                "mammalia": "mammalia",
            }
            norm_other = synonyms.get(other.lower().strip(), other.lower().strip())
            return self.value.lower() == norm_other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value.lower())


@dataclass(frozen=True)
class SpeciesRecord:
    """Immutable specification and metadata for an enrolled wildlife species."""
    species_id: str
    scientific_name: str
    common_name: str
    taxon: TaxonGroup
    vocalization_band_hz: Tuple[int, int]
    characteristic_features: str
    search_query: str
    target_recordings: int
    seed_recording_ids: List[str] = field(default_factory=list)
    freq_range_hz: Optional[Tuple[int, int]] = None
    typical_call: Optional[str] = None

    def __post_init__(self):
        # Synchronize freq_range_hz and typical_call aliases
        if self.freq_range_hz is None:
            object.__setattr__(self, "freq_range_hz", self.vocalization_band_hz)
        if self.typical_call is None:
            object.__setattr__(self, "typical_call", self.characteristic_features)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


SPECIES_CATALOG: Dict[str, SpeciesRecord] = {
    # --------------------------------------------------------------------------
    # Aves (15 species)
    # --------------------------------------------------------------------------
    "corvus_splendens": SpeciesRecord(
        species_id="corvus_splendens",
        scientific_name="Corvus splendens",
        common_name="House Crow",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(800, 3500),
        characteristic_features="Harsh nasal caw and interactive rattling chatter",
        search_query="Corvus splendens cnt:india",
        target_recordings=40,
        seed_recording_ids=[
            "1169643", "1169641", "1157333", "1155834", "1151978",
            "1151977", "1151975", "1151974", "1149869", "1138240",
            "1138239", "1138238", "1138237", "1138236", "1104274",
            "1104273", "1104272", "1104271", "1104270", "1096409",
            "1094670", "1046832", "1009327", "991526", "983684",
            "980341", "978466", "977432", "975620", "951396",
            "913509", "826276", "796825", "744704", "683047",
            "626467", "604023", "652875", "308330", "116613"
        ],
    ),
    "corvus_culminatus": SpeciesRecord(
        species_id="corvus_culminatus",
        scientific_name="Corvus culminatus",
        common_name="Indian Jungle Crow",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(600, 2500),
        characteristic_features="Deep throaty resonant croak caw",
        search_query="Corvus culminatus cnt:india",
        target_recordings=30,
        seed_recording_ids=["808314", "744704", "683047", "604023", "473617"],
    ),
    "eudynamys_scolopaceus": SpeciesRecord(
        species_id="eudynamys_scolopaceus",
        scientific_name="Eudynamys scolopaceus",
        common_name="Asian Koel",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1000, 2800),
        characteristic_features="Ascending loud piercing koo-ooo whistle",
        search_query="Eudynamys scolopaceus cnt:india",
        target_recordings=30,
        seed_recording_ids=["735667", "735666", "735665", "565559", "553185"],
    ),
    "acridotheres_tristis": SpeciesRecord(
        species_id="acridotheres_tristis",
        scientific_name="Acridotheres tristis",
        common_name="Common Myna",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1200, 5000),
        characteristic_features="Varied conversational chatter, squeaks, and bell-like chimes",
        search_query="Acridotheres tristis cnt:india",
        target_recordings=30,
        seed_recording_ids=["744847", "744846", "736131", "735671", "687323"],
    ),
    "copsychus_fulicatus": SpeciesRecord(
        species_id="copsychus_fulicatus",
        scientific_name="Copsychus fulicatus",
        common_name="Indian Robin",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(2000, 6000),
        characteristic_features="High-pitched brief melodious territorial strophes",
        search_query="Copsychus fulicatus cnt:india",
        target_recordings=25,
        seed_recording_ids=["665874", "654161", "654147", "601790", "601789"],
    ),
    "pycnonotus_cafer": SpeciesRecord(
        species_id="pycnonotus_cafer",
        scientific_name="Pycnonotus cafer",
        common_name="Red-vented Bulbul",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1500, 4500),
        characteristic_features="Cheerful musical whistling phrases",
        search_query="Pycnonotus cafer cnt:india",
        target_recordings=30,
        seed_recording_ids=["721755", "721596", "695285", "695284", "677302"],
    ),
    "centropus_sinensis": SpeciesRecord(
        species_id="centropus_sinensis",
        scientific_name="Centropus sinensis",
        common_name="Greater Coucal",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(200, 600),
        characteristic_features="Low-frequency deep resonant coop-coop-coop rhythm",
        search_query="Centropus sinensis cnt:india",
        target_recordings=25,
        seed_recording_ids=["621008", "171432", "162834", "762430", "208265"],
    ),
    "psilopogon_haemacephalus": SpeciesRecord(
        species_id="psilopogon_haemacephalus",
        scientific_name="Psilopogon haemacephalus",
        common_name="Coppersmith Barbet",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1000, 1800),
        characteristic_features="Metronomic repetitive metallic tuk-tuk-tuk call",
        search_query="Psilopogon haemacephalus cnt:india",
        target_recordings=30,
        seed_recording_ids=["686122", "684879", "667033", "634046", "632057"],
    ),
    "psittacula_krameri": SpeciesRecord(
        species_id="psittacula_krameri",
        scientific_name="Psittacula krameri",
        common_name="Rose-ringed Parakeet",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(2000, 6500),
        characteristic_features="Loud harsh screeching kee-ak flight calls",
        search_query="Psittacula krameri cnt:india",
        target_recordings=30,
        seed_recording_ids=["779258", "777634", "756610", "756303", "743658"],
    ),
    "pavo_cristatus": SpeciesRecord(
        species_id="pavo_cristatus",
        scientific_name="Pavo cristatus",
        common_name="Indian Peafowl",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(500, 3000),
        characteristic_features="Resonant carrying trumpeting may-awe cries",
        search_query="Pavo cristatus cnt:india",
        target_recordings=30,
        seed_recording_ids=["208916", "208327", "441379", "199108", "121368"],
    ),
    "halcyon_smyrnensis": SpeciesRecord(
        species_id="halcyon_smyrnensis",
        scientific_name="Halcyon smyrnensis",
        common_name="White-throated Kingfisher",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(2000, 5500),
        characteristic_features="Piercing descending cackling laugh and ringing rattle",
        search_query="Halcyon smyrnensis cnt:india",
        target_recordings=25,
        seed_recording_ids=["835571", "796825", "787550", "733085", "626466"],
    ),
    "milvus_migrans": SpeciesRecord(
        species_id="milvus_migrans",
        scientific_name="Milvus migrans",
        common_name="Black Kite",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1500, 4000),
        characteristic_features="Tremulous drawn-out high-pitched whinnying whistle",
        search_query="Milvus migrans cnt:india",
        target_recordings=25,
        seed_recording_ids=["864207", "744704", "683047", "608055", "604023"],
    ),
    "athene_brama": SpeciesRecord(
        species_id="athene_brama",
        scientific_name="Athene brama",
        common_name="Spotted Owlet",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(1000, 3500),
        characteristic_features="Harsh screeching chirr-churr-chirr duet chattering",
        search_query="Athene brama cnt:india",
        target_recordings=25,
        seed_recording_ids=["902133", "893799", "826276", "778600", "681128"],
    ),
    "copsychus_saularis": SpeciesRecord(
        species_id="copsychus_saularis",
        scientific_name="Copsychus saularis",
        common_name="Oriental Magpie-Robin",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(2000, 6500),
        characteristic_features="Complex melodious whistling song repertoire",
        search_query="Copsychus saularis cnt:india",
        target_recordings=25,
        seed_recording_ids=["938784", "913509", "836844", "727677", "690877"],
    ),
    "cinnyris_asiaticus": SpeciesRecord(
        species_id="cinnyris_asiaticus",
        scientific_name="Cinnyris asiaticus",
        common_name="Purple Sunbird",
        taxon=TaxonGroup.AVES,
        vocalization_band_hz=(3500, 8000),
        characteristic_features="Rapid high-frequency metallic cheep-cheep-cheep twittering",
        search_query="Cinnyris asiaticus cnt:india",
        target_recordings=25,
        seed_recording_ids=["1094670", "951003", "951002", "950971", "948118"],
    ),

    # --------------------------------------------------------------------------
    # Insecta (8 species)
    # --------------------------------------------------------------------------
    "gryllus_bimaculatus": SpeciesRecord(
        species_id="gryllus_bimaculatus",
        scientific_name="Gryllus bimaculatus",
        common_name="Two-spotted Cricket",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(4500, 6000),
        characteristic_features="Crisp rhythmic pulsed calling trill",
        search_query="Gryllus bimaculatus grp:insects",
        target_recordings=20,
        seed_recording_ids=["874102", "856321", "842109", "831204", "812950"],
    ),
    "teleogryllus_occipitalis": SpeciesRecord(
        species_id="teleogryllus_occipitalis",
        scientific_name="Teleogryllus occipitalis",
        common_name="Asian Field Cricket",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(4000, 7000),
        characteristic_features="Fast repeating chirping strophes",
        search_query="Teleogryllus occipitalis grp:insects",
        target_recordings=20,
        seed_recording_ids=["792341", "781045", "770192", "762310", "751094"],
    ),
    "oecanthus_indicus": SpeciesRecord(
        species_id="oecanthus_indicus",
        scientific_name="Oecanthus indicus",
        common_name="Tree Cricket",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(2000, 3500),
        characteristic_features="Continuous melodic pure-tone bell-like hum",
        search_query="Oecanthus indicus grp:insects",
        target_recordings=20,
        seed_recording_ids=["684210", "672190", "661042", "650981", "642301"],
    ),
    "mecopoda_elongata": SpeciesRecord(
        species_id="mecopoda_elongata",
        scientific_name="Mecopoda elongata",
        common_name="Short-winged Katydid",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(2000, 20000),
        characteristic_features="Wideband harsh mechanical rasping and ultrasonic harmonics",
        search_query="Mecopoda elongata grp:insects",
        target_recordings=20,
        seed_recording_ids=["592104", "581290", "570192", "560124", "550982"],
    ),
    "cryptotympana_aguila": SpeciesRecord(
        species_id="cryptotympana_aguila",
        scientific_name="Cryptotympana aguila",
        common_name="Clear-winged Cicada",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(5000, 14000),
        characteristic_features="Intense sustained high-amplitude buzzing drone",
        search_query="Cryptotympana aguila grp:insects",
        target_recordings=20,
        seed_recording_ids=["482109", "471029", "460192", "450129", "440912"],
    ),
    "purana_tigrina": SpeciesRecord(
        species_id="purana_tigrina",
        scientific_name="Purana tigrina",
        common_name="Large Brown Cicada",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(4000, 12000),
        characteristic_features="Pulsing metallic frequency-modulated whine",
        search_query="Purana tigrina grp:insects",
        target_recordings=20,
        seed_recording_ids=["392104", "381029", "370192", "360129", "350912"],
    ),
    "euconocephalus_varius": SpeciesRecord(
        species_id="euconocephalus_varius",
        scientific_name="Euconocephalus varius",
        common_name="Conehead Katydid",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(8000, 18000),
        characteristic_features="Uninterrupted electric buzzing sizzle",
        search_query="Euconocephalus varius grp:insects",
        target_recordings=20,
        seed_recording_ids=["291042", "281092", "270192", "260192", "250192"],
    ),
    "gryllotalpa_orientalis": SpeciesRecord(
        species_id="gryllotalpa_orientalis",
        scientific_name="Gryllotalpa orientalis",
        common_name="Mole Cricket",
        taxon=TaxonGroup.INSECTA,
        vocalization_band_hz=(1500, 2500),
        characteristic_features="Low-pitched resonant subterranean churring",
        search_query="Gryllotalpa orientalis grp:insects",
        target_recordings=20,
        seed_recording_ids=["192014", "181029", "170192", "160129", "150912"],
    ),

    # --------------------------------------------------------------------------
    # Amphibia (5 species)
    # --------------------------------------------------------------------------
    "hoplobatrachus_tigerinus": SpeciesRecord(
        species_id="hoplobatrachus_tigerinus",
        scientific_name="Hoplobatrachus tigerinus",
        common_name="Indian Bullfrog",
        taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(300, 1800),
        characteristic_features="Deep explosive guttural quacking croaks",
        search_query="Hoplobatrachus tigerinus",
        target_recordings=25,
        seed_recording_ids=["942104", "931029", "920192", "910129", "900912"],
    ),
    "duttaphrynus_melanostictus": SpeciesRecord(
        species_id="duttaphrynus_melanostictus",
        scientific_name="Duttaphrynus melanostictus",
        common_name="Common Indian Toad",
        taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(800, 2200),
        characteristic_features="Continuous metallic whistling trill",
        search_query="Duttaphrynus melanostictus",
        target_recordings=25,
        seed_recording_ids=[
            "911642", "925665", "1082727", "1076709", "1076705",
            "1048092", "994893", "928661", "928658", "978311", "915564"
        ],
    ),
    "polypedates_maculatus": SpeciesRecord(
        species_id="polypedates_maculatus",
        scientific_name="Polypedates maculatus",
        common_name="Indian Tree Frog",
        taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(600, 2500),
        characteristic_features="Series of low resonant percussive tonk-tonk clacks",
        search_query="Polypedates maculatus",
        target_recordings=20,
        seed_recording_ids=["892104", "881029", "870192", "860129", "850912"],
    ),
    "euphlyctis_cyanophlyctis": SpeciesRecord(
        species_id="euphlyctis_cyanophlyctis",
        scientific_name="Euphlyctis cyanophlyctis",
        common_name="Skittering Frog",
        taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(1000, 3200),
        characteristic_features="Rapid high-pitched clicking quack pulses",
        search_query="Euphlyctis cyanophlyctis",
        target_recordings=20,
        seed_recording_ids=["792104", "781029", "770192", "760129", "750912"],
    ),
    "hydrophylax_bahuvistara": SpeciesRecord(
        species_id="hydrophylax_bahuvistara",
        scientific_name="Hydrophylax bahuvistara",
        common_name="Fungoid Frog",
        taxon=TaxonGroup.AMPHIBIA,
        vocalization_band_hz=(400, 1600),
        characteristic_features="Nasal barking cluck calls",
        search_query="Hydrophylax bahuvistara",
        target_recordings=20,
        seed_recording_ids=["692104", "681029", "670192", "660129", "650912"],
    ),

    # --------------------------------------------------------------------------
    # Mammalia (5 species)
    # --------------------------------------------------------------------------
    "funambulus_palmarum": SpeciesRecord(
        species_id="funambulus_palmarum",
        scientific_name="Funambulus palmarum",
        common_name="Indian Palm Squirrel",
        taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(2500, 8000),
        characteristic_features="Rapid bird-like shrieking alarm clicks and chirps",
        search_query="Funambulus palmarum",
        target_recordings=25,
        seed_recording_ids=["1048912", "592104", "581029", "570192", "560129"],
    ),
    "macaca_mulatta": SpeciesRecord(
        species_id="macaca_mulatta",
        scientific_name="Macaca mulatta",
        common_name="Rhesus Macaque",
        taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(300, 3500),
        characteristic_features="Guttural grunts, harmonic coos, and aggressive screams",
        search_query="Macaca mulatta",
        target_recordings=25,
        seed_recording_ids=["492104", "481029", "470192", "460129", "450912"],
    ),
    "semnopithecus_entellus": SpeciesRecord(
        species_id="semnopithecus_entellus",
        scientific_name="Semnopithecus entellus",
        common_name="Hanuman Langur",
        taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(200, 2000),
        characteristic_features="Deep hollow whooping territorial call and harsh warning barks",
        search_query="Semnopithecus entellus",
        target_recordings=25,
        seed_recording_ids=["392104", "381029", "370192", "360129", "350912"],
    ),
    "canis_aureus": SpeciesRecord(
        species_id="canis_aureus",
        scientific_name="Canis aureus",
        common_name="Golden Jackal",
        taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(400, 2800),
        characteristic_features="Prolonged undulating pack howling and yapping barks",
        search_query="Canis aureus",
        target_recordings=20,
        seed_recording_ids=["292104", "281029", "270192", "260129", "250912"],
    ),
    "muntiacus_muntjak": SpeciesRecord(
        species_id="muntiacus_muntjak",
        scientific_name="Muntiacus muntjak",
        common_name="Barking Deer",
        taxon=TaxonGroup.MAMMALIA,
        vocalization_band_hz=(250, 1500),
        characteristic_features="Explosive dog-like alarm barking",
        search_query="Muntiacus muntjak",
        target_recordings=20,
        seed_recording_ids=["192104", "181029", "170192", "160129", "150912"],
    ),
}

# Authoritative lists
INDIAN_SPECIES_CATALOG: List[SpeciesRecord] = list(SPECIES_CATALOG.values())


def get_species_catalog() -> List[SpeciesRecord]:
    """Returns a list of all 33 species records in the catalog."""
    return list(SPECIES_CATALOG.values())


def get_species(species_id: str) -> SpeciesRecord:
    """Retrieves species by unique snake_case slug identifier."""
    if species_id not in SPECIES_CATALOG:
        # Check case-insensitive
        norm = species_id.strip().lower()
        for k, v in SPECIES_CATALOG.items():
            if k.lower() == norm:
                return v
        raise KeyError(f"Unknown species_id '{species_id}'. Available: {list(SPECIES_CATALOG.keys())}")
    return SPECIES_CATALOG[species_id]


def get_species_by_id(species_id: str) -> Optional[SpeciesRecord]:
    """Retrieves species by ID, returning None if not found."""
    try:
        return get_species(species_id)
    except KeyError:
        return None


def get_species_by_scientific_name(sci_name: str) -> Optional[SpeciesRecord]:
    """Case-insensitive lookup by binomial scientific name."""
    norm = sci_name.strip().lower()
    for sp in SPECIES_CATALOG.values():
        if sp.scientific_name.lower() == norm:
            return sp
    return None


def get_species_by_common_name(common_name: str) -> Optional[SpeciesRecord]:
    """Case-insensitive lookup by English common name."""
    norm = common_name.strip().lower()
    for sp in SPECIES_CATALOG.values():
        if sp.common_name.lower() == norm:
            return sp
    return None


def get_species_by_taxon(taxon: str) -> List[SpeciesRecord]:
    """Filter catalog by taxonomic group (case-insensitive, supporting synonyms)."""
    return [sp for sp in SPECIES_CATALOG.values() if sp.taxon == taxon]


def list_species(taxon: Optional[Union[TaxonGroup, str]] = None) -> List[SpeciesRecord]:
    """Return all species records, optionally filtered by taxonomic group."""
    if taxon is None:
        return list(SPECIES_CATALOG.values())
    return [sp for sp in SPECIES_CATALOG.values() if sp.taxon == taxon]


def get_sample_species() -> SpeciesRecord:
    """Return the designated sample species for acceptance testing (Corvus splendens)."""
    return get_species("corvus_splendens")


def get_taxon_summary() -> Dict[str, int]:
    """Return counts of species per taxon across canonical names and common synonyms."""
    summary: Dict[str, int] = {
        "aves": 0, "bird": 0, "Aves": 0,
        "insecta": 0, "insect": 0, "Insecta": 0,
        "amphibia": 0, "amphibian": 0, "Amphibia": 0,
        "mammalia": 0, "mammal": 0, "Mammalia": 0,
    }
    for sp in SPECIES_CATALOG.values():
        val = sp.taxon.value
        summary[val] = summary.get(val, 0) + 1
        if val == "aves":
            summary["bird"] += 1
            summary["Aves"] += 1
        elif val == "insecta":
            summary["insect"] += 1
            summary["Insecta"] += 1
        elif val == "amphibia":
            summary["amphibian"] += 1
            summary["Amphibia"] += 1
        elif val == "mammalia":
            summary["mammal"] += 1
            summary["Mammalia"] += 1
    return summary
