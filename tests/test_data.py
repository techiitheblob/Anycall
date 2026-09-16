"""Unit tests for AnyCall Data Subsystem (Milestone 1).

Covers:
- Curated 33 Indian wildlife species catalog completeness and querying.
- Dual-mode Xeno-Canto harvester (API v3 and direct download fallback).
- Local caching, metadata registry manifest, and rate limiting.
"""
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

from anycall.data.harvester import DownloadStatus, XenoCantoHarvester
from anycall.data.species import (
    SPECIES_CATALOG,
    TaxonGroup,
    get_sample_species,
    get_species,
    get_species_by_common_name,
    get_species_by_scientific_name,
    get_species_by_taxon,
    get_species_catalog,
    get_taxon_summary,
)


def test_species_catalog_completeness():
    """Verifies that catalog contains 33 species with correct taxonomic distribution."""
    catalog = get_species_catalog()
    assert len(catalog) == 33

    taxa_counts = {}
    for sp in catalog:
        t = sp.taxon.value.lower()
        taxa_counts[t] = taxa_counts.get(t, 0) + 1

    assert taxa_counts.get("aves") == 15
    assert taxa_counts.get("insecta") == 8
    assert taxa_counts.get("amphibia") == 5
    assert taxa_counts.get("mammalia") == 5


def test_corvus_splendens_seed_ids():
    """Verifies Corvus splendens catalog specifications and seed recordings."""
    sp = get_species_by_scientific_name("Corvus splendens")
    assert sp is not None
    assert sp.common_name == "House Crow"
    assert sp.taxon == "aves"
    assert len(sp.seed_recording_ids) >= 30


def test_species_lookup_helpers():
    """Verifies lookup by common name, scientific name, and taxon."""
    by_common = get_species_by_common_name("House Crow")
    assert by_common is not None
    assert by_common.species_id == "corvus_splendens"

    by_sci = get_species_by_scientific_name("hoplobatrachus tigerinus")
    assert by_sci is not None
    assert by_sci.species_id == "hoplobatrachus_tigerinus"

    frogs = get_species_by_taxon("amphibian")
    assert len(frogs) == 5

    summary = get_taxon_summary()
    assert summary["bird"] == 15
    assert summary["insect"] == 8
    assert summary["amphibian"] == 5
    assert summary["mammal"] == 5


def test_harvester_seed_fallback(tmp_path):
    """Verifies that harvester without API key operates in seed fallback mode."""
    harvester = XenoCantoHarvester(api_key=None, output_dir=tmp_path, rate_limit=0.0)
    assert harvester.mode == "seed_catalog_fallback"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFFfakeaudioWAVEfmt "
    mock_response.headers = {"Content-Type": "audio/mpeg"}

    with patch("requests.get", return_value=mock_response) as mock_get:
        downloaded = harvester.harvest_species("Corvus splendens", limit=5)
        assert len(downloaded) == 5
        assert mock_get.call_count == 5

        first_call_url = mock_get.call_args_list[0][0][0]
        assert "xeno-canto.org" in first_call_url
        assert "/download" in first_call_url


def test_harvester_metadata_manifest(tmp_path):
    """Verifies that metadata.json manifest is properly generated."""
    harvester = XenoCantoHarvester(api_key=None, output_dir=tmp_path, rate_limit=0.0)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake-audio-bytes"

    with patch("requests.get", return_value=mock_response):
        harvester.harvest_species("Corvus splendens", limit=2)
        manifest_path = tmp_path / "corvus_splendens" / "metadata.json"
        assert manifest_path.exists()
        assert manifest_path.stat().st_size > 0


def test_harvester_caching_skips_download(tmp_path):
    """Verifies that existing non-empty file is recognized as cached."""
    sp_dir = tmp_path / "corvus_splendens"
    sp_dir.mkdir(parents=True, exist_ok=True)
    existing_file = sp_dir / "1169643.mp3"
    existing_file.write_bytes(b"dummy-mp3-content-existing-on-disk" * 10)

    harvester = XenoCantoHarvester(api_key=None, output_dir=tmp_path, rate_limit=0.0)
    with patch("requests.get") as mock_get:
        summary = harvester.harvest_species("corvus_splendens", limit=1)
        assert summary.cached_count == 1
        assert mock_get.call_count == 0
