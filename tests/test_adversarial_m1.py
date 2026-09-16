"""Adversarial Verification Suite for Milestone 1.

Authored by challenger_m1_2.
Stress-tests:
1. All 74 processed audio files in data/processed/corvus_splendens/.
2. Harvester failure resistance (404, 500 backoff, network timeouts, zero-byte payloads).
3. Harvester caching logic and cache invalidation under corruption or force flag.
4. Harvester invalid species handling.
5. Rate limiting enforcement.
6. Species catalog integrity and seed ID duplication audit.
"""
import hashlib
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import requests
import soundfile as sf

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
)


# ==============================================================================
# 1. Processed Audio File Integrity Audit (All 74 files)
# ==============================================================================

PROCESSED_DIR = Path("data/processed/corvus_splendens")


def test_processed_dataset_quantity_and_presence():
    """Verify that data/processed/corvus_splendens exists and contains at least 20 files (expected 74)."""
    assert PROCESSED_DIR.exists(), f"Directory not found: {PROCESSED_DIR}"
    assert PROCESSED_DIR.is_dir()
    wav_files = list(PROCESSED_DIR.glob("*.wav"))
    assert len(wav_files) >= 20, f"Expected >= 20 files, found {len(wav_files)}"
    assert len(wav_files) == 74, f"Expected exactly 74 files from M1 run, found {len(wav_files)}"


def test_all_74_audio_files_strict_spec_compliance():
    """Verify every single WAV file conforms strictly to M1 contract specifications."""
    wav_files = sorted(list(PROCESSED_DIR.glob("*.wav")))
    assert len(wav_files) == 74

    for fpath in wav_files:
        info = sf.info(str(fpath))
        assert info.format == "WAV", f"{fpath.name}: format {info.format} != WAV"
        assert info.subtype == "PCM_16", f"{fpath.name}: subtype {info.subtype} != PCM_16"
        assert info.samplerate == 48000, f"{fpath.name}: sample rate {info.samplerate} != 48000"
        assert info.channels == 1, f"{fpath.name}: channels {info.channels} != 1 (mono)"
        assert info.frames == 144000, f"{fpath.name}: frames {info.frames} != 144000 (3.0s)"
        assert abs(info.duration - 3.0) < 1e-4, f"{fpath.name}: duration {info.duration} != 3.0"


def test_all_74_audio_files_numerical_integrity():
    """Verify waveform data: zero NaN/Inf, low DC offset, non-zero energy (RMS > 0.001), peak <= 1.0."""
    wav_files = sorted(list(PROCESSED_DIR.glob("*.wav")))
    assert len(wav_files) == 74

    for fpath in wav_files:
        data, sr = sf.read(str(fpath), dtype="float32")
        assert len(data) == 144000, f"{fpath.name}: len {len(data)} != 144000"
        assert not np.isnan(data).any(), f"{fpath.name} contains NaN"
        assert not np.isinf(data).any(), f"{fpath.name} contains Inf"

        # DC offset should be centered near 0
        dc_offset = float(np.mean(data))
        assert abs(dc_offset) < 0.01, f"{fpath.name} excessive DC offset: {dc_offset:.6f}"

        # Energy checks
        rms = float(np.sqrt(np.mean(data ** 2)))
        assert rms > 0.001, f"{fpath.name} degenerate silent audio: RMS {rms:.6f} <= 0.001"

        # Peak normalization check
        peak = float(np.max(np.abs(data)))
        assert peak <= 1.0001, f"{fpath.name} clipped peak: {peak:.4f} > 1.0"
        assert peak >= 0.05, f"{fpath.name} abnormally quiet peak: {peak:.4f}"


def test_all_74_audio_files_checksums_and_duplicates():
    """Empirically audit audio waveform hashes across all 74 files."""
    wav_files = sorted(list(PROCESSED_DIR.glob("*.wav")))
    assert len(wav_files) == 74

    audio_hashes = {}
    duplicates = []

    for fpath in wav_files:
        data, _ = sf.read(str(fpath), dtype="float32")
        h = hashlib.sha256(data.tobytes()).hexdigest()
        if h in audio_hashes:
            duplicates.append((fpath.name, audio_hashes[h], h))
        else:
            audio_hashes[h] = fpath.name

    # Check unique count
    assert len(audio_hashes) == 73, f"Expected 73 unique hashes, got {len(audio_hashes)}"
    assert len(duplicates) == 1, f"Expected exactly 1 duplicate pair, got {len(duplicates)}"

    dup_pair = duplicates[0]
    expected_pair = {"1104271_seg000.wav", "1104272_seg000.wav"}
    actual_pair = {dup_pair[0], dup_pair[1]}
    assert actual_pair == expected_pair, f"Unexpected duplicate pair: {actual_pair}"


# ==============================================================================
# 2. Harvester Failure Resistance & Network Simulation
# ==============================================================================

def test_harvester_network_404_handling(tmp_path):
    """Simulate HTTP 404 Not Found. Verify immediate failure without retries and clean state."""
    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0, max_retries=3)
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = harvester.download_recording("999999", tmp_path, "corvus_splendens")

        assert res.status == DownloadStatus.FAILED
        assert res.file_path is None
        assert "404" in res.error_message
        # 404 should NOT trigger retry
        assert mock_get.call_count == 1

        # Verify no .part or .mp3 files remained
        assert not (tmp_path / "999999.mp3").exists()
        assert not (tmp_path / "999999.mp3.part").exists()


def test_harvester_server_500_exponential_backoff(tmp_path):
    """Simulate HTTP 500 Internal Server Error. Verify 3 retries with exponential backoff delays."""
    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0, max_retries=3)
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("requests.get", return_value=mock_resp) as mock_get:
        with patch("time.sleep") as mock_sleep:
            res = harvester.download_recording("888888", tmp_path, "corvus_splendens")

            assert res.status == DownloadStatus.FAILED
            assert mock_get.call_count == 3
            # Attempts 1 and 2 sleep before retry: 1.0 * (2^0) = 1.0, 1.0 * (2^1) = 2.0
            sleep_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert sleep_delays == [1.0, 2.0]


def test_harvester_network_timeout_retry_and_backoff(tmp_path):
    """Simulate requests.exceptions.ConnectTimeout. Verify retries and backoff."""
    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0, max_retries=3)

    with patch("requests.get", side_effect=requests.exceptions.ConnectTimeout("Connection timed out")) as mock_get:
        with patch("time.sleep") as mock_sleep:
            res = harvester.download_recording("777777", tmp_path, "corvus_splendens")

            assert res.status == DownloadStatus.FAILED
            assert mock_get.call_count == 3
            sleep_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert sleep_delays == [1.0, 2.0]
            assert "Connection timed out" in res.error_message


def test_harvester_zero_byte_download_handling(tmp_path):
    """Simulate HTTP 200 with an empty body (0 bytes). Verify .part cleanup and failure status."""
    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0, max_retries=1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b""
    mock_resp.iter_content = MagicMock(return_value=[])

    with patch("requests.get", return_value=mock_resp):
        res = harvester.download_recording("666666", tmp_path, "corvus_splendens")

        assert res.status == DownloadStatus.FAILED
        assert "0 bytes" in res.error_message.lower()
        assert not (tmp_path / "666666.mp3").exists()
        assert not (tmp_path / "666666.mp3.part").exists()


# ==============================================================================
# 3. Harvester Caching & Invalidation
# ==============================================================================

def test_harvester_cache_hit(tmp_path):
    """Verify that existing valid file (>100 bytes) skips network requests entirely."""
    sp_dir = tmp_path / "corvus_splendens"
    sp_dir.mkdir(parents=True, exist_ok=True)
    target_file = sp_dir / "1169643.mp3"
    target_file.write_bytes(b"X" * 1024)

    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0)
    with patch("requests.get") as mock_get:
        summary = harvester.harvest_species("corvus_splendens", limit=1)
        assert summary.cached_count == 1
        assert summary.downloaded_count == 0
        assert mock_get.call_count == 0


def test_harvester_cache_invalidation_corrupted_file(tmp_path):
    """Verify that a corrupted / degenerate file (<= 100 bytes) is invalidated and re-downloaded."""
    sp_dir = tmp_path / "corvus_splendens"
    sp_dir.mkdir(parents=True, exist_ok=True)
    target_file = sp_dir / "1169643.mp3"
    target_file.write_bytes(b"corrupted-tiny-payload")  # 22 bytes <= 100 bytes

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"valid-full-audio-payload-content-" * 10
    mock_resp.iter_content = MagicMock(return_value=[mock_resp.content])

    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0)
    with patch("requests.get", return_value=mock_resp) as mock_get:
        summary = harvester.harvest_species("corvus_splendens", limit=1)
        assert summary.cached_count == 0
        assert summary.downloaded_count == 1
        assert mock_get.call_count == 1


def test_harvester_force_flag_bypasses_cache(tmp_path):
    """Verify that force=True overrides valid cache and forces fresh download."""
    sp_dir = tmp_path / "corvus_splendens"
    sp_dir.mkdir(parents=True, exist_ok=True)
    target_file = sp_dir / "1169643.mp3"
    target_file.write_bytes(b"X" * 1024)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fresh-forced-download-" * 10
    mock_resp.iter_content = MagicMock(return_value=[mock_resp.content])

    harvester = XenoCantoHarvester(output_dir=tmp_path, rate_limit=0.0)
    with patch("requests.get", return_value=mock_resp) as mock_get:
        summary = harvester.harvest_species("corvus_splendens", limit=1, force=True)
        assert summary.cached_count == 0
        assert summary.downloaded_count == 1
        assert mock_get.call_count == 1


# ==============================================================================
# 4. Harvester Input Validation & Rate Limiting
# ==============================================================================

def test_harvester_invalid_species_queries():
    """Verify ValueError is raised on non-existent or invalid species queries."""
    harvester = XenoCantoHarvester(rate_limit=0.0)

    for invalid in ["nonexistent_wildlife_species", "12345", "", "Unknown Fox"]:
        with pytest.raises(ValueError) as exc_info:
            harvester.harvest_species(invalid)
        assert "Could not resolve species record" in str(exc_info.value)


def test_harvester_rate_limiting_timing():
    """Verify rate limiting delays subsequent requests."""
    harvester = XenoCantoHarvester(rate_limit=0.1)
    t0 = time.time()
    harvester._enforce_rate_limit()
    harvester._enforce_rate_limit()
    elapsed = time.time() - t0
    assert elapsed >= 0.09, f"Rate limiting failed to delay: elapsed={elapsed:.4f}s"


# ==============================================================================
# 5. Species Catalog Integrity Audit
# ==============================================================================

def test_species_catalog_completeness():
    """Verify 33 species across 4 taxonomic groups."""
    catalog = get_species_catalog()
    assert len(catalog) == 33

    counts = {"aves": 0, "insecta": 0, "amphibia": 0, "mammalia": 0}
    for sp in catalog:
        assert sp.species_id in SPECIES_CATALOG
        assert len(sp.scientific_name.split()) >= 2
        assert len(sp.common_name) > 0
        assert isinstance(sp.taxon, TaxonGroup)
        counts[sp.taxon.value] += 1
        assert len(sp.vocalization_band_hz) == 2
        low, high = sp.vocalization_band_hz
        assert 0 < low < high <= 24000
        assert sp.target_recordings >= 20
        assert len(sp.seed_recording_ids) > 0
        assert all(s.isdigit() for s in sp.seed_recording_ids)

    assert counts == {"aves": 15, "insecta": 8, "amphibia": 5, "mammalia": 5}


def test_species_catalog_seed_id_cross_taxa_overlap_audit():
    """Adversarial check: Surface and verify any shared seed recording IDs across species."""
    catalog = get_species_catalog()
    seed_to_species = {}
    shared_seeds = []

    for sp in catalog:
        for sid in set(sp.seed_recording_ids):
            if sid in seed_to_species:
                shared_seeds.append((sid, seed_to_species[sid], sp.species_id))
            else:
                seed_to_species[sid] = sp.species_id

    # Document that shared seeds exist in catalog fallback IDs
    assert len(shared_seeds) > 0
    # Specifically, verify the shared IDs between House Crow and Indian Jungle Crow / Black Kite
    crow_shared = [s for s in shared_seeds if s[1] == "corvus_splendens" and s[2] == "corvus_culminatus"]
    assert len(crow_shared) == 3  # "744704", "683047", "604023"
