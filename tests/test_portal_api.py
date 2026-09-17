"""Unit and integration tests for AnyCall Field Control Portal REST API."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from anycall.portal.api import create_app
from tests.fixtures.synth_audio import generate_animal_call, save_wav_file, generate_broadband_noise


from anycall.storage.db import DatabaseManager


@pytest.fixture
def portal_client():
    tmp = tempfile.TemporaryDirectory()
    db_path = str(Path(tmp.name) / "test_portal.db")
    upload_dir = str(Path(tmp.name) / "uploads")
    app = create_app(
        db_path=db_path,
        backbone_name="mock",
        default_threshold=0.65,
        upload_dir=upload_dir,
    )
    with TestClient(app) as client:
        yield client
    DatabaseManager.close_all()
    try:
        tmp.cleanup()
    except Exception:
        pass


import wave


def _synth_wav_bytes(taxon: str = "aves", seed: int = 42) -> io.BytesIO:
    """Generate in-memory WAV bytes of synthetic animal call."""
    audio = generate_animal_call(taxon=taxon, duration=3.0, sr=48000, seed=seed)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf


def test_portal_stats_endpoint(portal_client):
    """Verifies /api/stats returns active status, backbone, and counts."""
    res = portal_client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["active_backbone"] == "mock"
    assert 0.0 <= data["threshold"] <= 1.0
    assert "total_detections" in data
    assert "total_enrolled_species" in data


def test_portal_species_enrollment_lifecycle(portal_client):
    """Enrolls a new species with synthetic audio and checks it appears in /api/species."""
    wav_bytes = _synth_wav_bytes("aves", seed=10)
    files = [("files", ("crow1.wav", wav_bytes, "audio/wav"))]
    data = {
        "species_id": "corvus_splendens",
        "common_name": "House Crow",
        "taxon": "Aves",
    }

    res = portal_client.post("/api/species/enroll", data=data, files=files)
    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "success"
    assert result["species_id"] == "corvus_splendens"
    assert result["sample_count"] >= 1

    # Verify presence in list
    list_res = portal_client.get("/api/species")
    assert list_res.status_code == 200
    sp_list = list_res.json()["species"]
    assert any(s["species_id"] == "corvus_splendens" for s in sp_list)


def test_portal_species_incremental_add_samples(portal_client):
    """Tests adding new exemplars to an existing species increments sample count."""
    # First enroll
    wav1 = _synth_wav_bytes("aves", seed=1)
    portal_client.post(
        "/api/species/enroll",
        data={"species_id": "sp_test", "common_name": "Test Species", "taxon": "Aves"},
        files=[("files", ("call1.wav", wav1, "audio/wav"))],
    )

    # Add 2 more exemplars
    wav2 = _synth_wav_bytes("aves", seed=2)
    wav3 = _synth_wav_bytes("aves", seed=3)
    res = portal_client.post(
        "/api/species/sp_test/add-samples",
        files=[
            ("files", ("call2.wav", wav2, "audio/wav")),
            ("files", ("call3.wav", wav3, "audio/wav")),
        ],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["new_sample_count"] >= 2


def test_portal_audio_classification_and_detection_logging(portal_client):
    """Uploads audio to /api/classify, checks top prediction, and verifies detection log."""
    # Enroll a bird
    wav_enroll = _synth_wav_bytes("aves", seed=10)
    portal_client.post(
        "/api/species/enroll",
        data={"species_id": "corvus_splendens", "common_name": "House Crow", "taxon": "Aves"},
        files=[("files", ("crow_enroll.wav", wav_enroll, "audio/wav"))],
    )

    # Classify same call
    wav_query = _synth_wav_bytes("aves", seed=10)
    res = portal_client.post(
        "/api/classify",
        files={"file": ("query.wav", wav_query, "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] == "corvus_splendens"
    assert data["is_known"] is True
    assert data["confidence"] > 0.65

    # Check detection log
    det_res = portal_client.get("/api/detections")
    assert det_res.status_code == 200
    detections = det_res.json()["detections"]
    assert len(detections) >= 1
    assert detections[0]["species_id"] == "corvus_splendens"


def test_portal_unidentified_quarantine_and_cluster_promotion(portal_client):
    """Unknown sounds scoring below threshold are quarantined, clustered, and promoted."""
    # Enroll bird
    portal_client.post(
        "/api/species/enroll",
        data={"species_id": "corvus_splendens", "common_name": "House Crow", "taxon": "Aves"},
        files=[("files", ("crow.wav", _synth_wav_bytes("aves", seed=1), "audio/wav"))],
    )

    # Submit 2 mystery insect calls (not enrolled)
    mystery1 = _synth_wav_bytes("insecta", seed=100)
    mystery2 = _synth_wav_bytes("insecta", seed=100)
    res1 = portal_client.post("/api/classify", files={"file": ("m1.wav", mystery1, "audio/wav")})
    res2 = portal_client.post("/api/classify", files={"file": ("m2.wav", mystery2, "audio/wav")})
    assert res1.status_code == 200
    assert res2.status_code == 200

    # Query clusters
    clusters_res = portal_client.get("/api/unidentified/clusters?min_size=2")
    assert clusters_res.status_code == 200
    clusters_data = clusters_res.json()
    assert clusters_data["candidate_clusters_found"] >= 1

    first_cluster = clusters_data["clusters"][0]
    cid = first_cluster["cluster_id"]

    # Promote cluster to known species
    promote_res = portal_client.post(
        "/api/unidentified/promote",
        json={
            "cluster_id": cid,
            "species_id": "discovered_cricket",
            "common_name": "Discovered Cricket",
            "taxon": "Insecta",
        },
    )
    assert promote_res.status_code == 200
    promote_data = promote_res.json()
    assert promote_data["species_id"] == "discovered_cricket"

    # Now verify it's recognized as known species in classifier
    query_promoted = _synth_wav_bytes("insecta", seed=100)
    class_res = portal_client.post("/api/classify", files={"file": ("check.wav", query_promoted, "audio/wav")})
    assert class_res.status_code == 200
    assert class_res.json()["predicted_label"] == "discovered_cricket"
    assert class_res.json()["is_known"] is True


def test_portal_settings_update(portal_client):
    """Tests updating threshold and switching backbone via /api/settings."""
    res = portal_client.post("/api/settings", json={"threshold": 0.78})
    assert res.status_code == 200
    data = res.json()
    assert abs(data["current_threshold"] - 0.78) < 1e-4

    # Invalid threshold rejected
    bad_res = portal_client.post("/api/settings", json={"threshold": 1.5})
    assert bad_res.status_code == 400


def test_portal_mic_devices_endpoint(portal_client):
    """Tests querying available audio input devices via /api/mic/devices."""
    res = portal_client.get("/api/mic/devices")
    assert res.status_code == 200
    data = res.json()
    assert "devices" in data
    assert isinstance(data["devices"], list)


def test_portal_mic_status_and_lifecycle(portal_client):
    """Tests starting, checking status, and stopping the continuous mic listener."""
    # Check initial status
    res = portal_client.get("/api/mic/status")
    assert res.status_code == 200
    status = res.json()
    assert "is_running" in status
    assert "current_rms" in status

    # Start mic
    start_res = portal_client.post("/api/mic/start", json={})
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["status"] == "started"
    assert start_data["details"]["is_running"] is True

    # Stop mic
    stop_res = portal_client.post("/api/mic/stop")
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["status"] == "stopped"
    assert stop_data["details"]["is_running"] is False

