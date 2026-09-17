"""AnyCall Field Portal — FastAPI REST API and Control Backend."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from anycall.audio.standardize import standardize_audio
from anycall.audio.vad import slice_audio_segments
from anycall.classifier.engine import (
    NovelCluster,
    PredictionResult,
    PrototypicalClassifier,
    UnidentifiedSoundBank,
)
from anycall.embeddings import get_backbone
from anycall.storage.db import DatabaseManager
from anycall.stream.listener import ContinuousMicrophoneListener


class MicStartRequest(BaseModel):
    device: Optional[int] = None


class SettingUpdateRequest(BaseModel):
    threshold: Optional[float] = None
    backbone: Optional[str] = None


class PromoteClusterRequest(BaseModel):
    cluster_id: str
    species_id: str
    common_name: str
    taxon: str


def create_app(
    db_path: str = "anycall.db",
    backbone_name: str = "birdnet",
    default_threshold: float = 0.65,
    upload_dir: str = "data/uploads",
) -> FastAPI:
    """Factory creating and configuring the AnyCall Portal FastAPI application."""
    app = FastAPI(
        title="AnyCall Field Station Portal",
        description="Autonomous wildlife acoustic classifier, prototype enrollment, and novel cluster explorer.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State storage
    uploads_path = Path(upload_dir)
    uploads_path.mkdir(parents=True, exist_ok=True)

    db_mgr = DatabaseManager(db_path)
    classifier = PrototypicalClassifier(threshold=default_threshold)
    sound_bank = UnidentifiedSoundBank(cluster_similarity=0.70)

    # Initialize backbone
    current_backbone_name = backbone_name
    try:
        backbone = get_backbone(current_backbone_name)
    except Exception as err:
        print(f"[Portal] Warning: failed to load {current_backbone_name}: {err}. Falling back to mock.")
        current_backbone_name = "mock"
        backbone = get_backbone("mock")

    # Load existing prototypes into classifier
    try:
        existing_rows = db_mgr._conn.execute("SELECT * FROM species").fetchall()
        for r in existing_rows:
            sp_id = r["species_id"]
            proto_bytes = r["prototype"]
            vec = np.frombuffer(proto_bytes, dtype=np.float32)
            classifier.enroll(sp_id, [vec])
        print(f"[Portal] Loaded {len(existing_rows)} enrolled species from {db_path}")
    except Exception as e:
        print(f"[Portal] Note: No existing species loaded: {e}")

    # Load previously stored unidentified sounds into sound_bank
    try:
        unidentified_records = db_mgr.get_unidentified(limit=500)
        for u in unidentified_records:
            sound_bank.add(
                embedding=u["embedding"],
                audio_path=u.get("audio_path", ""),
                timestamp=u.get("detected_at", ""),
            )
        print(f"[Portal] Loaded {len(unidentified_records)} unidentified sounds into memory bank")
    except Exception as e:
        print(f"[Portal] Note: No existing unidentified sounds loaded: {e}")

    # Continuous Microphone Stream Listener
    mic_listener = ContinuousMicrophoneListener(
        classifier=classifier,
        backbone=backbone,
        db_mgr=db_mgr,
        sound_bank=sound_bank,
        rejection_threshold=default_threshold,
    )
    app.state.mic_listener = mic_listener

    @app.on_event("shutdown")
    def shutdown_listener():
        mic_listener.stop()

    # Mount static assets
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "AnyCall Field Station API is active. Static UI not found."}

    # ----------------------------------------------------------------------
    # Continuous Microphone Monitoring
    # ----------------------------------------------------------------------

    @app.get("/api/mic/status")
    def get_mic_status():
        """Retrieve real-time telemetry and running state of continuous mic listener."""
        return mic_listener.get_status()

    @app.get("/api/mic/devices")
    def get_mic_devices():
        """List available physical audio input devices."""
        return {"devices": ContinuousMicrophoneListener.list_input_devices()}

    @app.post("/api/mic/start")
    def start_mic(req: Optional[MicStartRequest] = None):
        """Start continuous microphone monitoring."""
        dev = req.device if req else None
        try:
            mic_listener.start(device=dev)
            return {"status": "started", "details": mic_listener.get_status()}
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Failed to start microphone: {ex}")

    @app.post("/api/mic/stop")
    def stop_mic():
        """Stop continuous microphone monitoring."""
        mic_listener.stop()
        return {"status": "stopped", "details": mic_listener.get_status()}

    # ----------------------------------------------------------------------
    # System Stats & Metadata
    # ----------------------------------------------------------------------

    @app.get("/api/stats")
    def get_stats():
        """Retrieve real-time telemetry and monitoring statistics."""
        try:
            total_detections = db_mgr._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        except Exception:
            total_detections = 0

        try:
            total_species = db_mgr._conn.execute("SELECT COUNT(*) FROM species").fetchone()[0]
        except Exception:
            total_species = len(classifier.enrolled_species)

        try:
            total_unidentified = sound_bank.total_unidentified
        except Exception:
            total_unidentified = 0

        # Discovered candidate clusters
        discovered = sound_bank.discover_clusters(min_cluster_size=2)

        # Most frequent detections
        top_species = []
        try:
            rows = db_mgr._conn.execute("""
                SELECT species_id, COUNT(*) as cnt, AVG(confidence) as avg_conf
                FROM detections
                GROUP BY species_id
                ORDER BY cnt DESC
                LIMIT 5
            """).fetchall()
            top_species = [
                {"species_id": r[0], "count": r[1], "avg_confidence": round(float(r[2]), 3)}
                for r in rows
            ]
        except Exception:
            pass

        return {
            "status": "online",
            "active_backbone": current_backbone_name,
            "threshold": classifier.threshold,
            "total_detections": total_detections,
            "total_enrolled_species": total_species,
            "total_unidentified": total_unidentified,
            "candidate_novel_clusters": len(discovered),
            "top_species": top_species,
            "database_path": db_path,
        }

    # ----------------------------------------------------------------------
    # Detections Feed
    # ----------------------------------------------------------------------

    @app.get("/api/detections")
    def list_detections(
        limit: int = Query(50, ge=1, le=500),
        species_id: Optional[str] = None,
        is_known: Optional[bool] = None,
    ):
        """Fetch chronological log of detection events."""
        query = "SELECT * FROM detections"
        params: List[Any] = []
        where_clauses: List[str] = []

        if species_id:
            where_clauses.append("species_id = ?")
            params.append(species_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = db_mgr._conn.execute(query, params).fetchall()
        detections = []
        for r in rows:
            d = dict(r)
            d["confidence"] = round(float(d["confidence"]), 4)
            d["is_known"] = d["species_id"].lower() != "unknown"
            detections.append(d)

        return {"total": len(detections), "detections": detections}

    @app.delete("/api/detections")
    @app.post("/api/detections/clear")
    def clear_detections():
        """Clear all historical detection events from the database."""
        count = db_mgr.clear_detections()
        return {
            "status": "success",
            "message": f"Cleared {count} detection events.",
            "cleared_count": count,
        }

    # ----------------------------------------------------------------------
    # Species Management (Few-Shot Enrollment)
    # ----------------------------------------------------------------------

    @app.get("/api/species")
    def list_species():
        """List all currently enrolled species prototypes."""
        rows = db_mgr._conn.execute(
            "SELECT species_id, common_name, taxon, radius, sample_count, updated_at FROM species ORDER BY common_name"
        ).fetchall()
        species_list = [dict(r) for r in rows]
        return {"total": len(species_list), "species": species_list}

    @app.post("/api/species/enroll")
    async def enroll_species(
        species_id: str = Form(...),
        common_name: str = Form(""),
        taxon: str = Form("Aves"),
        files: List[UploadFile] = File(...),
    ):
        """Enroll a new species prototype using few-shot audio exemplars without retraining."""
        if not files:
            raise HTTPException(status_code=400, detail="At least one audio file must be uploaded.")

        clean_sp_id = species_id.lower().strip().replace(" ", "_")
        sp_upload_dir = uploads_path / clean_sp_id
        sp_upload_dir.mkdir(parents=True, exist_ok=True)

        extracted_embeddings: List[np.ndarray] = []
        processed_files: List[str] = []

        for f in files:
            file_path = sp_upload_dir / f.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)

            try:
                std_audio, sr = standardize_audio(file_path, target_sr=backbone.target_sample_rate)
                slices = slice_audio_segments(
                    std_audio,
                    sr=sr,
                    segment_duration=backbone.target_duration_seconds,
                    vad_filter=True,
                )
                if not slices:
                    # Fallback to unpadded slice if VAD discarded everything
                    slices = [std_audio[: int(sr * backbone.target_duration_seconds)]]

                for s in slices:
                    emb = backbone.embed(s, sr=sr)
                    extracted_embeddings.append(emb)
                processed_files.append(f.filename)
            except Exception as ex:
                print(f"[Portal] Warning: failed processing {f.filename}: {ex}")

        if not extracted_embeddings:
            raise HTTPException(
                status_code=400,
                detail="Could not extract valid audio features from the uploaded files.",
            )

        # Compute centroid prototype
        classifier.enroll(clean_sp_id, extracted_embeddings)
        centroid = classifier.compute_prototype(extracted_embeddings)

        # Persist to database
        db_mgr.save_prototype(
            species_id=clean_sp_id,
            common_name=common_name or clean_sp_id.replace("_", " ").title(),
            taxon=taxon or "Unknown",
            prototype=centroid,
            radius=0.15,
            sample_count=len(extracted_embeddings),
        )

        return {
            "status": "success",
            "message": f"Successfully enrolled '{clean_sp_id}' with {len(extracted_embeddings)} exemplars.",
            "species_id": clean_sp_id,
            "sample_count": len(extracted_embeddings),
            "files_processed": processed_files,
        }

    @app.post("/api/species/{species_id}/add-samples")
    async def add_species_samples(
        species_id: str,
        files: List[UploadFile] = File(...),
    ):
        """Incrementally update an existing species prototype with new exemplars."""
        clean_sp_id = species_id.lower().strip().replace(" ", "_")
        existing = db_mgr.get_prototype(clean_sp_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Species '{clean_sp_id}' not found.")

        new_embeddings: List[np.ndarray] = []
        for f in files:
            temp_path = uploads_path / f"temp_{f.filename}"
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(f.file, buf)

            try:
                std_audio, sr = standardize_audio(temp_path, target_sr=backbone.target_sample_rate)
                slices = slice_audio_segments(
                    std_audio,
                    sr=sr,
                    segment_duration=backbone.target_duration_seconds,
                    vad_filter=True,
                )
                for s in slices:
                    emb = backbone.embed(s, sr=sr)
                    new_embeddings.append(emb)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        if not new_embeddings:
            raise HTTPException(status_code=400, detail="No active vocal segments found in uploads.")

        # Incremental prototype update
        updated_centroid = classifier.update_prototype(clean_sp_id, new_embeddings)
        new_count = existing["sample_count"] + len(new_embeddings)

        db_mgr.save_prototype(
            species_id=clean_sp_id,
            common_name=existing["common_name"],
            taxon=existing["taxon"],
            prototype=updated_centroid,
            radius=existing["radius"],
            sample_count=new_count,
        )

        return {
            "status": "success",
            "message": f"Updated '{clean_sp_id}' prototype with {len(new_embeddings)} new exemplars (total: {new_count}).",
            "species_id": clean_sp_id,
            "new_sample_count": new_count,
        }

    # ----------------------------------------------------------------------
    # Live Audio Inference / Upload Classifier
    # ----------------------------------------------------------------------

    @app.post("/api/classify")
    async def classify_audio(
        file: UploadFile = File(...),
        threshold: Optional[float] = Form(None),
    ):
        """Classify an audio recording, logging known detections or quarantining unknowns."""
        save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        saved_audio_path = uploads_path / save_name

        with open(saved_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            std_audio, sr = standardize_audio(saved_audio_path, target_sr=backbone.target_sample_rate)
            slices = slice_audio_segments(
                std_audio,
                sr=sr,
                segment_duration=backbone.target_duration_seconds,
                vad_filter=True,
            )
            if not slices:
                # Fallback if VAD rejects (e.g. faint call)
                target_len = int(sr * backbone.target_duration_seconds)
                slices = [std_audio[:target_len] if len(std_audio) >= target_len else np.pad(std_audio, (0, target_len - len(std_audio)))]

            effective_theta = threshold if threshold is not None else classifier.threshold

            segment_results = []
            for idx, seg in enumerate(slices):
                emb = backbone.embed(seg, sr=sr)
                pred: PredictionResult = classifier.predict(emb, threshold=effective_theta)

                audio_hash = hashlib.sha256(seg.tobytes()).hexdigest()[:16]

                # Log to detections table
                db_mgr.log_detection(
                    species_id=pred.predicted_label,
                    confidence=pred.confidence,
                    audio_hash=audio_hash,
                    threshold=effective_theta,
                )

                # If unknown / rejected, quarantine to UnidentifiedSoundBank
                if not pred.is_known:
                    sound_bank.add(
                        embedding=emb,
                        audio_path=str(saved_audio_path),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    db_mgr.save_unidentified(
                        embedding=emb,
                        best_match=pred.predicted_label,
                        score=pred.confidence,
                        audio_path=str(saved_audio_path),
                    )

                # Sort candidates
                sorted_scores = sorted(
                    pred.scores.items(), key=lambda item: item[1], reverse=True
                )[:5]

                segment_results.append({
                    "segment_index": idx,
                    "predicted_label": pred.predicted_label,
                    "confidence": round(pred.confidence, 4),
                    "is_known": pred.is_known,
                    "candidates": [
                        {"species": k, "score": round(v, 4)} for k, v in sorted_scores
                    ],
                })

            # Headline prediction is segment with highest confidence
            best_seg = max(segment_results, key=lambda x: x["confidence"])

            return {
                "filename": file.filename,
                "audio_url": f"/api/audio/{save_name}",
                "segments_analyzed": len(slices),
                "predicted_label": best_seg["predicted_label"],
                "confidence": best_seg["confidence"],
                "is_known": best_seg["is_known"],
                "threshold_applied": effective_theta,
                "segments": segment_results,
            }

        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Classification failed: {err}")

    # ----------------------------------------------------------------------
    # Mystery Sounds & Novel Clusters Explorer
    # ----------------------------------------------------------------------

    @app.get("/api/unidentified/clusters")
    def get_novel_clusters(min_size: int = Query(2, ge=2, le=50)):
        """Discover and return recurring mystery sound clusters."""
        clusters = sound_bank.discover_clusters(min_cluster_size=min_size)
        res = []
        for c in clusters:
            # Map sample paths to stream URLs
            audio_urls = [
                f"/api/audio/{Path(p).name}" for p in c.audio_paths if Path(p).exists()
            ]
            res.append({
                "cluster_id": c.cluster_id,
                "occurrences": c.n_samples,
                "cohesion_score": c.cohesion,
                "audio_urls": audio_urls[:5],
                "sample_indices": c.sample_indices,
            })
        return {
            "total_unidentified": sound_bank.total_unidentified,
            "candidate_clusters_found": len(res),
            "clusters": res,
        }

    @app.post("/api/unidentified/promote")
    def promote_cluster(req: PromoteClusterRequest):
        """Promote a discovered novel cluster to an enrolled species without retraining."""
        if req.cluster_id not in sound_bank._clusters:
            raise HTTPException(status_code=404, detail=f"Cluster '{req.cluster_id}' not found.")

        cluster: NovelCluster = sound_bank._clusters[req.cluster_id]
        clean_sp_id = req.species_id.lower().strip().replace(" ", "_")

        # Promote in classifier
        sound_bank.promote_to_species(req.cluster_id, clean_sp_id, classifier)

        # Save to database
        db_mgr.save_prototype(
            species_id=clean_sp_id,
            common_name=req.common_name or clean_sp_id.replace("_", " ").title(),
            taxon=req.taxon or "Unknown",
            prototype=cluster.centroid,
            radius=0.15,
            sample_count=cluster.n_samples,
        )

        return {
            "status": "success",
            "message": f"Successfully promoted {req.cluster_id} to '{clean_sp_id}' ({cluster.n_samples} exemplars).",
            "species_id": clean_sp_id,
            "sample_count": cluster.n_samples,
        }

    @app.delete("/api/unidentified")
    @app.post("/api/unidentified/clear")
    def clear_unidentified():
        """Clear all quarantined mystery sounds and reset sound bank."""
        count = db_mgr.clear_unidentified()
        sound_bank.clear()
        return {
            "status": "success",
            "message": f"Cleared {count} quarantined mystery sounds.",
            "cleared_count": count,
        }

    # ----------------------------------------------------------------------
    # Field Settings & System Controls
    # ----------------------------------------------------------------------

    @app.post("/api/settings")
    def update_settings(req: SettingUpdateRequest):
        """Update runtime threshold or switch active backbone."""
        nonlocal current_backbone_name, backbone

        changes = []
        if req.threshold is not None:
            if not 0.0 <= req.threshold <= 1.0:
                raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
            classifier.threshold = float(req.threshold)
            mic_listener.rejection_threshold = classifier.threshold
            changes.append(f"threshold set to {classifier.threshold:.2f}")

        if req.backbone is not None and req.backbone != current_backbone_name:
            try:
                new_bb = get_backbone(req.backbone)
                backbone = new_bb
                current_backbone_name = req.backbone
                mic_listener.backbone = backbone
                changes.append(f"active backbone switched to {current_backbone_name}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed switching backbone: {e}")

        return {
            "status": "success",
            "message": ", ".join(changes) or "No settings changed.",
            "current_threshold": classifier.threshold,
            "current_backbone": current_backbone_name,
        }

    @app.get("/api/audio/{filename}")
    def get_audio_file(filename: str):
        """Stream an audio file for in-browser playback."""
        target = uploads_path / filename
        if not target.exists():
            detections_match = Path("data/detections") / filename
            if detections_match.exists():
                target = detections_match
            else:
                # Search in processed data as fallback
                processed_match = list(Path("data/processed").glob(f"**/{filename}"))
                if processed_match:
                    target = processed_match[0]
                else:
                    raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")

        media_type = "audio/wav" if target.suffix.lower() == ".wav" else "audio/mpeg"
        return FileResponse(str(target), media_type=media_type)

    return app
