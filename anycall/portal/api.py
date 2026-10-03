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
from anycall.data.species import SPECIES_CATALOG
from anycall.embeddings import get_backbone
from anycall.storage.db import DatabaseManager
from anycall.stream.adaptive_listener import AdaptiveSpectrogramListener
from anycall.stream.spectrogram import SpectrogramEventDetector


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
    default_threshold: float = 0.70,
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

    # Shared HPSS detector — used by ALL enrollment and classify routes to guarantee
    # that every embedding is extracted with the same pipeline as the live listener.
    hpss_detector = SpectrogramEventDetector(sample_rate=backbone.target_sample_rate)

    def hpss_extract_clips(audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Run HPSS detection on audio and return a list of centered 3-second clips.

        This is the single canonical preprocessing function for the entire system.
        Both live microphone inference and file-upload enrollment/classify go through here,
        ensuring no distribution mismatch between prototypes and queries.
        """
        import librosa
        # Resample to target sr if needed
        if sr != backbone.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=backbone.target_sample_rate)
            sr = backbone.target_sample_rate

        target_len = int(backbone.target_duration_seconds * sr)
        events = hpss_detector.find_events(audio)

        clips: List[np.ndarray] = []
        for (start_t, end_t) in events:
            center = (start_t + end_t) / 2.0
            win_start = max(0.0, center - backbone.target_duration_seconds / 2)
            win_end = win_start + backbone.target_duration_seconds
            # Clamp to buffer
            if win_end > len(audio) / sr:
                win_end = len(audio) / sr
                win_start = max(0.0, win_end - backbone.target_duration_seconds)

            s_idx = int(win_start * sr)
            e_idx = int(win_end * sr)
            clip = audio[s_idx:e_idx]
            if len(clip) < target_len:
                clip = np.pad(clip, (0, target_len - len(clip)), "constant")
            clips.append(clip.astype(np.float32))

        if not clips:
            # No events detected — treat the whole file as one clip (center-crop/pad)
            if len(audio) >= target_len:
                start = (len(audio) - target_len) // 2
                clips = [audio[start: start + target_len].astype(np.float32)]
            else:
                clips = [np.pad(audio, (0, target_len - len(audio)), "constant").astype(np.float32)]

        return clips

    # Load existing prototypes into classifier
    try:
        existing_rows = db_mgr._conn.execute("SELECT * FROM species").fetchall()
        for r in existing_rows:
            sp_id = r["species_id"]
            proto_bytes = r["prototype"]
            vec = np.frombuffer(proto_bytes, dtype=np.float32)
            if r.get("sub_prototypes"):
                subs = np.frombuffer(r["sub_prototypes"], dtype=np.float32).reshape(-1, vec.shape[0])
                classifier.enroll(sp_id, [subs[i] for i in range(subs.shape[0])])
            else:
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
    mic_listener = AdaptiveSpectrogramListener(
        classifier=classifier,
        backbone=backbone,
        db_mgr=db_mgr,
        sound_bank=sound_bank,
        multi_label_threshold=default_threshold,
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
    def list_devices():
        """List available input devices (if sounddevice is installed)."""
        import sounddevice as sd
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append({
                    "index": idx,
                    "name": dev.get("name", f"Device #{idx}"),
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": int(dev.get("default_samplerate", 48000)),
                    "is_default": (idx == default_input),
                })
        return {"devices": input_devices}

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
        """Fetch chronological log of detection events with species common names."""
        query = """
            SELECT d.*, s.common_name as sp_common_name, s.taxon as sp_taxon 
            FROM detections d 
            LEFT JOIN species s ON d.species_id = s.species_id
        """
        params: List[Any] = []
        where_clauses: List[str] = []

        if species_id:
            where_clauses.append("d.species_id = ?")
            params.append(species_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY d.id DESC LIMIT ?"
        params.append(limit)

        rows = db_mgr._conn.execute(query, params).fetchall()
        detections = []
        for r in rows:
            d = dict(r)
            d["confidence"] = round(float(d["confidence"]), 4)
            d["is_known"] = d["species_id"].lower() != "unknown"

            # Resolve common_name and taxon
            if d.get("sp_common_name"):
                d["common_name"] = d["sp_common_name"]
            else:
                meta = SPECIES_CATALOG.get(d["species_id"])
                if meta:
                    d["common_name"] = meta.common_name
                    d["taxon"] = meta.taxon.value.capitalize() if hasattr(meta.taxon, "value") else str(meta.taxon).capitalize()
                else:
                    d["common_name"] = d["species_id"].replace("_", " ").title()

            if d.get("sp_taxon"):
                d["taxon"] = str(d["sp_taxon"]).capitalize()

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
        """Enroll a new species prototype using few-shot audio exemplars without retraining.
        
        Uses the same HPSS preprocessing pipeline as the live microphone listener so that
        enrolled prototypes and live queries occupy the same embedding distribution.
        """
        if not files:
            raise HTTPException(status_code=400, detail="At least one audio file must be uploaded.")

        import soundfile as sf
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
                audio, sr = sf.read(str(file_path), dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                # Run HPSS-based clip extraction (same as live listener)
                clips = hpss_extract_clips(audio, sr)
                for clip in clips:
                    emb = backbone.embed(clip, sr=backbone.target_sample_rate)
                    extracted_embeddings.append(emb)
                processed_files.append(f.filename)
                print(f"[Portal] Enrolled {len(clips)} HPSS clips from {f.filename}")
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
            "message": f"Successfully enrolled '{clean_sp_id}' with {len(extracted_embeddings)} HPSS exemplars.",
            "species_id": clean_sp_id,
            "sample_count": len(extracted_embeddings),
            "files_processed": processed_files,
        }


    @app.post("/api/species/{species_id}/add-samples")
    async def add_species_samples(
        species_id: str,
        files: List[UploadFile] = File(...),
    ):
        """Incrementally update an existing species prototype with new exemplars.
        
        Uses the same HPSS preprocessing pipeline as the live microphone listener.
        """
        import soundfile as sf
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
                audio, sr = sf.read(str(temp_path), dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                clips = hpss_extract_clips(audio, sr)
                for clip in clips:
                    emb = backbone.embed(clip, sr=backbone.target_sample_rate)
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
            "message": f"Updated '{clean_sp_id}' prototype with {len(new_embeddings)} new HPSS exemplars (total: {new_count}).",
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
        """Classify an audio recording using HPSS preprocessing + top-1 nearest-centroid matching.
        
        Uses identical preprocessing to the live listener so uploaded file results are comparable.
        """
        import soundfile as sf
        save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        saved_audio_path = uploads_path / save_name

        with open(saved_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            audio, sr = sf.read(str(saved_audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            clips = hpss_extract_clips(audio, sr)
            effective_theta = threshold if threshold is not None else classifier.threshold
            MIN_MARGIN = 0.01

            segment_results = []
            for idx, clip in enumerate(clips):
                emb = backbone.embed(clip, sr=backbone.target_sample_rate)
                pred: PredictionResult = classifier.predict(emb, threshold=effective_theta)

                sorted_scores = sorted(pred.scores.items(), key=lambda x: x[1], reverse=True)
                best_label, best_score = sorted_scores[0] if sorted_scores else ("Unknown", 0.0)
                second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
                margin = best_score - second_score

                is_match = pred.is_known and best_score >= effective_theta and margin >= MIN_MARGIN
                final_label = best_label if is_match else "Unknown"
                audio_hash = hashlib.sha256(clip.tobytes()).hexdigest()[:16]

                db_mgr.log_detection(
                    species_id=final_label,
                    confidence=best_score,
                    audio_hash=audio_hash,
                    threshold=effective_theta,
                )

                if not is_match:
                    sound_bank.add(
                        embedding=emb,
                        audio_path=str(saved_audio_path),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    db_mgr.save_unidentified(
                        embedding=emb,
                        best_match=best_label,
                        score=best_score,
                        audio_path=str(saved_audio_path),
                    )

                # Build candidate list with common names
                cand_list = []
                for k, v in sorted_scores[:5]:
                    meta = SPECIES_CATALOG.get(k)
                    sp_common = meta.common_name if meta else k.replace("_", " ").title()
                    cand_list.append({
                        "species": k,
                        "common_name": sp_common,
                        "score": round(v, 4)
                    })

                meta_final = SPECIES_CATALOG.get(final_label)
                final_common = meta_final.common_name if meta_final else (final_label if final_label == "Unknown" else final_label.replace("_", " ").title())

                segment_results.append({
                    "segment_index": idx,
                    "predicted_label": final_label,
                    "common_name": final_common,
                    "confidence": round(best_score, 4),
                    "margin": round(margin, 4),
                    "is_known": is_match,
                    "candidates": cand_list,
                })

            # Headline: best-confidence confirmed match, or best candidate if all unknown
            known = [s for s in segment_results if s["is_known"]]
            best_seg = max(known or segment_results, key=lambda x: x["confidence"])

            return {
                "filename": file.filename,
                "audio_url": f"/api/audio/{save_name}",
                "segments_analyzed": len(clips),
                "predicted_label": best_seg["predicted_label"],
                "common_name": best_seg.get("common_name", best_seg["predicted_label"].replace("_", " ").title()),
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
            sub_prototypes=None
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
