import logging
import threading
import time
import hashlib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import librosa
import soundfile as sf

from anycall.classifier.engine import PredictionResult, PrototypicalClassifier, UnidentifiedSoundBank
from anycall.embeddings.base import BaseAudioEmbeddingBackbone
from anycall.storage.db import DatabaseManager
from anycall.stream.spectrogram import SpectrogramEventDetector

logger = logging.getLogger("anycall.stream.adaptive")

class AdaptiveSpectrogramListener:
    """Live microphone ingestion using continuous Spectrogram Feature Extraction (Adaptive ROI)."""

    def __init__(
        self,
        classifier: PrototypicalClassifier,
        backbone: BaseAudioEmbeddingBackbone,
        db_mgr: DatabaseManager,
        sound_bank: Optional[UnidentifiedSoundBank] = None,
        sample_rate: int = 48000,
        buffer_seconds: float = 4.0,
        hop_seconds: float = 2.0,
        detections_dir: Union[str, Path] = "data/detections",
        multi_label_threshold: float = 0.50,
        # Minimum RMS amplitude before we even run HPSS inference.
        # Real bird calls played at reasonable volume through a speaker: ~0.01+
        # Quiet room ambient noise (AC, fan, breath): ~0.001-0.004
        noise_floor_rms: float = 0.008,
    ):
        self.classifier = classifier
        self.backbone = backbone
        self.db_mgr = db_mgr
        self.sound_bank = sound_bank if sound_bank else UnidentifiedSoundBank(cluster_similarity=0.70)
        
        self.sample_rate = int(sample_rate)
        self.buffer_seconds = float(buffer_seconds)
        self.hop_seconds = float(hop_seconds)
        self.hop_samples = int(round(self.sample_rate * self.hop_seconds))
        self.buffer_samples = int(round(self.sample_rate * self.buffer_seconds))
        
        self.multi_label_threshold = multi_label_threshold
        self.noise_floor_rms = noise_floor_rms
        
        self.detections_dir = Path(detections_dir)
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        
        # Spectrogram event detector
        self.detector = SpectrogramEventDetector(sample_rate=self.sample_rate)
        
        self._is_running = False
        self._stream = None
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._audio_buffer = deque(maxlen=self.buffer_samples)
        self._new_samples_count = 0
        
        # Deduplication: don't log the exact same audio twice within a short window
        self._recent_hashes: deque = deque(maxlen=20)
        
        # Telemetry
        self._current_rms = 0.0
        self._last_detections: List[Dict[str, Any]] = []


    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "current_rms": round(self._current_rms, 6),
            "last_detection": self._last_detections[-1] if self._last_detections else None,
            "rejection_threshold": self.multi_label_threshold,
            "detections_logged": self.db_mgr._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] if hasattr(self, 'db_mgr') and self.db_mgr else 0,
        }

    def _audio_callback(self, indata, frames, time_info, status):
        """Low-latency callback from sounddevice InputStream."""
        if status:
            logger.debug(f"Audio stream status flag: {status}")
        if not self._is_running:
            return

        mono = np.mean(indata, axis=1, dtype=np.float32) if indata.ndim > 1 else indata.flatten().astype(np.float32)
        
        self._current_rms = float(np.sqrt(np.mean(mono ** 2, dtype=np.float64) + 1e-12))
        
        with self._lock:
            self._audio_buffer.extend(mono.tolist())
            self._new_samples_count += len(mono)

    def _inference_worker_loop(self):
        """Background worker thread continuously evaluating incoming audio chunks."""
        logger.info("[AdaptiveMic] Spectrogram ingestion thread started.")

        while self._is_running:
            has_enough = False
            with self._lock:
                if len(self._audio_buffer) == self.buffer_samples and self._new_samples_count >= self.hop_samples:
                    buf_list = list(self._audio_buffer)
                    # We process the latest `hop_samples` + some overlap context
                    window_audio = np.array(buf_list, dtype=np.float32)
                    self._new_samples_count = 0
                    has_enough = True

            if not has_enough:
                time.sleep(0.05)
                continue

            # --- GATE 1: Absolute amplitude check ---
            # HPSS normalizes the spectrogram locally, so it will find "events" even in 
            # silence (AC hum, fan noise, breath all have harmonic content). We must gate
            # on absolute amplitude before even calling HPSS.
            buffer_rms = float(np.sqrt(np.mean(window_audio ** 2, dtype=np.float64) + 1e-12))
            if buffer_rms < self.noise_floor_rms:
                logger.debug(f"[AdaptiveMic] Skipping buffer: RMS {buffer_rms:.5f} < floor {self.noise_floor_rms:.5f}")
                continue

            # Find acoustic ROIs in the spectrogram
            events = self.detector.find_events(window_audio)
            
            # The buffer is 4.0s long. The hop is 2.0s.
            # Only process events whose center falls in [1.0, 3.0) to guarantee centering.
            for (start_t, end_t) in events:
                event_center = (start_t + end_t) / 2.0
                
                if not (1.0 <= event_center < 3.0):
                    continue
                
                # --- GATE 2: Minimum event ROI amplitude ---
                # Verify the actual ROI window has meaningful energy (not just HPSS artifact)
                roi_start = int(start_t * self.sample_rate)
                roi_end = int(end_t * self.sample_rate)
                roi_rms = float(np.sqrt(np.mean(window_audio[roi_start:roi_end] ** 2 + 1e-12)))
                if roi_rms < self.noise_floor_rms * 1.5:
                    continue
                    
                self._process_roi(window_audio, start_t, end_t)


        logger.info("[AdaptiveMic] Ingest worker thread terminated.")

    def _process_roi(self, audio: np.ndarray, start_t: float, end_t: float):
        """Extract a centered 3-second window around the ROI and classify."""
        event_center_t = (start_t + end_t) / 2.0
        
        # We need a 3.0s window for BirdNET (1.5s on each side of the center)
        half_window = 1.5
        win_start_t = max(0.0, event_center_t - half_window)
        win_end_t = min(self.buffer_seconds, win_start_t + 3.0)
        
        # If we hit the end of the buffer, shift start back to get full 3s
        if win_end_t - win_start_t < 3.0:
            win_start_t = max(0.0, win_end_t - 3.0)
            
        start_idx = int(win_start_t * self.sample_rate)
        end_idx = int(win_end_t * self.sample_rate)
        
        clip = audio[start_idx:end_idx]
        
        # Pad with zeros if clip is still too short (e.g. buffer wasn't full)
        target_len = int(3.0 * self.sample_rate)
        if len(clip) < target_len:
            pad_len = target_len - len(clip)
            clip = np.pad(clip, (0, pad_len), "constant")
            
        self._classify_and_log(clip, event_duration=(end_t - start_t))

    def _classify_and_log(self, audio_clip: np.ndarray, event_duration: float):
        """Extract embeddings and apply TOP-1 nearest-neighbour classification with margin guard."""
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            audio_hash = hashlib.sha256(audio_clip.tobytes()).hexdigest()[:16]
            save_path = self.detections_dir / f"roi_{ts_str}.wav"
            
            # Deduplication: skip if we've seen this exact audio clip very recently
            # (the sliding window can pick up the same event across consecutive hops)
            with self._lock:
                if audio_hash in self._recent_hashes:
                    logger.debug(f"[AdaptiveMic] Dedup skip: hash {audio_hash} already processed.")
                    return
                self._recent_hashes.append(audio_hash)
            
            # Embed
            emb = self.backbone.embed(audio_clip, sr=self.sample_rate)
            
            # Predict - top-1 nearest-centroid match
            theta = self.multi_label_threshold
            raw_pred: PredictionResult = self.classifier.predict(emb, threshold=theta)
            
            # TOP-1 + MARGIN GUARD:
            # Only accept the best match if:
            #   1. Its cosine similarity is above the absolute threshold (theta)
            #   2. It beats the second-best by at least MIN_MARGIN (discriminability check)
            # 0.01 margin allows closely related species (e.g., House Crow vs Jungle Crow) to pass.
            MIN_MARGIN = 0.01  # top-1 must be at least 1 percentage point above top-2
            
            sorted_scores = sorted(raw_pred.scores.items(), key=lambda x: x[1], reverse=True)
            best_label, best_score = sorted_scores[0] if sorted_scores else ("Unknown", 0.0)
            second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
            margin = best_score - second_score
            
            is_confident_match = (
                raw_pred.is_known and
                best_score >= theta and
                margin >= MIN_MARGIN
            )

            if not is_confident_match:
                # Quarantined as Unknown
                sf.write(str(save_path), audio_clip, self.sample_rate, subtype="PCM_16", format="WAV")
                self.sound_bank.add(embedding=emb, audio_path=str(save_path), timestamp=now_iso)
                self.db_mgr.save_unidentified(
                    embedding=emb, 
                    best_match=raw_pred.predicted_label, 
                    score=raw_pred.confidence, 
                    audio_path=str(save_path)
                )
                self.db_mgr.log_detection(
                    species_id="Unknown", confidence=raw_pred.confidence, audio_hash=audio_hash, threshold=theta
                )
                
                logger.info(
                    f"[AdaptiveMic] Unknown ROI (best={best_label} {best_score:.3f}, "
                    f"margin={margin:.3f}, dur={event_duration:.2f}s)"
                )
                
                current_event_logs = [{
                    "timestamp": now_iso,
                    "species_id": "Unknown",
                    "confidence": raw_pred.confidence,
                    "threshold": theta,
                    "is_known": False,
                    "common_name": "Unknown Sound",
                    "taxon": "Unknown"
                }]
                
                with self._lock:
                    self._last_detections.extend(current_event_logs)
                return

            # Log single top-1 detection
            sf.write(str(save_path), audio_clip, self.sample_rate, subtype="PCM_16", format="WAV")
            
            self.db_mgr.log_detection(
                species_id=best_label, confidence=best_score, audio_hash=audio_hash, threshold=theta
            )
            
            sp_info = self.db_mgr.get_prototype(best_label)
            common_name = sp_info.get("common_name", "") if sp_info else ""
            taxon = sp_info.get("taxon", "") if sp_info else ""
            
            current_event_logs = [{
                "timestamp": now_iso,
                "species_id": best_label,
                "common_name": common_name,
                "taxon": taxon,
                "confidence": round(best_score, 4),
                "margin": round(margin, 4),
                "is_known": True,
                "audio_path": str(save_path.name),
            }]
            logger.info(f"[AdaptiveMic] Detection: {best_label} (conf={best_score:.3f}, margin={margin:.3f})")

            self._last_detections = current_event_logs
            
        except Exception as ex:
            logger.error(f"[AdaptiveMic] Error during classification: {ex}")

    def start(self, device: Optional[Union[int, str]] = None):
        import sounddevice as sd
        self._is_running = True
        self._audio_buffer.clear()
        self._new_samples_count = 0
        self._last_detections.clear()
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1) # 100ms blocks
        )
        self._stream.start()
        self._worker_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._is_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
