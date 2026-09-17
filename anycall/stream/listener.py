"""AnyCall Continuous Microphone Stream Listener.

Captures continuous audio from the system microphone, applies adaptive VAD gating,
extracts embeddings via the active backbone, performs zero-retraining prototypical
classification, and automatically persists detections and quarantined mystery sounds.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from anycall.audio.vad import EnergyVAD
from anycall.classifier.engine import (
    PredictionResult,
    PrototypicalClassifier,
    UnidentifiedSoundBank,
)
from anycall.embeddings.base import BaseAudioEmbeddingBackbone
from anycall.storage.db import DatabaseManager

logger = logging.getLogger("anycall.stream.listener")


class ContinuousMicrophoneListener:
    """Continuous microphone ingestion daemon with adaptive VAD & edge inference."""

    def __init__(
        self,
        classifier: PrototypicalClassifier,
        backbone: BaseAudioEmbeddingBackbone,
        db_mgr: DatabaseManager,
        sound_bank: Optional[UnidentifiedSoundBank] = None,
        sample_rate: int = 48000,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.5,
        energy_threshold: float = 0.005,
        detections_dir: Union[str, Path] = "data/detections",
        rejection_threshold: Optional[float] = None,
        ema_alpha: float = 0.55,
        margin_threshold: float = 0.015,
    ):
        self.classifier = classifier
        self.backbone = backbone
        self.db_mgr = db_mgr
        self.sound_bank = sound_bank if sound_bank is not None else UnidentifiedSoundBank(cluster_similarity=0.70)
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.window_samples = int(round(self.sample_rate * self.window_seconds))
        self.hop_samples = int(round(self.sample_rate * self.hop_seconds))
        self.energy_threshold = float(energy_threshold)
        self.rejection_threshold = rejection_threshold
        self.ema_alpha = float(ema_alpha)
        self.margin_threshold = float(margin_threshold)

        self.detections_dir = Path(detections_dir)
        self.detections_dir.mkdir(parents=True, exist_ok=True)

        self.vad = EnergyVAD(sr=self.sample_rate)

        # Temporal EMA smoothing & Margin gating state
        self._smoothed_scores: Dict[str, float] = {}
        self._last_vocal_time: float = 0.0

        # Threading and buffers
        self._is_running = False
        self._stream = None
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Rolling audio circular buffer
        # Buffer holds up to ~10 seconds of raw audio samples
        max_buffer_samples = self.sample_rate * 10
        self._audio_buffer = deque(maxlen=max_buffer_samples)
        self._new_samples_count = 0

        # Telemetry metrics
        self._total_samples_captured = 0
        self._clips_evaluated = 0
        self._clips_with_activity = 0
        self._detections_logged = 0
        self._current_rms = 0.0
        self._active_device: Optional[Union[int, str]] = None
        self._last_detection: Optional[Dict[str, Any]] = None
        self._start_time: Optional[float] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """Queries available audio input devices via sounddevice."""
        try:
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
            return input_devices
        except Exception as e:
            logger.warning(f"Failed to query sounddevice input devices: {e}")
            return []

    def _audio_callback(self, indata, frames, time_info, status):
        """Low-latency callback from sounddevice InputStream."""
        if status:
            logger.debug(f"Audio stream status flag: {status}")

        if not self._is_running:
            return

        # Downmix to 1D mono float32
        if indata.ndim > 1:
            mono = np.mean(indata, axis=1, dtype=np.float32)
        else:
            mono = indata.flatten().astype(np.float32)

        # Quick RMS calculation for live meter
        rms = float(np.sqrt(np.mean(mono ** 2, dtype=np.float64) + 1e-12))
        self._current_rms = rms

        with self._lock:
            self._audio_buffer.extend(mono.tolist())
            self._new_samples_count += len(mono)
            self._total_samples_captured += len(mono)

    def _inference_worker_loop(self):
        """Background worker thread continuously evaluating incoming audio chunks."""
        logger.info("[ContinuousMic] Ingest worker thread started.")

        while self._is_running:
            # Check if we have gathered enough new samples for the next hop
            has_enough = False
            with self._lock:
                if len(self._audio_buffer) >= self.window_samples and self._new_samples_count >= self.hop_samples:
                    # Extract the latest window
                    buf_list = list(self._audio_buffer)
                    window_audio = np.array(buf_list[-self.window_samples:], dtype=np.float32)
                    self._new_samples_count = 0
                    has_enough = True

            if not has_enough:
                time.sleep(0.05)
                continue

            # Process window
            self._clips_evaluated += 1
            peak_amp = float(np.max(np.abs(window_audio)))

            # Fast energy gating
            if peak_amp < self.energy_threshold:
                continue

            # Adaptive VAD check
            is_vocal = self.vad.is_speech_or_vocal(window_audio)
            if not is_vocal:
                continue

            self._clips_with_activity += 1
            self._process_active_vocalization(window_audio)

        logger.info("[ContinuousMic] Ingest worker thread terminated.")

    def _process_active_vocalization(self, audio_clip: np.ndarray):
        """Performs embedding extraction, EMA temporal smoothing, margin gating, and logging."""
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            save_path = self.detections_dir / f"live_mic_{ts_str}.wav"

            # Save 16-bit PCM WAV recording
            sf.write(str(save_path), audio_clip, self.sample_rate, subtype="PCM_16", format="WAV")

            # Extract embedding using active backbone
            emb = self.backbone.embed(audio_clip, sr=self.sample_rate)

            # Determine rejection threshold theta
            theta = (
                self.rejection_threshold
                if self.rejection_threshold is not None
                else self.classifier.threshold
            )

            raw_pred: PredictionResult = self.classifier.predict(emb, threshold=theta)
            audio_hash = hashlib.sha256(audio_clip.tobytes()).hexdigest()[:16]

            # Temporal EMA Smoothing across stream hops
            current_time = time.time()
            if current_time - self._last_vocal_time > 3.5:
                # Reset smoothed scores if silence/gap between vocalizations exceeds 3.5s
                self._smoothed_scores.clear()
            self._last_vocal_time = current_time

            for sp, score in raw_pred.scores.items():
                if sp in self._smoothed_scores:
                    self._smoothed_scores[sp] = (
                        self.ema_alpha * score + (1.0 - self.ema_alpha) * self._smoothed_scores[sp]
                    )
                else:
                    self._smoothed_scores[sp] = score

            # Sort candidate species by smoothed scores
            sorted_candidates = sorted(
                self._smoothed_scores.items(), key=lambda item: item[1], reverse=True
            )

            best_candidate_hint = "Unknown"
            if sorted_candidates:
                top1_sp, top1_score = sorted_candidates[0]
                top2_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
                margin = top1_score - top2_score
                best_candidate_hint = top1_sp

                # Gating: Must meet both absolute threshold theta AND minimum top-1 separation margin
                if top1_score >= theta and margin >= self.margin_threshold:
                    predicted_label = top1_sp
                    confidence = float(top1_score)
                    is_known = True
                else:
                    predicted_label = "Unknown"
                    confidence = float(top1_score)
                    is_known = False
            else:
                predicted_label = "Unknown"
                confidence = 0.0
                is_known = False

            # Log to SQLite detections table
            self.db_mgr.log_detection(
                species_id=predicted_label,
                confidence=confidence,
                audio_hash=audio_hash,
                threshold=theta,
            )
            self._detections_logged += 1

            # If rejected (< theta or insufficient margin), quarantine to UnidentifiedSoundBank
            if not is_known:
                self.sound_bank.add(
                    embedding=emb,
                    audio_path=str(save_path),
                    timestamp=now_iso,
                )
                self.db_mgr.save_unidentified(
                    embedding=emb,
                    best_match=best_candidate_hint,
                    score=confidence,
                    audio_path=str(save_path),
                )

            # Update latest detection telemetry
            top_candidates = sorted_candidates[:5] if sorted_candidates else []

            detection_event = {
                "timestamp": now_iso,
                "species_id": predicted_label,
                "confidence": round(confidence, 4),
                "is_known": is_known,
                "audio_path": str(save_path.name),
                "threshold": theta,
                "candidates": [
                    {"species": k, "score": round(v, 4)} for k, v in top_candidates
                ],
            }
            self._last_detection = detection_event

            status_str = "KNOWN" if is_known else "UNKNOWN (QUARANTINED)"
            logger.info(
                f"[ContinuousMic] Event detected: {predicted_label} "
                f"(conf: {confidence:.3f}, status: {status_str}, margin: {(margin if sorted_candidates else 0.0):.4f})"
            )

        except Exception as ex:
            logger.error(f"[ContinuousMic] Error processing vocalization: {ex}", exc_info=True)

    def start(self, device: Optional[Union[int, str]] = None):
        """Starts continuous microphone listening."""
        if self._is_running:
            logger.warning("[ContinuousMic] Listener is already running.")
            return

        import sounddevice as sd

        self._active_device = device
        self._is_running = True
        self._start_time = time.time()
        self._audio_buffer.clear()
        self._new_samples_count = 0
        self._smoothed_scores.clear()
        self._last_vocal_time = 0.0

        # Launch background audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            device=device,
            blocksize=int(self.sample_rate * 0.25),  # 250ms chunks
        )
        self._stream.start()

        # Launch inference worker thread
        self._worker_thread = threading.Thread(
            target=self._inference_worker_loop,
            daemon=True,
            name="AnyCallMicWorker",
        )
        self._worker_thread.start()

        logger.info(f"[ContinuousMic] Started continuous listening on device={device} (sr={self.sample_rate}Hz).")

    def stop(self):
        """Stops continuous microphone listening."""
        if not self._is_running:
            return

        self._is_running = False
        self._smoothed_scores.clear()
        self._last_vocal_time = 0.0

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing sounddevice stream: {e}")
            self._stream = None

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        logger.info("[ContinuousMic] Stopped continuous listening.")

    def get_status(self) -> Dict[str, Any]:
        """Returns runtime status and telemetry metrics."""
        uptime = (time.time() - self._start_time) if (self._is_running and self._start_time) else 0.0
        return {
            "is_running": self._is_running,
            "uptime_seconds": round(uptime, 1),
            "active_device": self._active_device,
            "current_rms": round(self._current_rms, 6),
            "samples_captured": self._total_samples_captured,
            "clips_evaluated": self._clips_evaluated,
            "clips_with_activity": self._clips_with_activity,
            "detections_logged": self._detections_logged,
            "last_detection": self._last_detection,
            "rejection_threshold": (
                self.rejection_threshold
                if self.rejection_threshold is not None
                else self.classifier.threshold
            ),
        }


def main():
    """CLI runner for continuous microphone monitoring."""
    import argparse
    from anycall.embeddings import get_backbone

    parser = argparse.ArgumentParser(description="AnyCall Continuous Microphone Stream Listener")
    parser.add_argument("--db", type=str, default="anycall.db", help="Path to SQLite database")
    parser.add_argument("--backbone", type=str, default="birdnet", help="Embedding backbone (birdnet, perch, panns, mock)")
    parser.add_argument("--threshold", type=float, default=0.72, help="Rejection threshold theta (default: 0.72)")
    parser.add_argument("--device", type=int, default=None, help="Input device index (optional)")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        devices = ContinuousMicrophoneListener.list_input_devices()
        print("Available Audio Input Devices:")
        for d in devices:
            default_marker = " [DEFAULT]" if d["is_default"] else ""
            print(f"  [{d['index']}] {d['name']} ({d['channels']} ch, {d['default_samplerate']} Hz){default_marker}")
        return

    print("=" * 72)
    print("           AnyCall Continuous Wildlife Microphone Streamer          ")
    print("=" * 72)
    print(f"  • Database          : {args.db}")
    print(f"  • Backbone          : {args.backbone}")
    print(f"  • Rejection Theta   : {args.threshold:.2f}")
    print(f"  • Device Index      : {args.device if args.device is not None else 'Default'}")
    print("=" * 72)

    db_mgr = DatabaseManager(args.db)
    backbone = get_backbone(args.backbone)
    classifier = PrototypicalClassifier(threshold=args.threshold)
    sound_bank = UnidentifiedSoundBank(cluster_similarity=0.70)

    # Load existing species prototypes
    try:
        existing_rows = db_mgr._conn.execute("SELECT * FROM species").fetchall()
        for r in existing_rows:
            sp_id = r["species_id"]
            proto_bytes = r["prototype"]
            vec = np.frombuffer(proto_bytes, dtype=np.float32)
            classifier.enroll(sp_id, [vec])
        print(f"[ContinuousMic] Loaded {len(existing_rows)} enrolled species prototypes from {args.db}")
    except Exception as e:
        print(f"[ContinuousMic] Note: No existing species prototypes: {e}")

    listener = ContinuousMicrophoneListener(
        classifier=classifier,
        backbone=backbone,
        db_mgr=db_mgr,
        sound_bank=sound_bank,
        rejection_threshold=args.threshold,
    )

    listener.start(device=args.device)
    print("Listening on microphone... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(2.0)
            status = listener.get_status()
            last = status.get("last_detection")
            last_msg = (
                f"Last: {last['species_id']} ({last['confidence']:.2f})"
                if last
                else "No detections yet"
            )
            print(
                f"\r[Uptime: {status['uptime_seconds']}s | "
                f"RMS: {status['current_rms']:.4f} | "
                f"Vocal events: {status['clips_with_activity']} | "
                f"{last_msg}]",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopping microphone listener...")
    finally:
        listener.stop()
        print("Listener stopped cleanly.")


if __name__ == "__main__":
    main()
