# AnyCall Changelog

All notable changes to the AnyCall project, written by Claude Sonnet.

---

## [Audit Session] -- 2026-09-28

### Fixed -- CRITICAL: Broken Mean-Centering Destroying All Detections
**File:** `anycall/classifier/engine.py`

`PrototypicalClassifier.predict()` applied dynamic mean-centering by default. With 33 cross-taxa species,
the global mean is large enough that subtracting it rotated the embedding geometry into a degenerate subspace,
collapsing self-cosine-similarities from 74-93% down to 39-57% -- all below the 60% acceptance threshold.
Every live audio event was being rejected as Unknown.

Root cause: Mean-centering only works with very few classes (2-5). At 33 species the mean is a large,
meaningful vector and subtracting it ruins the geometry.

Fix: Changed use_mean_centering default from True to False.

Verified score improvement (same file queried against its own prototype):
  corvus_splendens:      57.5% (rejected) -> 83.4% (match)
  acridotheres_tristis:  41.8% (rejected) -> 78.2% (match)
  pavo_cristatus:        39.5% (rejected) -> 74.3% (match)

---

### Fixed -- CRITICAL: Shotgun Multi-Label False Positives
**File:** `anycall/stream/adaptive_listener.py`

_classify_and_log() used a multi-label strategy: it reported every species whose cosine similarity
exceeded the acceptance threshold. Because BirdNET's 1024-d embedding space is crowded, unrelated
species cluster at 0.60-0.75, so a single bird call triggered 10-22 simultaneous false detections.

Example from audit: One parakeet clip triggered Eudynamys, Corvus splendens, Copsychus, Athene brama,
Oecanthus, Gryllus, Funambulus, Cryptotympana, and more -- all at 60-63%.

Fix: Replaced multi-label logic with top-1 nearest-neighbour + margin guard.
  - Only the single best-matching species is reported per audio clip.
  - MIN_MARGIN = 0.05: top-1 must beat runner-up by >= 5 percentage points.
  - If margin is too small the clip is quarantined as Unknown.

Effect on parakeet example:
  Before: 10 species, all at 60-63%
  After:  psittacula_krameri at 81.6%, margin 0.108 -- correct and clean

---

### Fixed -- DB Rebuild Clip Cap Bug
**File:** `scripts/rebuild_roi_db.py`

Outer loop capped at 30 clips per species, but a leftover inner-loop break from speed-testing sessions
broke at 5 clips. All 33 species were built from only 5 ROI clips each, producing weaker centroids.

Fix: Inner loop cap raised from 5 to 30 to match outer loop.

---

### Changed -- Default Acceptance Threshold Raised to 0.70
After disabling mean-centering, correct matches sit at 74-93% and unrelated species cluster at 65-72%.
Raised threshold from 0.60 to 0.70 for a cleaner acceptance margin.

---

### Fixed -- Sliding Window Event Center Filter
**File:** `anycall/stream/adaptive_listener.py`

The inference worker used a simple end_t < overlap_start_sec filter causing chirps near buffer edges
to be extracted in position-shifted windows (call at the far end, not centered), deforming embeddings
vs. the centered database prototypes.

Fix: Only process events whose temporal center falls in [1.0s, 3.0s) of the 4-second buffer,
guaranteeing every extracted window is perfectly centered around the chirp.

---

## [Earlier Sessions -- Feature Branch: feature-spectrogram-roi]

### Added -- HPSS Spectrogram Event Detector
**File:** `anycall/stream/spectrogram.py` (new)

Replaced naive energy-based VAD with SpectrogramEventDetector using:
1. STFT of the audio buffer.
2. Harmonic-Percussive Source Separation (margin=1.2) to isolate bird call streaks.
3. dB Mel spectrogram of harmonic component, normalized to [0, 1].
4. Threshold at energy_threshold=0.20 to create binary activity mask.
5. Morphological closing (3x5 structure) to connect fragmented streaks.
6. Connected component labeling to extract (start_time, end_time) events.
7. Merge events within 0.2 seconds of each other.

Benchmark result: Few-shot accuracy jumped from 55.9% to 96.3%.

### Added -- AdaptiveSpectrogramListener
**File:** `anycall/stream/adaptive_listener.py` (new)

Live microphone ingestion class:
- 4-second rolling deque buffer at 48kHz
- Inference fires every 2-second hop (50% overlap)
- Delegates event detection to SpectrogramEventDetector
- Extracts perfectly centered 3-second window around each detected chirp
- Embeds via BirdNET backbone and classifies via PrototypicalClassifier

### Added -- detections_logged Field in get_status()
**File:** `anycall/stream/adaptive_listener.py`

Frontend JavaScript polls this field every 500ms and hangs forever if missing.

### Changed -- Default energy_threshold Lowered to 0.20
**File:** `anycall/stream/spectrogram.py`

Original default was 0.85, discarding >90% of bird chirps. Lowered to 0.20.

### Added -- DB Re-Enrollment Pipeline
**File:** `scripts/rebuild_roi_db.py` (new)

Script to wipe and rebuild data/processed_roi/ for all 33 curated species using HPSS ROI
extraction -- matching the exact logic used by the live listener and eliminating the distribution
mismatch that caused 24% cosine similarities.

### Fixed -- mkdir exist_ok Crash in Rebuild Script
**File:** `scripts/rebuild_roi_db.py`

Fixed FileExistsError crash by passing exist_ok=True to mkdir() calls.
