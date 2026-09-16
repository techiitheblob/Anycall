# Original User Request

## 2026-09-16T08:00:29Z

Build AnyCall: an open-set, few-shot, cross-taxa acoustic wildlife classifier. It uses frozen embedding backbones (BirdNET, Google Perch, PANNs) and a nearest-centroid prototypical classifier to identify species without retraining, running locally on edge hardware.

Working directory: ~/teamwork_projects/anycall
Integrity mode: development

Reference material: C:\Users\kahaa\.gemini\antigravity\brain\0cb25917-3506-4428-ad71-ef8c70267949\handoff_gemini_pro.md

## Requirements

### R1. Data Pipeline (Xeno-Canto)
Build a script to programmatically download 20-40 recordings per species for a target list of 30-40 Indian species across taxa (birds, insects, amphibians, mammals) using the Xeno-Canto API. The pipeline must convert audio to a standardized format (e.g., 48kHz WAV) and extract 3-second segments.

### R2. Embedding Extraction
Implement a unified interface to extract embeddings from three candidate backbones: BirdNET (EfficientNet-B0), Google Perch (EfficientNet variant), and PANNs (MobileNetV2 or CNN14). The interface must accept a 3-second audio segment and return an L2-normalized feature vector.

### R3. Prototypical Classification Engine
Implement a few-shot classifier that computes species prototypes (centroids) from enrollment embeddings and classifies query embeddings using cosine similarity. It must support a tunable rejection threshold (θ) for Unknown sounds and allow incremental prototype updates.

### R4. Benchmarking Suite
Develop an automated benchmarking script that runs five experiments: BirdNET failure analysis (comparing stock BirdNET against ground truth), few-shot accuracy curves (5, 10, 20 shots) for all three backbones, cross-taxa generalization breakdown, and rejection quality (ROC curve for false acceptance rate).

## Acceptance Criteria

### Data Pipeline
- [ ] Script successfully downloads and processes at least 20 recordings for a sample species (e.g., *Corvus splendens*) from Xeno-Canto.
- [ ] Processed output consists of uniform 3-second, 48kHz mono WAV files.

### Embedding Extraction
- [ ] The pipeline can successfully extract an L2-normalized embedding vector from a sample 3-second WAV file using each of the three backbones (BirdNET, Perch, PANNs) locally.

### Classification Engine
- [ ] The engine correctly computes a centroid prototype given a list of embedding vectors.
- [ ] Given a query embedding, the engine returns the correct species label from the bank if cosine similarity > θ, and Unknown if similarity < θ.

### Benchmarking
- [ ] The benchmark script outputs a quantitative report (e.g., CSV or JSON) comparing 5-shot, 10-shot, and 20-shot top-1 accuracy for the three backbones on a provided subset of the test data.

## 2026-09-16T14:03:12Z

Continue building AnyCall: an open-set, few-shot, cross-taxa acoustic wildlife classifier. Milestone 1 (Data Pipeline) is ALREADY COMPLETE and must not be re-implemented.

Working directory: C:\Users\kahaa\teamwork_projects\anycall
Integrity mode: development

Reference material (read before starting): `C:\Users\kahaa\teamwork_projects\anycall\PROJECT.md` and `C:\Users\kahaa\.gemini\antigravity\brain\0cb25917-3506-4428-ad71-ef8c70267949\handoff_gemini_pro.md`

## Context: What Is Already Done

- Python 3.11 virtual environment at `.venv/` is provisioned with audio/DSP libraries.
- `anycall/audio/` — `standardize.py` (48kHz mono conversion) and `vad.py` (energy-based VAD) are complete.
- `anycall/data/` — `species.py` (33 Indian species across Aves/Insecta/Amphibia/Mammalia) and `harvester.py` (Xeno-Canto dual-mode harvester) are complete.
- `data/raw/corvus_splendens/` contains 25 real Xeno-Canto MP3 recordings.
- `data/processed/corvus_splendens/` contains 74 verified 3-second 48kHz mono WAV files.
- `tests/` has Tier 1 and Tier 2 E2E test suites and `run_e2e_tests.py`.
- `TEST_INFRA.md` and `PROJECT.md` are on disk.
- Milestone 1 verification gate passed unanimously (all 49 unit + E2E tests green, forensic audio audit clean).

## Requirements: What Remains to Be Built

### R1. Embedding Extraction (Milestone 2)
Implement `anycall/embeddings/` with a unified abstract interface (`BaseAudioEmbeddingBackbone`) and concrete wrappers for three candidate backbones: BirdNET (EfficientNet-B0, 48kHz/3s input), Google Perch (32kHz/5s input), and PANNs CNN14 or MobileNetV2 (32kHz/3s input). Each wrapper must accept a 3-second audio segment and return an L2-normalized float32 feature vector. Also include a deterministic `MockBackbone` for fast offline testing.

### R2. Prototypical Classification Engine (Milestone 3)
Implement `anycall/classifier/` and `anycall/storage/` — a few-shot classifier that computes species prototypes (centroids of enrollment embeddings), classifies query embeddings via cosine similarity, supports a tunable rejection threshold θ for "Unknown" sounds, supports online incremental prototype updates, and persists species prototypes and detection events to SQLite.

### R3. Automated Benchmarking Suite (Milestone 4)
Implement `anycall/benchmark/` — an automated runner that executes five experiments using the downloaded Xeno-Canto data and all three embedding backbones:
1. BirdNET failure analysis (stock BirdNET softmax accuracy on Indian species vs. AnyCall prototype approach)
2. Few-shot accuracy curves at 5, 10, and 20 shots for all three backbones (5-fold cross-validated)
3. Cross-taxa generalization breakdown (accuracy per taxon: Aves, Insecta, Amphibia, Mammalia)
4. Rejection quality ROC (FAR vs TAR sweeping θ, EER and AUROC reported)
5. Edge latency and RAM profiling (inference time per 3-second segment)

Export results to both `benchmark_report.json` and `benchmark_report.csv`.

### R4. E2E Test Suite Completion (Milestone 5)
Extend the existing test suite with Tier 3 (cross-feature integration) and Tier 4 (real-world end-to-end) tests. All E2E tests (Tiers 1–4) must pass at 100%.

## Acceptance Criteria

### Embedding Extraction
- [ ] Each of the three real backbones (BirdNET, Perch, PANNs) successfully extracts an L2-normalized embedding from one of the real *Corvus splendens* WAV files in `data/processed/corvus_splendens/`.
- [ ] `abs(np.linalg.norm(embedding) - 1.0) < 1e-5` holds for all extracted embeddings.

### Classification Engine
- [ ] Given enrollment embeddings for two distinct species and a query embedding from one of those species, `predict()` returns the correct species label with cosine similarity > θ.
- [ ] Given a query embedding from a species NOT in the bank, `predict()` returns `"Unknown"` when cosine similarity < θ.
- [ ] Incremental prototype update formula produces a unit-normalized vector matching the expected centroid.

### Benchmarking
- [ ] The benchmark runner executes all five experiments without crashing and produces `benchmark_report.json` and `benchmark_report.csv`.
- [ ] The few-shot accuracy table includes columns for backbone name, K (shots), fold, and top-1 accuracy.

### Full E2E
- [ ] Running `python tests/run_e2e_tests.py` passes 100% of all Tier 1–4 tests.

## 2026-09-16T14:45:58Z

CRITICAL INSTRUCTION FROM USER: Immediately after clearing the Milestone 2 (M2) Verification Gate, you MUST PAUSE execution. Do not start Milestone 3 exploration or dispatch any M3 workers. Halt your execution loop and wait for a "RESUME" message from me before continuing to M3.

## 2026-09-16T14:50:32Z

Continue building AnyCall: an open-set, few-shot, cross-taxa acoustic wildlife classifier.
Milestones 1 and 2 are ALREADY COMPLETE.

Working directory: C:\Users\kahaa\teamwork_projects\anycall
Integrity mode: development

**CRITICAL INSTRUCTION: The user is low on credits. Be extremely efficient. Skip adversarial E2E Tier 4 stress tests and extensive forensic audit swarms. Focus strictly on functional implementation and basic unit testing. Do not spawn unnecessary explorer/auditor agents if a single worker can accomplish the task.**

## Context: What Is Already Done
- M1 Data Pipeline (`anycall/audio/`, `anycall/data/`) is fully implemented and tested.
- M2 Embedding Extraction (`anycall/embeddings/`) is fully implemented with wrappers for BirdNET, Perch, PANNs, and Mock.
- `PROJECT.md` and `TEST_INFRA.md` are on disk.

## Requirements: What Remains to Be Built

### R1. Prototypical Classification Engine (Milestone 3)
Implement `anycall/classifier/` and `anycall/storage/` — a few-shot classifier that computes species prototypes (centroids of enrollment embeddings), classifies query embeddings via cosine similarity, supports a tunable rejection threshold θ for "Unknown" sounds, supports online incremental prototype updates, and persists species prototypes and detection events to SQLite.

### R2. Automated Benchmarking Suite (Milestone 4)
Implement `anycall/benchmark/` — an automated runner that executes five experiments using the downloaded Xeno-Canto data and all three embedding backbones:
1. BirdNET failure analysis
2. Few-shot accuracy curves at 5, 10, and 20 shots
3. Cross-taxa generalization breakdown
4. Rejection quality ROC
5. Edge latency and RAM profiling
Export results to both `benchmark_report.json` and `benchmark_report.csv`.

## Acceptance Criteria

### Classification Engine
- [ ] Given enrollment embeddings for two distinct species and a query embedding from one of those species, `predict()` returns the correct species label with cosine similarity > θ.
- [ ] Given a query embedding from a species NOT in the bank, `predict()` returns `"Unknown"` when cosine similarity < θ.
- [ ] Incremental prototype update formula produces a unit-normalized vector matching the expected centroid.

### Benchmarking
- [ ] The benchmark runner executes all five experiments without crashing and produces `benchmark_report.json` and `benchmark_report.csv`.
- [ ] The few-shot accuracy table includes columns for backbone name, K (shots), fold, and top-1 accuracy.

## 2026-09-16T14:51:56Z

UPDATE FROM USER REGARDING EFFICIENCY MODE: Do not skip the verification tests entirely. Instead, scale them down. Perform a quick check on the outcomes of the M1 and M2 forensic and stress tests. If they uncovered actual weaknesses or bugs, continue doing scaled-down verification tests for M3 and M4. If the M1/M2 extreme tests found nothing and were not useful, then you may skip the heavy adversarial tests entirely. Balance efficiency with functional safety.

## 2026-09-16T14:59:27Z

UPDATE FROM USER REGARDING EFFICIENCY MODE: Do not compromise on implementation quality to save credits. If using multiple agents (e.g., parallel explorers or specialized workers) will improve the quality of the classification engine or the benchmarking suite, you MUST use them. Do not restrict yourself to a single worker if it risks harming the final product. You may continue to scale down the *post-implementation adversarial verification tests*, but do not compromise on the *implementation* phase.
