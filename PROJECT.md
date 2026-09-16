# Project: AnyCall

## Architecture
AnyCall is an open-set, few-shot, cross-taxa acoustic wildlife classifier designed for edge deployment without retraining. Field researchers can enroll any species (birds, insects, amphibians, mammals) using 5–20 audio recordings and perform real-time local classification on edge hardware (Raspberry Pi Zero 2W or laptop).

### Data Flow
1. **Audio Ingestion & Preprocessing (`anycall.audio`, `anycall.data`)**:
   - Ingests raw audio from local files, microphone streams, or Xeno-Canto downloads.
   - Standardizes audio to 48,000 Hz, 16-bit mono PCM WAV.
   - Applies an adaptive energy VAD (50ms RMS frames, $\alpha=0.01$ tracking, trigger at $\text{RMS} > 3.0 \times \text{floor}$ for $\ge 300\text{ ms}$).
   - Extracts centered, non-silent 3.0-second segments (144,000 samples at 48kHz).
2. **Unified Embedding Extraction (`anycall.embeddings`)**:
   - Standardized contract `BaseAudioEmbeddingBackbone.embed(audio_array_or_path) -> np.ndarray [D]`.
   - L2-normalization enforced on all outputs ($\|f(x)\|_2 = 1.0$).
   - Three candidate backbones:
     - **BirdNET**: Native 48 kHz / 3.0s input. EfficientNet-B0 backbone extracting 1024-d (or 320-d penultimate) embeddings.
     - **Google Perch**: Internal polyphase resampling from 48 kHz to 32 kHz (96,000 samples) and zero-padding to 5.0s (160,000 samples). Extracts 1280-d (or 1536-d) linearly separable embeddings.
     - **PANNs**: Internal polyphase resampling to 32 kHz (96,000 samples). AudioTagging backbone extracting 2048-d (CNN14) or 1280-d (MobileNetV2) embeddings.
     - **MockBackbone**: Fast, deterministic, orthogonal vector generator for millisecond-level offline unit and E2E testing.
3. **Prototypical Classification Engine (`anycall.classifier`, `anycall.storage`)**:
   - Computes class prototype centroids: $c_k = \frac{1}{|S_k|} \sum_{x \in S_k} f(x)$, normalized to unit length $\hat{c}_k = \frac{c_k}{\|c_k\|_2}$.
   - Classification via cosine similarity: $s(q, c_k) = \langle q, \hat{c}_k \rangle$.
   - Open-set rejection threshold $\theta \in [0.0, 1.0]$: If $\max_k s(q, c_k) < \theta$, predict `"Unknown"`.
   - Online incremental update: $c_k^{(N+M)} = \text{normalize}\left(\frac{N \cdot c_k^{(N)} + \sum_{j=1}^M f(x_j)}{N + M}\right)$ without re-evaluating historical exemplars.
   - Sub-prototype clustering ($K$-Means) activated when intra-class variance $> 0.05$ to handle multi-call repertoires.
   - SQLite persistent storage for enrolled species prototypes (`species` table) and detection history (`detections` table).
4. **Benchmarking Suite (`anycall.benchmark`)**:
   - Automated runner executing 5 rigorous experiments:
     - Exp 1: Stock BirdNET Failure Quantification (evaluating stock BirdNET closed-set softmax on non-avian and Indian taxa).
     - Exp 2: Few-Shot Accuracy Curves ($K \in \{5, 10, 20\}$ shots across all 3 backbones with 5-fold cross-validation).
     - Exp 3: Cross-Taxa Generalization Breakdown (Aves, Insecta, Amphibia, Mammalia + Cosine Separability Ratio).
     - Exp 4: Open-Set Rejection Quality (ROC curve, FAR vs TAR sweep over $\theta \in [0.0, 1.0]$, EER, AUROC).
     - Exp 5: Edge Deployment Profiling (inference latency per 3s segment, RAM footprint).
   - Exports quantitative results to CSV and JSON formats.
5. **Dual Track Strategy**:
   - **Implementation Track**: Builds modules M1 through M4, culminating in M5 (passing 100% of E2E tests + adversarial coverage hardening).
   - **E2E Testing Track**: Independently designs opaque-box requirement-driven test suite (Tiers 1–4) and publishes `TEST_READY.md`.

---

## Feature Inventory
Every feature identified during the Phase 0 Survey is mapped to its assigned milestone below. No feature is left unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Xeno-Canto API Harvester | Programmatic querying of species recordings, pagination, rate-limiting, and error handling | M1 | Survey R1 |
| 2 | Dual-Mode Download & Fallback | API v3 authentication (`&key=...`) with unauthenticated direct MP3 download fallback (`/<id>/download`) | M1 | Survey R1 |
| 3 | Indian Wildlife Species Directory | Curated catalog of 33 target species across Aves (15), Insecta (8), Amphibia (5), and Mammalia (5) | M1 | Survey R1 |
| 4 | Audio Format Standardization | Conversion of arbitrary audio inputs to 48 kHz, 16-bit mono PCM WAV | M1 | Survey R1 |
| 5 | Uniform Audio Slicing | Splitting standardized audio into uniform 3.0-second (144,000 sample) segments | M1 | Survey R1 |
| 6 | RMS Energy VAD & Noise Floor | 50ms RMS frame energy filter with adaptive noise floor tracking ($\alpha=0.01$) to discard silent segments | M1 | Survey R1 |
| 7 | Unified Backbone Interface | `BaseAudioEmbeddingBackbone` abstract contract returning L2-normalized 1D feature vectors | M2 | Survey R2 |
| 8 | BirdNET Backbone Wrapper | EfficientNet-B0 embedding extractor (48 kHz, 3.0s input, 1024-d / 320-d L2-normalized output) | M2 | Survey R2 |
| 9 | Google Perch Backbone Wrapper | Bioacoustic EfficientNet extractor (polyphase resample to 32 kHz, 5.0s zero-pad, 1280-d output) | M2 | Survey R2 |
| 10 | PANNs Backbone Wrapper | Pretrained Audio Neural Network extractor (resample to 32 kHz, 3.0s input, 2048-d / 1280-d output) | M2 | Survey R2 |
| 11 | Fast Mock Backbone Fixture | Deterministic, orthogonal synthetic embedding generator for instantaneous (<5s) offline test suites | M2 | Survey R2 |
| 12 | Centroid Prototype Computation | Mathematical centroid calculation $c_k = \frac{1}{\|S_k\|}\sum f(x)$ with unit L2 normalization | M3 | Survey R3 |
| 13 | Cosine Similarity Classifier | Inner product ranking on unit hypersphere for top-1 species prediction | M3 | Survey R3 |
| 14 | Open-Set Rejection Engine | Tunable rejection threshold $\theta \in [0.0, 1.0]$ rejecting query as `"Unknown"` if $\max s(q, c_k) < \theta$ | M3 | Survey R3 |
| 15 | Online Incremental Updater | Centroid update formula updating prototypes with new samples without batch re-computation | M3 | Survey R3 |
| 16 | Multi-Call Sub-Prototype Clustering | $K$-Means sub-prototype generator when intra-class cosine variance $> 0.05$ | M3 | Survey R3 |
| 17 | SQLite Species Prototype Store | SQLite `species` table storing serialized BLOB prototypes, radii, taxon, and metadata | M3 | Survey R3 |
| 18 | SQLite Detection Event Logger | SQLite `detections` table logging detection timestamp, species, confidence, and audio hash | M3 | Survey R3 |
| 19 | Experiment 1: BirdNET Failure Analysis | Quantitative evaluation of stock BirdNET softmax vs ground truth on Indian birds & non-avian taxa | M4 | Survey R4 |
| 20 | Experiment 2: Few-Shot Accuracy Curves | 5-fold cross-validated top-1 accuracy curves at 5, 10, and 20 shots across all 3 backbones | M4 | Survey R4 |
| 21 | Experiment 3: Cross-Taxa Generalization | Accuracy and Cosine Separability Ratio broken down across Birds, Insects, Frogs, and Mammals | M4 | Survey R4 |
| 22 | Experiment 4: Rejection Quality ROC | False Acceptance Rate (FAR) vs True Acceptance Rate (TAR) curve, EER, and AUROC over $\theta$ sweeps | M4 | Survey R4 |
| 23 | Experiment 5: Edge Latency/RAM Profiling | On-device profiling measuring inference latency per 3s chunk and memory footprint | M4 | Survey R4 |
| 24 | Benchmark Report Generator | Automatic formatting and export of benchmark results to JSON and CSV reports | M4 | Survey R4 |
| 25 | 4-Tier Decoupled E2E Test Suite | Opaque-box test suite spanning Feature (T1), Boundary (T2), Cross-Feature (T3), and Real-World (T4) | M5 | Survey Dual Track |
| 26 | Adversarial Coverage Hardening | Tier 5 white-box stress testing, corner cases, and forensic integrity verification | M5 | Survey Dual Track |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Pipeline & Audio Standardization | `anycall.audio`, `anycall.data`, species catalog, Xeno-Canto harvester v3 + direct fallback, 48kHz WAV conversion, 3s windowing, RMS VAD filtering, sample *Corvus splendens* download | None | DONE |
| M2 | Unified Embedding Extraction Backbones | `anycall.embeddings`, BaseAudioEmbeddingBackbone, BirdNET wrapper, Perch wrapper, PANNs wrapper, MockBackbone, sample 3s WAV L2-norm extraction verification | M1 | PLANNED |
| M3 | Prototypical Classification Engine & Storage | `anycall.classifier`, `anycall.storage`, Centroid math, cosine similarity, open-set $\theta$ rejection, online incremental updates, sub-prototype clustering, SQLite database schemas | M2 | PLANNED |
| M4 | Automated Benchmarking Suite | `anycall.benchmark`, Automated 5-experiment runner (Exp 1-5), quantitative JSON & CSV reporting | M1, M2, M3 | PLANNED |
| M5 | E2E Test Suite Pass (100%) & Coverage Hardening | Pass 100% of Tiers 1-4 tests published in `TEST_READY.md`, followed by Tier 5 adversarial stress testing and forensic audit verification | M1, M2, M3, M4, TEST_READY.md | PLANNED |

In parallel:
- **E2E Testing Track**: Requirement-driven opaque-box test harness (`tests/e2e/`), synthetic audio generators, test fixtures, Tiers 1-4 test implementation, publishing `TEST_INFRA.md` and `TEST_READY.md`.

---

## Interface Contracts

### 1. `anycall.audio` ↔ `anycall.data`
- **Function**: `standardize_audio(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, target_sr: int = 48000) -> Tuple[np.ndarray, int]`
  - Converts input audio (MP3/WAV) to mono 48,000 Hz 16-bit PCM float32 array normalized to $[-1.0, 1.0]$.
  - Raises `AudioFormatError` if file is unreadable or corrupted.
- **Function**: `slice_audio_segments(audio: np.ndarray, sr: int = 48000, segment_duration: float = 3.0, hop_duration: float = 3.0, vad_filter: bool = True) -> List[np.ndarray]`
  - Returns a list of 1D float32 numpy arrays of shape `(144000,)`.
  - Discards segments that do not meet the RMS energy VAD threshold ($\text{RMS} > 3.0 \times \text{noise\_floor}$).

### 2. `anycall.embeddings` Interface
- **Abstract Class**: `BaseAudioEmbeddingBackbone(ABC)`
  - **Method**: `embed(audio: Union[str, Path, np.ndarray], sr: int = 48000) -> np.ndarray`
    - Accepts a 3.0-second audio segment (or file path).
    - Returns a 1D numpy array `embedding` of shape `(D,)`, dtype `float32`.
    - **Contract**: Vector MUST be L2-normalized: `abs(np.linalg.norm(embedding) - 1.0) < 1e-5`.
  - **Property**: `embedding_dim -> int` (BirdNET: 1024 or 320; Perch: 1280; PANNs: 2048; Mock: 256).
  - **Property**: `name -> str`

### 3. `anycall.classifier` Interface
- **Class**: `PrototypicalClassifier`
  - **Method**: `compute_prototype(embeddings: Sequence[np.ndarray]) -> np.ndarray`
    - Computes centroid $c_k = \frac{1}{K}\sum e_i$, returns unit L2-normalized vector.
  - **Method**: `enroll_species(species_id: str, common_name: str, taxon: str, embeddings: Sequence[np.ndarray]) -> None`
  - **Method**: `predict(query_embedding: np.ndarray, threshold: float = 0.70) -> PredictionResult`
    - `PredictionResult(predicted_label: str, confidence: float, is_known: bool, top_k_similarities: Dict[str, float])`
    - If `confidence < threshold`, `predicted_label = "Unknown"` and `is_known = False`.
  - **Method**: `update_prototype(species_id: str, new_embeddings: Sequence[np.ndarray]) -> np.ndarray`
    - Implements online incremental formula: $c_k^{(N+M)} = \text{normalize}\left(\frac{N c_k^{(N)} + \sum e_j}{N + M}\right)$.

### 4. `anycall.storage` Interface
- **Class**: `DatabaseManager`
  - SQLite schema with tables `species` (id, common_name, taxon, prototype_blob, radius, sample_count, updated_at) and `detections` (id, timestamp, species_id, confidence, audio_hash, threshold).
  - Methods: `save_prototype`, `get_prototype`, `list_species`, `log_detection`.

### 5. `anycall.benchmark` Interface
- **Class**: `BenchmarkRunner`
  - **Method**: `run_experiment_1_birdnet_failure(...) -> Dict`
  - **Method**: `run_experiment_2_few_shot_curves(shots=[5, 10, 20], folds=5) -> Dict`
  - **Method**: `run_experiment_3_cross_taxa(...) -> Dict`
  - **Method**: `run_experiment_4_rejection_roc(theta_range=np.linspace(0, 1, 101)) -> Dict`
  - **Method**: `run_experiment_5_latency_profiling(...) -> Dict`
  - **Method**: `export_reports(output_dir: Path) -> Tuple[Path, Path]` (generates `benchmark_report.json` and `benchmark_report.csv`).

---

## Code Layout & Ownership Boundaries
The project lives entirely in `C:\Users\kahaa\teamwork_projects\anycall`:

```
anycall/
├── anycall/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── standardize.py     # Owned by M1 Worker
│   │   └── vad.py             # Owned by M1 Worker
│   ├── data/
│   │   ├── __init__.py
│   │   ├── species.py         # Owned by M1 Worker (33 Indian species)
│   │   └── harvester.py       # Owned by M1 Worker (Xeno-Canto API v3 & fallback)
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py            # Owned by M2 Worker
│   │   ├── birdnet.py         # Owned by M2 Worker
│   │   ├── perch.py           # Owned by M2 Worker
│   │   ├── panns.py           # Owned by M2 Worker
│   │   └── mock.py            # Owned by M2 Worker (Deterministic test backbones)
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── engine.py          # Owned by M3 Worker
│   │   ├── incremental.py     # Owned by M3 Worker
│   │   └── clustering.py      # Owned by M3 Worker
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py              # Owned by M3 Worker
│   └── benchmark/
│       ├── __init__.py
│       ├── runner.py          # Owned by M4 Worker
│       ├── experiments.py     # Owned by M4 Worker
│       └── report.py          # Owned by M4 Worker
├── tests/
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── synth_audio.py     # Owned by E2E Test Writer
│   │   └── mock_data.py       # Owned by E2E Test Writer
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── test_tier1_features.py       # Owned by E2E Test Writer
│   │   ├── test_tier2_boundaries.py     # Owned by E2E Test Writer
│   │   ├── test_tier3_cross_feature.py   # Owned by E2E Test Writer
│   │   └── test_tier4_real_world.py     # Owned by E2E Test Writer
│   └── run_e2e_tests.py                 # Owned by E2E Test Writer
├── pyproject.toml
├── README.md
└── PROJECT.md
```

Write ownership rule: Workers on concurrent milestones MUST NOT edit the same files. File ownership is strictly delineated above.
