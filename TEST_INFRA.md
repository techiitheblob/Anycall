# AnyCall: E2E Testing Infrastructure & Verification Specification

## 1. Overview & Architecture

AnyCall is an open-set, few-shot, cross-taxa acoustic wildlife classifier designed for autonomous on-edge execution. The testing harness follows the **Project Pattern Dual Track**:
- **Implementation Track**: Builds modules `anycall.audio`, `anycall.data`, `anycall.embeddings`, `anycall.classifier`, `anycall.storage`, and `anycall.benchmark`.
- **E2E Testing Track**: Independently designs and enforces opaque-box, requirement-driven test suites across 4 progressive tiers, backed by deterministic synthetic audio generators and test fixtures.

This decoupled architecture allows the test suite to run **100% offline**, with **zero external audio dependencies**, executing within **< 5.0 seconds** on standard commodity hardware.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Tier 4: Real-World Scenarios (E2E)                     │
│    Real multi-taxa audio, full field lifecycle, end-to-end detection     │
├──────────────────────────────────────────────────────────────────────────┤
│             Tier 3: Cross-Feature Integration Pipelines                  │
│    Download mock -> Preprocess -> Multi-Backbone Embed -> Engine -> DB   │
├──────────────────────────────────────────────────────────────────────────┤
│              Tier 2: Boundary & Corner Cases (Adversarial)               │
│   Silence, 0-byte, NaN/Inf, clipped audio, extreme thresholds, K=1       │
├──────────────────────────────────────────────────────────────────────────┤
│                 Tier 1: Feature Coverage (Unit & Contract)               │
│     WAV decoding, VAD energy, vector math, centroid, SQLite CRUD         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory & Component Structure

All testing code is isolated in the `tests/` directory:

```
tests/
├── fixtures/
│   ├── __init__.py
│   ├── synth_audio.py         # Pure Python/NumPy synthetic audio & WAV generator
│   └── mock_data.py           # Species catalog, mock Xeno-Canto API, mock embeddings
├── e2e/
│   ├── __init__.py
│   ├── test_tier1_features.py # Tier 1: Audio standardization, VAD, species catalog
│   ├── test_tier2_boundaries.py# Tier 2: Boundary conditions, corrupted/clipped audio
│   ├── test_tier3_cross_feature.py # Tier 3: Pipeline integration (M3/M4)
│   └── test_tier4_real_world.py    # Tier 4: Real-world scenarios (M4/M5)
└── run_e2e_tests.py           # Unified test runner with metrics & exit codes
```

---

## 3. Synthetic Audio Synthesis Engine (`tests/fixtures/synth_audio.py`)

To eliminate network downloads and binary blob dependencies, the test harness synthesizes high-fidelity audio signals directly in NumPy:

### 3.1 Synthesis Primitives
1. **Pure Tones (`generate_pure_tone`)**:
   $$s(t) = A \sin(2\pi f t + \phi)$$
   Generates deterministic single-frequency sinusoids for frequency response and resampling verification.

2. **Harmonic Chirps (`generate_harmonic_chirp`)**:
   Linear frequency sweep with $H$ harmonics:
   $$f(t) = f_{\text{start}} + \frac{f_{\text{end}} - f_{\text{start}}}{T} t$$
   $$s(t) = A \sum_{k=1}^H \frac{1}{k} \sin\left(2\pi k \int_0^t f(\tau) d\tau\right)$$
   Simulates frequency-modulated calls such as bird whistles and crow caws.

3. **Broadband Noise (`generate_broadband_noise`)**:
   - **White Noise**: Uncorrelated Gaussian $\mathcal{N}(0, \sigma^2)$ flat spectral density.
   - **Pink Noise**: $1/f$ spectral slope using colored filter poles.
   - **Brownian Noise**: $1/f^2$ integrated noise.

4. **Impulsive Transients (`generate_transient_pulse`)**:
   Periodic high-frequency carrier bursts modulated by narrow rectangular or exponential pulse envelopes:
   $$s(t) = \Pi(t, T_{\text{pulse}}) \cdot \sin(2\pi f_c t)$$
   Simulates insect stridulation clicks and bioacoustic snapping.

5. **Taxon-Specific Animal Calls (`generate_animal_call`)**:
   - **Aves (Bird)**: Frequency-modulated harmonic call (1.5 kHz to 4.5 kHz with 3 harmonics).
   - **Insecta (Insect)**: High-frequency pulsed stridulation train (5.0 kHz to 8.0 kHz, 25 Hz pulse repetition).
   - **Amphibia (Frog)**: Low-frequency resonant pulses (300 Hz to 1200 Hz with fast exponential decay).
   - **Mammalia (Mammal)**: Wideband multi-tonal chatter with amplitude modulation.

6. **Boundary Signals**:
   - **Digital Silence (`generate_silence`)**: Exact $0.0$ float32 vectors.
   - **DC Offset (`generate_dc_offset`)**: Audio superposed with constant voltage bias.
   - **Hard Clipped Audio (`generate_clipped_audio`)**: Driven into saturation $\text{clip}(A \cdot s(t), -1.0, 1.0)$.

### 3.2 WAV Binary Serialization
- Standard library `wave` and `struct` packaging:
  - Supports 8-bit, 16-bit PCM, and float32 mono or stereo WAV.
  - Generates corrupted WAV files on demand: 0-byte files, truncated RIFF headers, invalid chunk sizes, and corrupted format codes.

---

## 4. Metadata Fixtures (`tests/fixtures/mock_data.py`)

1. **Indian Wildlife Species Catalog**:
   - Complete 33 target species catalog across the 4 taxa:
     - **Aves (15 species)**: e.g., *Corvus splendens* (House Crow), *Centropus sinensis* (Greater Coucal), *Pycnonotus cafer* (Red-vented Bulbul), *Psittacula krameri* (Rose-ringed Parakeet), *Dicrurus macrocercus* (Black Drongo), etc.
     - **Insecta (8 species)**: e.g., *Gryllodes sigillatus* (Indian Cricket), *Acheta domesticus*, *Schizodactylus monstrosus*, etc.
     - **Amphibia (5 species)**: e.g., *Hoplobatrachus tigerinus* (Indian Bullfrog), *Duttaphrynus melanostictus*, *Euphlyctis cyanophlyctis*, etc.
     - **Mammalia (5 species)**: e.g., *Funambulus palmarum* (Indian Palm Squirrel), *Semnopithecus entellus*, *Canis aureus*, etc.
   - Full attributes: `species_id`, `scientific_name`, `common_name`, `taxon`, `expected_freq_range_hz`.

2. **Mock Xeno-Canto API Responses**:
   - Simulates Xeno-Canto API v2/v3 responses with valid JSON payloads, pagination (`page`, `numPages`, `numRecordings`), audio download URLs, and licensing info.

3. **Deterministic Mock Embeddings**:
   - Deterministic orthogonal embeddings on the unit hypersphere:
     $$\|v\|_2 = 1.0 \pm 10^{-6}$$
   - Same-species samples exhibit tight cosine similarity ($> 0.85$).
   - Different species exhibit high cosine distance ($sim < 0.20$).

---

## 5. Test Tier Specifications

### Tier 1: Feature Coverage (`tests/e2e/test_tier1_features.py`)
Verifies primary happy path behaviors and interface contracts:
1. **Audio Standardization (`standardize_audio`)**:
   - Input: WAV files of various sample rates (22,050 Hz, 44,100 Hz, 96,000 Hz) and channel counts (mono, stereo).
   - Expected Output: 1D float32 numpy array, sampling rate = 48,000 Hz, normalized in $[-1.0, 1.0]$.
2. **Audio Slicing (`slice_audio_segments`)**:
   - Input: Standardized audio array of duration 9.0s.
   - Expected Output: List of 3 slices, each exactly 144,000 samples ($3.0 \times 48000$).
3. **Short Audio Padding**:
   - Input: Audio array of duration 1.5s (< 3.0s).
   - Expected Output: Padded to exactly 144,000 samples.
4. **VAD Energy Filtering**:
   - Input: Digital silence vs high-energy animal call.
   - Expected Output: Silence is discarded (`len(segments) == 0`); animal call is retained.
5. **Species Catalog Query**:
   - Input: Query by taxon or species name.
   - Expected Output: Correct subset of species matching the 33 target species catalog.

### Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundaries.py`)
Verifies robustness against degenerate, adversarial, and edge condition inputs:
1. **Empty / 0-Byte File**:
   - Input: File with size = 0 bytes.
   - Expected Output: Raises `AudioFormatError` or `CorruptedAudioError` (never an unhandled exception or crash).
2. **Corrupted File Header**:
   - Input: Non-WAV binary file or text file renamed to `.wav`.
   - Expected Output: Raises `AudioFormatError`.
3. **Truncated File**:
   - Input: Incomplete WAV file where data chunk ends prematurely.
   - Expected Output: Raises `AudioFormatError`.
4. **Digital Silence (All Zeros)**:
   - Input: Float32 array containing all zeros ($A = 0.0$).
   - Expected Output: No `ZeroDivisionError` during normalization, VAD, or feature extraction.
5. **Extreme Clipping / Saturated Audio**:
   - Input: Square wave clipped at $\pm 1.0$.
   - Expected Output: Valid processed array with zero `NaN` or `Inf` elements.
6. **DC Offset Audio**:
   - Input: Audio with $+0.5$ constant DC offset.
   - Expected Output: Processed without float overflow.
7. **Extreme Sampling Rates**:
   - Input: 8,000 Hz (telephony) and 96,000 Hz (ultrasonic) audio files.
   - Expected Output: Successfully resampled to 48,000 Hz.
8. **Sub-3s Audio Boundary**:
   - Input: Audio shorter than 1 frame (e.g. 100 samples) or 1.0s.
   - Expected Output: Safely handled without out-of-bounds indexing.

---

## 6. Authoritative Source of Expected Outputs

| Test Case | Input | Expected Output Source | Authoritative Derivation |
|:---|:---|:---|:---|
| Sampling Rate Conversion | WAV @ 44.1 kHz / 96 kHz | `PROJECT.md` § Interface Contracts | Standardized to exactly 48,000 Hz float32 array |
| Channel Downmixing | Stereo 2-channel WAV | `PROJECT.md` § Audio Ingestion | Monophonic average $\frac{1}{C}\sum c_i$, 1D shape |
| Segment Slicing | Audio array (432,000 samples) | `PROJECT.md` § Uniform Audio Slicing | Exactly three segments of shape `(144000,)` |
| VAD Silence Rejection | All-zeros audio ($A = 0$) | `PROJECT.md` § RMS Energy VAD | $\text{RMS} = 0 \le 3.0 \times \text{floor} \implies$ Discarded |
| VAD Animal Call Retention | Chirp ($A = 0.8$) + Noise floor | `PROJECT.md` § RMS Energy VAD | $\text{RMS} \gg 3.0 \times \text{floor} \implies$ Retained |
| Species Catalog Completeness | Taxon query | `PROJECT.md` § Feature 3 | 33 Indian species (15 Aves, 8 Insecta, 5 Amphibia, 5 Mammalia) |
| Degenerate 0-Byte Input | 0-byte file | `PROJECT.md` § Contract 1 | Raises `AudioFormatError` |
| Corrupted Header | Text string renamed `.wav` | `PROJECT.md` § Contract 1 | Raises `AudioFormatError` |
| Digital Silence Norm | $v = \mathbf{0}$ | IEEE 754 & Vector Math | Safe zero norm handling, no division by zero |

---

## 7. Execution & Runner Specification (`tests/run_e2e_tests.py`)

### 7.1 Running Tests
Execute the standalone test runner:
```powershell
python tests/run_e2e_tests.py
```

### 7.2 Command-Line Options
- `--tier <1|2|all>`: Run a specific tier or all tests (default: `all`).
- `-v`, `--verbose`: Detailed test progress output.
- `--fast`: Enforce strict execution budget (< 5.0 seconds).
- `--json-output <path>`: Write structured test results to JSON file.

### 7.3 Performance Budget
- **Tier 1 (Features)**: $< 2.0$ seconds.
- **Tier 2 (Boundaries)**: $< 1.5$ seconds.
- **Full Suite**: $< 5.0$ seconds total.

### 7.4 Exit Codes
- `0`: All executed tests passed (or skipped due to pending milestones).
- `1`: One or more tests failed or encountered errors.
- `2`: Test runner configuration error.
