# AnyCall: E2E Test Suite Ready Specification (`TEST_READY.md`)

## 1. Overview & Architecture

AnyCall's end-to-end testing suite is designed under the **Dual Track Strategy** to enforce opaque-box, requirement-driven verification across all 4 operational tiers:
- **Tier 1 (Feature Coverage & Contracts)**: WAV standardization, energy-based VAD, audio slicing, and Indian wildlife species catalog schema.
- **Tier 2 (Boundary & Corner Cases)**: Corrupted WAV headers, 0-byte files, digital silence, hard-clipped signals, DC offsets, sub-frame segments.
- **Tier 3 (Cross-Feature Integration)**: Multi-component pipelines connecting audio ingestion, slicing, deterministic embedding extraction (`MockBackbone`), prototypical classification, online incremental updates, sub-prototype clustering, and SQLite persistence.
- **Tier 4 (Real-World Scenarios)**: Autonomous edge field monitoring across all 4 target taxa (Aves, Insecta, Amphibia, Mammalia), continuous audio streams, realistic noise SNR mixing (0–20 dB), and open-set unknown sound rejection.

The suite runs **100% offline**, requiring **zero external network downloads** or heavyweight neural model weights, executing in **< 3.0 seconds** (well within the 5.0-second performance ceiling).

---

## 2. Test Tier Coverage Table

| Tier | Test Module | Target Scope | Total Tests | Passed (M1 Ready) | Skipped (M2-M4 Pending) | Failed / Errors | Execution Time |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Tier 1** | `tests/e2e/test_tier1_features.py` | Standardization, Slicing, VAD, Species Catalog, Fixtures | 19 | 19 | 0 | 0 / 0 | 0.22s |
| **Tier 2** | `tests/e2e/test_tier2_boundaries.py` | Degenerate audio, 0-byte, truncated RIFF, saturation, DC | 14 | 14 | 0 | 0 / 0 | 1.01s |
| **Tier 3** | `tests/e2e/test_tier3_cross_feature.py` | Audio -> Embed -> Prototype -> Rejection -> SQLite DB | 15 | 3 | 12 | 0 / 0 | 0.06s |
| **Tier 4** | `tests/e2e/test_tier4_real_world.py` | 4 Taxa field streaming, VAD detection log, SNR rejection | 9 | 3 | 6 | 0 / 0 | 0.07s |
| **TOTAL** | **All 4 Tiers** | **Full System E2E Coverage** | **57** | **39** | **18** | **0 / 0** | **2.57s** |

*Note: The 18 skipped tests in Tier 3 and Tier 4 will automatically unskip and assert against `anycall.embeddings`, `anycall.classifier`, and `anycall.storage` as soon as M2 and M3 workers deliver their respective modules.*

---

## 3. Test Inventory & Feature Mapping

### Tier 1: Feature Coverage (`test_tier1_features.py` — 19 Tests)
- `TestAudioStandardizationContract`:
  - `test_standardize_audio_from_wav_file`: Contract 1 float32 48kHz mono output.
  - `test_standardize_audio_output_path_saving`: WAV writing to disk.
  - `test_standardize_audio_various_sample_rates`: 22.05kHz, 44.1kHz, 48kHz, 96kHz resampling.
- `TestAudioSlicingAndVADContract`:
  - `test_slice_audio_segments_uniform_shape`: Exactly 144,000 samples per 3s slice.
  - `test_slice_audio_segments_padding_short_audio`: Zero-padding for < 3s segments.
  - `test_slice_audio_segments_vad_silence_discard`: Pure digital silence rejected by VAD.
  - `test_slice_audio_segments_vad_animal_call_retained`: High-energy animal calls preserved.
- `TestIndianSpeciesCatalogContract`:
  - `test_species_catalog_completeness_33_species`: 33 Indian species catalog check.
  - `test_species_catalog_taxa_distribution`: 15 Aves, 8 Insecta, 5 Amphibia, 5 Mammalia.
  - `test_species_catalog_fields_schema`: `species_id`, `scientific_name`, `common_name`, `taxon`.
  - `test_species_catalog_includes_corvus_splendens`: Baseline control species check.
- `TestSyntheticAudioFixturesSelfVerification`:
  - `test_pure_tone_generation_properties`
  - `test_harmonic_chirp_properties`
  - `test_broadband_noise_determinism`
  - `test_transient_pulse_properties`
  - `test_animal_call_all_taxa`
  - `test_wav_roundtrip_fidelity`
- `TestMockDataFixturesSelfVerification`:
  - `test_mock_xeno_canto_response_schema`
  - `test_mock_embeddings_unit_norm_and_clustering`

### Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py` — 14 Tests)
- `TestCorruptedAndDegenerateAudio`:
  - `test_empty_zero_byte_file_raises_error`: 0-byte file handling.
  - `test_corrupted_header_non_wav_raises_error`: Non-WAV file disguised as `.wav`.
  - `test_truncated_wav_header_raises_error`: Incomplete 12-byte header.
  - `test_invalid_magic_bytes_raises_error`: Header missing `RIFF` tag.
  - `test_random_binary_garbage_raises_error`: Random binary bytes.
- `TestExtremeAudioSignals`:
  - `test_digital_silence_all_zeros_no_division_by_zero`: All-zeros vector norm stability.
  - `test_clipped_audio_overflow_no_nan_or_inf`: Saturated waveforms bounded in [-1, 1].
  - `test_dc_offset_audio_stability`: Constant DC bias without float overflow.
  - `test_extreme_sample_rates_telephony_and_ultrasonic`: 8 kHz and 96 kHz resampling.
  - `test_extreme_dynamic_range_inaudible_signal`: 1e-6 amplitude handling.
  - `test_sub_frame_short_audio_length_safety`: Sub-50ms audio safely bounded.
- `TestBoundaryFixturesSelfVerification`:
  - `test_corrupted_wav_generator_file_sizes`
  - `test_clipped_audio_fixture_saturation`
  - `test_dc_offset_fixture_mean`

### Tier 3: Cross-Feature Integration Pipelines (`test_tier3_cross_feature.py` — 15 Tests)
- `TestAudioToEmbeddingPipelineIntegration`:
  - `test_wav_file_to_standardized_slices_pipeline`: Multi-taxa stream standardization & VAD segmentation.
  - `test_slices_to_mock_backbone_embedding`: Standardized segment -> MockBackbone -> L2-normalized vector.
  - `test_backbone_file_path_and_array_parity`: Audio file path vs. numpy array embedding parity.
- `TestPrototypicalClassificationEngineContract`:
  - `test_prototype_centroid_unit_normalization`: Mathematical centroid $c_k = \frac{1}{K}\sum e_i$ with $\|c_k\|_2 = 1.0$.
  - `test_multi_species_enrollment_and_classification`: Top-1 cosine similarity prediction.
  - `test_open_set_rejection_below_threshold`: Open-set threshold $\theta$ rejecting novel sounds as `"Unknown"`.
  - `test_tunable_rejection_threshold_sweep`: Behavioral monotonicity over $\theta \in [0.1, 0.99]$.
- `TestIncrementalPrototypeUpdate`:
  - `test_incremental_prototype_update_mathematical_identity`: Incremental update formula matches batch recomputation ($sim > 0.9999$).
  - `test_single_sample_incremental_update`: Online 1-shot update ($M=1$).
- `TestSubPrototypeClustering`:
  - `test_sub_prototype_clustering_multi_call_variance`: Multi-call clustering activated when variance $> 0.05$.
- `TestSQLiteDatabasePersistencePipeline`:
  - `test_save_and_retrieve_prototype_blob_fidelity`: Float32 array BLOB serialization & retrieval.
  - `test_log_detection_event_and_query`: Audit trail logging in `detections` table.
- `TestCrossFeatureMathematicalAndSchemaContracts`:
  - `test_centroid_formula_algebraic_correctness`: Standalone algebraic verification of centroid updates.
  - `test_sqlite_schema_specification_contract`: Standalone SQLite schema contract validation.
- `TestEndToEndCrossFeatureIntegrationPipeline`:
  - `test_full_cross_feature_lifecycle`: Audio -> Slicing -> Backbone -> Classifier -> Database.

### Tier 4: Real-World Scenarios & Field Monitoring (`test_tier4_real_world.py` — 9 Tests)
- `TestFieldMonitoringContinuousStream`:
  - `test_continuous_audio_stream_field_monitoring_pipeline`: Continuous outdoor recording stream with mixed silence, ambient noise, and animal vocalizations producing SQLite detection events.
- `TestCrossTaxaScenariosAllFourTaxa`:
  - `test_all_four_taxa_enrollment_and_classification`: Multi-class enrollment and prediction across Aves, Insecta, Amphibia, and Mammalia.
  - `test_cross_taxa_confusion_rejection`: Inter-taxa separation preventing false cross-taxa classification.
- `TestUnknownSoundRejectionUnderRealisticSNR`:
  - `test_unknown_novel_wildlife_rejection`: Un-enrolled species rejection under realistic signal conditions.
  - `test_rejection_of_broadband_environmental_noise`: Rejection of ambient wind/rain broadband noise without false positives.
- `TestFieldMonitoringEdgeAndStressScenarios`:
  - `test_saturated_clipped_call_pipeline_stability`: Microphone overdrive / hard clipping recovery.
  - `test_high_volume_rapid_detection_logging`: High-concurrency detection event logging without SQLite lock contention.
- `TestRealWorldFixturesAndSNRMathematics`:
  - `test_snr_mixing_mathematical_precision`: Verifies SNR mixing formulas within $\pm 0.5$ dB.
  - `test_all_taxa_animal_call_duration_and_rms_validity`: Validates synthetic call properties across all 4 taxa.

---

## 4. Verification & Execution Instructions

Execute the standalone test runner across all tiers:
```powershell
python tests/run_e2e_tests.py
```

Enforce strict performance budgeting (< 5.0 seconds):
```powershell
python tests/run_e2e_tests.py --fast
```

Run a specific tier with verbose reporting:
```powershell
python tests/run_e2e_tests.py --tier 3 -v
python tests/run_e2e_tests.py --tier 4 -v
```

Export structured JSON execution report:
```powershell
python tests/run_e2e_tests.py --tier all --json-output tests/reports/e2e_metrics.json
```

---

## 5. Exit Code Protocol
- `0`: All executed tests passed (or skipped cleanly due to pending milestone dependencies).
- `1`: Test failures, errors, or execution budget violations (> 5.0 seconds).
- `2`: Invalid CLI arguments or configuration error.
