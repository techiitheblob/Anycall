"""Test fixtures for AnyCall E2E test suite."""

from tests.fixtures.synth_audio import (
    generate_pure_tone,
    generate_harmonic_chirp,
    generate_broadband_noise,
    generate_transient_pulse,
    generate_animal_call,
    generate_silence,
    generate_dc_offset,
    generate_clipped_audio,
    save_wav_file,
    read_wav_file,
    create_corrupted_wav_file,
    create_multichannel_wav_file,
)
from tests.fixtures.mock_data import (
    MOCK_INDIAN_SPECIES_CATALOG,
    get_mock_catalog,
    get_mock_species_by_taxon,
    get_mock_species_by_id,
    generate_mock_xeno_canto_response,
    generate_mock_embedding,
    generate_mock_batch_embeddings,
)

__all__ = [
    "generate_pure_tone",
    "generate_harmonic_chirp",
    "generate_broadband_noise",
    "generate_transient_pulse",
    "generate_animal_call",
    "generate_silence",
    "generate_dc_offset",
    "generate_clipped_audio",
    "save_wav_file",
    "read_wav_file",
    "create_corrupted_wav_file",
    "create_multichannel_wav_file",
    "MOCK_INDIAN_SPECIES_CATALOG",
    "get_mock_catalog",
    "get_mock_species_by_taxon",
    "get_mock_species_by_id",
    "generate_mock_xeno_canto_response",
    "generate_mock_embedding",
    "generate_mock_batch_embeddings",
]
