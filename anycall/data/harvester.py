"""AnyCall Dual-Mode Bioacoustic Data Harvester for Xeno-Canto.

Supports:
1. Mode A: Authenticated API v3 search and pagination.
2. Mode B: Direct unauthenticated seed download fallback (https://xeno-canto.org/<id>/download).
Enforces rate limiting, caching, metadata registry writing, and atomic file saving.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

import requests

from anycall.data.species import (
    SpeciesRecord,
    get_sample_species,
    get_species,
    get_species_by_common_name,
    get_species_by_scientific_name,
)

logger = logging.getLogger("anycall.data.harvester")


class DownloadStatus(str, Enum):
    DOWNLOADED = "downloaded"
    CACHED = "cached"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class HarvestResult:
    """Result of an individual recording harvest attempt."""
    recording_id: str
    species_id: str
    file_path: Optional[Path]
    status: DownloadStatus
    duration_sec: Optional[float] = None
    quality: Optional[str] = None
    recordist: Optional[str] = None
    license_url: Optional[str] = None
    error_message: Optional[str] = None


class HarvestSummary(list):
    """Summary of harvesting operations for a species, behaving as a list of valid file paths."""

    def __init__(
        self,
        species_id: str,
        total_requested: int,
        downloaded_count: int,
        cached_count: int,
        failed_count: int,
        results: List[HarvestResult],
    ):
        self.species_id = species_id
        self.total_requested = total_requested
        self.downloaded_count = downloaded_count
        self.cached_count = cached_count
        self.failed_count = failed_count
        self.results = results
        # Initialize the underlying list with valid file paths
        valid_paths = [r.file_path for r in results if r.file_path and r.file_path.exists()]
        super().__init__(valid_paths)

    @property
    def valid_file_paths(self) -> List[Path]:
        return list(self)


class XenoCantoHarvester:
    """Dual-mode bioacoustic harvester for Xeno-Canto recordings."""

    BASE_API_URL = "https://xeno-canto.org/api/3/recordings"
    BASE_DOWNLOAD_URL = "https://xeno-canto.org/{recording_id}/download"
    USER_AGENT = "AnyCall-Research-Harvester/1.0 (+https://github.com/anycall/anycall)"

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
        rate_limit: float = 1.0,
        min_delay_seconds: Optional[float] = None,
        timeout: float = 15.0,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 3,
        force_fallback: bool = False,
    ):
        target_dir = output_dir if output_dir is not None else (raw_dir if raw_dir is not None else Path("data/raw"))
        self.output_dir = Path(target_dir)
        self.raw_dir = self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.environ.get("XENO_CANTO_API_KEY")
        self.rate_limit = min_delay_seconds if min_delay_seconds is not None else rate_limit
        self.timeout = timeout_seconds if timeout_seconds is not None else timeout
        self.max_retries = max_retries
        self.force_fallback = force_fallback or (self.api_key is None)
        self.mode = "api_v3" if (not self.force_fallback and self.api_key) else "seed_catalog_fallback"

        self._last_request_time: float = 0.0

    def _enforce_rate_limit(self) -> None:
        """Enforces minimum delay between consecutive network requests."""
        if self.rate_limit <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def resolve_species_record(self, species_query: Union[str, SpeciesRecord]) -> SpeciesRecord:
        """Resolves species input to a canonical SpeciesRecord."""
        if isinstance(species_query, SpeciesRecord):
            return species_query

        # Check by id, scientific name, or common name
        norm = str(species_query).strip()
        try:
            return get_species(norm)
        except KeyError:
            pass

        sp = get_species_by_scientific_name(norm)
        if sp:
            return sp

        sp = get_species_by_common_name(norm)
        if sp:
            return sp

        raise ValueError(f"Could not resolve species record for query: '{species_query}'")

    def harvest_species(
        self,
        species_query: Union[str, SpeciesRecord],
        limit: int = 20,
        max_recordings: Optional[int] = None,
        quality: Optional[List[str]] = None,
        country: Optional[str] = "india",
        force: bool = False,
        dry_run: bool = False,
    ) -> HarvestSummary:
        """Harvests recordings for the specified species.

        Uses Mode A (API v3) if an API key is present and fallback is not forced;
        otherwise uses Mode B (curated direct download fallback).
        """
        target_count = max_recordings if max_recordings is not None else limit
        species_rec = self.resolve_species_record(species_query)
        species_raw_dir = self.output_dir / species_rec.species_id
        species_raw_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info(f"[Dry Run] Querying recordings for {species_rec.common_name} (target: {target_count})")
            return HarvestSummary(species_rec.species_id, target_count, 0, 0, 0, [])

        if self.mode == "api_v3":
            logger.info(f"Harvesting {species_rec.common_name} via Mode A (API v3 Live Search)")
            results = self._harvest_mode_a(species_rec, species_raw_dir, target_count, quality, country, force)
        else:
            logger.info(f"Harvesting {species_rec.common_name} via Mode B (Direct Seed Fallback)")
            results = self._harvest_mode_b(species_rec, species_raw_dir, target_count, force)

        # Update metadata.json manifest
        self._update_metadata_registry(species_raw_dir, species_rec, results)

        downloaded = sum(1 for r in results if r.status == DownloadStatus.DOWNLOADED)
        cached = sum(1 for r in results if r.status == DownloadStatus.CACHED)
        failed = sum(1 for r in results if r.status == DownloadStatus.FAILED)

        return HarvestSummary(
            species_id=species_rec.species_id,
            total_requested=target_count,
            downloaded_count=downloaded,
            cached_count=cached,
            failed_count=failed,
            results=results,
        )

    def _harvest_mode_a(
        self,
        species: SpeciesRecord,
        dest_dir: Path,
        target_count: int,
        quality: Optional[List[str]],
        country: Optional[str],
        force: bool,
    ) -> List[HarvestResult]:
        """Mode A: Queries API v3 with pagination and downloads results."""
        results: List[HarvestResult] = []
        page = 1
        recordings_meta: List[Dict[str, Any]] = []

        query = species.search_query
        if country and "cnt:" not in query.lower():
            query = f"{query} cnt:{country}"

        while len(recordings_meta) < target_count:
            self._enforce_rate_limit()
            params = {
                "query": query,
                "key": self.api_key,
                "page": page,
            }
            try:
                resp = requests.get(
                    self.BASE_API_URL,
                    params=params,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=self.timeout,
                )
                if resp.status_code == 401:
                    logger.warning("API v3 returned 401 Unauthorized. Falling back to Mode B.")
                    self.mode = "seed_catalog_fallback"
                    return self._harvest_mode_b(species, dest_dir, target_count, force)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"API v3 query failed: {e}. Falling back to Mode B.")
                self.mode = "seed_catalog_fallback"
                return self._harvest_mode_b(species, dest_dir, target_count, force)

            page_recs = data.get("recordings", [])
            if not page_recs:
                break

            for rec in page_recs:
                if quality:
                    rec_q = rec.get("q", "").upper()
                    if rec_q and rec_q not in [q.upper() for q in quality]:
                        continue
                recordings_meta.append(rec)
                if len(recordings_meta) >= target_count:
                    break

            num_pages = int(data.get("numPages", 1))
            if page >= num_pages:
                break
            page += 1

        for rec in recordings_meta:
            rec_id = str(rec["id"])
            res = self.download_recording(
                recording_id=rec_id,
                dest_dir=dest_dir,
                species_id=species.species_id,
                meta=rec,
                force=force,
            )
            results.append(res)

        return results

    def _harvest_mode_b(
        self,
        species: SpeciesRecord,
        dest_dir: Path,
        target_count: int,
        force: bool,
    ) -> List[HarvestResult]:
        """Mode B: Downloads directly from curated seed IDs without authentication."""
        results: List[HarvestResult] = []
        seed_ids = species.seed_recording_ids[:target_count]

        if not seed_ids:
            logger.warning(f"No seed recording IDs found for {species.species_id}")
            return results

        for rec_id in seed_ids:
            res = self.download_recording(
                recording_id=rec_id,
                dest_dir=dest_dir,
                species_id=species.species_id,
                force=force,
            )
            results.append(res)

        return results

    def download_recording(
        self,
        recording_id: str,
        dest_dir: Path,
        species_id: str,
        meta: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> HarvestResult:
        """Downloads a single recording with atomic writing and caching."""
        dest_file = dest_dir / f"{recording_id}.mp3"
        part_file = dest_dir / f"{recording_id}.mp3.part"

        # Check existing valid cache
        if dest_file.exists() and not force:
            size = dest_file.stat().st_size
            if size > 100:  # Valid non-empty file
                return HarvestResult(
                    recording_id=recording_id,
                    species_id=species_id,
                    file_path=dest_file,
                    status=DownloadStatus.CACHED,
                    quality=meta.get("q") if meta else None,
                    recordist=meta.get("rec") if meta else None,
                    license_url=meta.get("lic") if meta else None,
                )

        download_url = self.BASE_DOWNLOAD_URL.format(recording_id=recording_id)
        self._enforce_rate_limit()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    download_url,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True,
                )
                if resp.status_code != 200:
                    if attempt < self.max_retries and resp.status_code in {429, 500, 502, 503, 504}:
                        time.sleep(1.0 * (2 ** (attempt - 1)))
                        continue
                    return HarvestResult(
                        recording_id=recording_id,
                        species_id=species_id,
                        file_path=None,
                        status=DownloadStatus.FAILED,
                        error_message=f"HTTP status {resp.status_code}",
                    )

                # Write content to temporary .part file
                with open(part_file, "wb") as f:
                    if hasattr(resp, "iter_content"):
                        has_chunks = False
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                has_chunks = True
                        if not has_chunks and hasattr(resp, "content") and resp.content:
                            f.write(resp.content)
                    elif hasattr(resp, "content") and resp.content:
                        f.write(resp.content)

                if not part_file.exists() or part_file.stat().st_size == 0:
                    part_file.unlink(missing_ok=True)
                    return HarvestResult(
                        recording_id=recording_id,
                        species_id=species_id,
                        file_path=None,
                        status=DownloadStatus.FAILED,
                        error_message="Downloaded payload is 0 bytes",
                    )

                # Atomic replace
                part_file.replace(dest_file)
                return HarvestResult(
                    recording_id=recording_id,
                    species_id=species_id,
                    file_path=dest_file,
                    status=DownloadStatus.DOWNLOADED,
                    quality=meta.get("q") if meta else None,
                    recordist=meta.get("rec") if meta else None,
                    license_url=meta.get("lic") if meta else None,
                )
            except Exception as e:
                part_file.unlink(missing_ok=True)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** (attempt - 1)))
                    continue
                return HarvestResult(
                    recording_id=recording_id,
                    species_id=species_id,
                    file_path=None,
                    status=DownloadStatus.FAILED,
                    error_message=str(e),
                )

        return HarvestResult(
            recording_id=recording_id,
            species_id=species_id,
            file_path=None,
            status=DownloadStatus.FAILED,
            error_message="Max retries exceeded",
        )

    def _update_metadata_registry(
        self,
        dest_dir: Path,
        species: SpeciesRecord,
        results: List[HarvestResult],
    ) -> None:
        """Writes or updates metadata.json in species directory."""
        meta_file = dest_dir / "metadata.json"
        existing_meta: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = {}

        recordings_dict = existing_meta.get("recordings", {})
        for r in results:
            if r.file_path and r.file_path.exists():
                recordings_dict[r.recording_id] = {
                    "recording_id": r.recording_id,
                    "filename": r.file_path.name,
                    "file_url": self.BASE_DOWNLOAD_URL.format(recording_id=r.recording_id),
                    "file_size_bytes": r.file_path.stat().st_size,
                    "status": r.status.value,
                    "quality": r.quality,
                    "recordist": r.recordist,
                    "license": r.license_url or "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

        payload = {
            "species_id": species.species_id,
            "species_scientific": species.scientific_name,
            "species_common": species.common_name,
            "taxon": species.taxon.value,
            "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": self.mode,
            "count": len(recordings_dict),
            "recordings": recordings_dict,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for Xeno-Canto harvester."""
    parser = argparse.ArgumentParser(
        prog="python -m anycall.data.harvester",
        description="AnyCall Bioacoustic Data Harvester (Xeno-Canto API v3 & Fallback)"
    )
    parser.add_argument(
        "--species", "-s",
        type=str,
        default="Corvus splendens",
        help="Target species scientific name, common name, or slug (default: 'Corvus splendens')"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Harvest sample species Corvus splendens"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Target recording count per species (default: 20)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw audio files (default: data/raw)"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Xeno-Canto API v3 key"
    )
    parser.add_argument(
        "--quality", "-q",
        type=str,
        default="A,B",
        help="Comma-separated allowed audio qualities (default: 'A,B')"
    )
    parser.add_argument(
        "--country", "-c",
        type=str,
        default="india",
        help="Country search filter (default: 'india')"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Rate limiting delay in seconds between HTTP requests (default: 1.0)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15.0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cached files"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Immediately run audio standardization on downloaded files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidate recordings without downloading"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose DEBUG logging"
    )
    return parser


def main() -> int:
    """CLI entry point for harvester."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    species_target = "Corvus splendens" if args.sample_only else args.species

    try:
        harvester = XenoCantoHarvester(
            api_key=args.api_key,
            output_dir=args.output_dir,
            rate_limit=args.rate_limit,
            timeout=args.timeout,
        )

        downloaded = harvester.harvest_species(
            species_query=species_target,
            limit=args.limit,
            quality=args.quality.split(",") if args.quality else None,
            country=args.country,
            force=args.force,
            dry_run=args.dry_run,
        )

        logger.info(f"Harvested {len(downloaded)} valid files for {species_target}")

        if args.process:
            from anycall.audio.standardize import AudioStandardizer
            species_rec = harvester.resolve_species_record(species_target)
            raw_sp_dir = args.output_dir / species_rec.species_id
            proc_sp_dir = Path("data/processed") / species_rec.species_id
            logger.info(f"Processing audio from {raw_sp_dir} to {proc_sp_dir}")
            standardizer = AudioStandardizer(target_sr=48000, segment_duration=3.0, vad_filter=True)
            total_seg = standardizer.process_directory(raw_sp_dir, proc_sp_dir, force=args.force)
            logger.info(f"Audio processing complete: generated {total_seg} 3.0s WAV segments.")

        return 0 if len(downloaded) >= args.limit else 3
    except Exception as exc:
        logger.error(f"Harvester failed: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
