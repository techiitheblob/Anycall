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
import urllib.request
import urllib.parse
import requests

from anycall.data.species import SpeciesRecord, get_species_catalog

logger = logging.getLogger(__name__)

class DownloadStatus(Enum):
    DOWNLOADED = "downloaded"
    CACHED = "cached"
    FAILED = "failed"

@dataclass
class HarvestResult:
    recording_id: str
    species_id: str
    file_path: Optional[Path]
    status: DownloadStatus
    metadata: Dict[str, Any]

@dataclass
class HarvestSummary:
    species_id: str
    total_requested: int
    downloaded_count: int
    cached_count: int
    failed_count: int
    results: List[HarvestResult]
    
    @property
    def valid_file_paths(self) -> List[Path]:
        return [r.file_path for r in self.results if r.status != DownloadStatus.FAILED and r.file_path is not None]

class XenoCantoHarvester:
    BASE_API_URL = "https://xeno-canto.org/api/3/recordings"
    BASE_DOWNLOAD_URL = "https://xeno-canto.org/{id}/download"
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
        self.api_key = api_key
        
        if raw_dir is not None:
            self.output_dir = Path(raw_dir)
        elif output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("data/raw")
            
        self.rate_limit = min_delay_seconds if min_delay_seconds is not None else rate_limit
        self.timeout = timeout_seconds if timeout_seconds is not None else timeout
        self.max_retries = max_retries
        self.force_fallback = force_fallback or (self.api_key is None)
        self.mode = "api_v3" if (not self.force_fallback and self.api_key) else "seed_catalog_fallback"

        self._last_request_time: float = 0.0

    def _enforce_rate_limit(self) -> None:
        if self.rate_limit <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def resolve_species_record(self, species_query: Union[str, SpeciesRecord]) -> SpeciesRecord:
        if isinstance(species_query, SpeciesRecord):
            return species_query
        catalog = get_species_catalog()
        for sp in catalog:
            if sp.species_id == species_query or sp.scientific_name.lower() == species_query.lower() or sp.common_name.lower() == species_query.lower():
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
        results: List[HarvestResult] = []
        page = 1
        recordings_meta: List[Dict[str, Any]] = []

        query = species.search_query
        if not query.startswith("sp:") and not query.startswith("gen:"):
            query = f'sp:"{query}"'
        if country and "cnt:" not in query.lower():
            query = f"{query} cnt:{country}"
        query = f"{query} len:\"<16\""

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
                        
                # Skip clips longer than 45 seconds to prevent background species pollution
                length_str = rec.get("length", "0:0")
                try:
                    parts = length_str.split(":")
                    if len(parts) == 2:
                        mins, secs = int(parts[0]), int(parts[1])
                        if mins * 60 + secs > 45:
                            continue
                except ValueError:
                    pass

                recordings_meta.append(rec)
                if len(recordings_meta) >= target_count:
                    break

            if page >= data.get("numPages", 1):
                break
            page += 1

        for meta in recordings_meta:
            rec_id = meta.get("id")
            if not rec_id: continue
            res = self._download_file(rec_id, species.species_id, dest_dir, meta, force)
            results.append(res)

        return results

    def _harvest_mode_b(
        self,
        species: SpeciesRecord,
        dest_dir: Path,
        target_count: int,
        force: bool,
    ) -> List[HarvestResult]:
        results: List[HarvestResult] = []
        if not species.seed_recording_ids:
            logger.warning(f"No seed recording IDs found for {species.species_id}")
            return results

        for rec_id in species.seed_recording_ids[:target_count]:
            meta = {"id": rec_id, "_mode": "seed_fallback"}
            res = self._download_file(rec_id, species.species_id, dest_dir, meta, force)
            results.append(res)

        return results

    def _download_file(
        self,
        recording_id: str,
        species_id: str,
        dest_dir: Path,
        metadata: Dict[str, Any],
        force: bool,
    ) -> HarvestResult:
        file_path = dest_dir / f"{recording_id}.mp3"
        
        if file_path.exists() and file_path.stat().st_size > 1024 and not force:
            return HarvestResult(
                recording_id=recording_id,
                species_id=species_id,
                file_path=file_path,
                status=DownloadStatus.CACHED,
                metadata=metadata,
            )

        download_url = self.BASE_DOWNLOAD_URL.format(id=recording_id)
        
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
                        metadata=metadata,
                    )

                temp_path = dest_dir / f".{recording_id}.mp3.tmp"
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                temp_path.rename(file_path)
                return HarvestResult(
                    recording_id=recording_id,
                    species_id=species_id,
                    file_path=file_path,
                    status=DownloadStatus.DOWNLOADED,
                    metadata=metadata,
                )
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** (attempt - 1)))
                else:
                    return HarvestResult(
                        recording_id=recording_id,
                        species_id=species_id,
                        file_path=None,
                        status=DownloadStatus.FAILED,
                        metadata=metadata,
                    )
                    
        return HarvestResult(recording_id, species_id, None, DownloadStatus.FAILED, metadata)

    def _update_metadata_registry(self, dest_dir: Path, species: SpeciesRecord, results: List[HarvestResult]) -> None:
        registry_path = dest_dir / "metadata.json"
        registry = {}
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except json.JSONDecodeError:
                pass
                
        for res in results:
            if res.status != DownloadStatus.FAILED:
                registry[res.recording_id] = res.metadata
                
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
