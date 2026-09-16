"""AnyCall Automated Benchmarking Runner.

Runs 5 experiments for the APSCON paper:
  Exp 1: BirdNET stock classifier failure analysis vs. AnyCall
  Exp 2: Few-shot accuracy curves (K=5,10,20) for all 3 backbones
  Exp 3: Cross-taxa generalization breakdown (Aves/Insecta/Amphibia/Mammalia)
  Exp 4: Rejection quality ROC (FAR vs. TAR sweeping θ, EER & AUROC)
  Exp 5: Edge latency & RAM profiling per backbone

Outputs: benchmark_report.json, benchmark_report.csv
"""
from __future__ import annotations

import csv
import gc
import json
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings import get_backbone
from anycall.embeddings.base import BaseAudioEmbeddingBackbone


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FewShotResult:
    backbone: str
    k_shots: int
    fold: int
    top1_accuracy: float
    n_queries: int


@dataclass
class CrossTaxaResult:
    backbone: str
    taxon: str
    k_shots: int
    top1_accuracy: float
    n_species: int
    n_queries: int


@dataclass
class RocResult:
    backbone: str
    threshold: float
    far: float   # False Acceptance Rate (open-set intruder accepted)
    tar: float   # True Acceptance Rate (genuine accepted)


@dataclass
class LatencyResult:
    backbone: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_trials: int
    embedding_dim: int


@dataclass
class BirdNetFailureResult:
    species_label: str
    taxon: str
    birdnet_correct: bool
    anycall_correct: bool
    birdnet_top_prediction: str
    birdnet_score: float
    anycall_score: float


@dataclass
class BenchmarkReport:
    exp1_birdnet_failure: List[Dict] = field(default_factory=list)
    exp2_few_shot: List[Dict] = field(default_factory=list)
    exp3_cross_taxa: List[Dict] = field(default_factory=list)
    exp4_roc: List[Dict] = field(default_factory=list)
    exp5_latency: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates all 5 benchmark experiments.

    Parameters
    ----------
    data_dir : Path
        Root of processed data: data_dir/processed/{species_id}/*.wav
    k_shots_list : list of int
        Support set sizes to sweep. Default [5, 10, 20].
    n_folds : int
        Number of cross-validation folds. Default 5.
    rejection_thresholds : list of float
        θ values swept for the ROC curve. Default linspace(0.0, 1.0, 51).
    backbones : list of str
        Backbone names to evaluate. Default ['mock', 'birdnet', 'perch', 'panns'].
        Use 'mock' for fast CI runs.
    n_latency_trials : int
        Number of forward passes for latency profiling. Default 20.
    seed : int
        Random seed for reproducibility.
    """

    DEFAULT_BACKBONES = ["birdnet", "perch", "panns"]

    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        k_shots_list: List[int] = None,
        n_folds: int = 5,
        rejection_thresholds: Optional[List[float]] = None,
        backbones: Optional[List[str]] = None,
        n_latency_trials: int = 20,
        seed: int = 42,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._k_shots_list = k_shots_list or [5, 10, 20]
        self._n_folds = n_folds
        self._thresholds = rejection_thresholds or list(np.linspace(0.0, 1.0, 51))
        self._backbone_names = backbones or self.DEFAULT_BACKBONES
        self._n_latency_trials = n_latency_trials
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run_all(
        self,
        output_json: str | Path = "benchmark_report.json",
        output_csv: str | Path = "benchmark_report.csv",
    ) -> BenchmarkReport:
        """Run all 5 experiments and write results to disk.

        Returns the populated BenchmarkReport.
        """
        report = BenchmarkReport()

        # Discover species and their WAV segments
        species_map = self._discover_species()
        if not species_map:
            report.errors.append(
                f"No processed species directories found under {self._data_dir}. "
                "Run the data pipeline first."
            )
            self._write_outputs(report, output_json, output_csv)
            return report

        print(f"[Benchmark] Found {len(species_map)} species.")

        for backbone_name in self._backbone_names:
            print(f"\n[Benchmark] === Backbone: {backbone_name.upper()} ===")
            try:
                backbone = get_backbone(backbone_name)
            except Exception as exc:
                msg = f"Failed to load backbone '{backbone_name}': {exc}"
                print(f"  [WARN] {msg}")
                report.errors.append(msg)
                continue

            # Pre-extract all embeddings for this backbone
            print(f"  [Benchmark] Extracting embeddings...")
            embeddings_map = self._extract_all_embeddings(backbone, species_map)

            # Exp 2: Few-shot curves
            print(f"  [Benchmark] Exp 2: Few-shot accuracy curves...")
            few_shot_rows = self._exp2_few_shot(backbone_name, embeddings_map)
            report.exp2_few_shot.extend([asdict(r) for r in few_shot_rows])

            # Exp 3: Cross-taxa
            print(f"  [Benchmark] Exp 3: Cross-taxa generalization...")
            taxa_rows = self._exp3_cross_taxa(backbone_name, embeddings_map, species_map)
            report.exp3_cross_taxa.extend([asdict(r) for r in taxa_rows])

            # Exp 4: ROC
            print(f"  [Benchmark] Exp 4: Rejection quality ROC...")
            roc_rows = self._exp4_roc(backbone_name, embeddings_map)
            report.exp4_roc.extend([asdict(r) for r in roc_rows])

            # Exp 5: Latency
            print(f"  [Benchmark] Exp 5: Latency & RAM profiling...")
            lat_row = self._exp5_latency(backbone_name, backbone, species_map)
            if lat_row:
                report.exp5_latency.append(asdict(lat_row))

            del backbone
            gc.collect()

        # Exp 1: BirdNET stock failure analysis (uses birdnet backbone if available)
        print(f"\n[Benchmark] Exp 1: BirdNET stock failure analysis...")
        try:
            failure_rows = self._exp1_birdnet_failure(species_map)
            report.exp1_birdnet_failure.extend([asdict(r) for r in failure_rows])
        except Exception as exc:
            msg = f"Exp 1 failed: {exc}\n{traceback.format_exc()}"
            print(f"  [WARN] {msg}")
            report.errors.append(msg)

        # Build summary
        report.summary = self._build_summary(report)

        self._write_outputs(report, output_json, output_csv)
        print(f"\n[Benchmark] Done. Reports written to {output_json} and {output_csv}")
        return report

    # ------------------------------------------------------------------
    # Data discovery
    # ------------------------------------------------------------------

    def _discover_species(self) -> Dict[str, List[Path]]:
        """Return {species_id: [wav_path, ...]} for all species with >= 5 WAVs."""
        result: Dict[str, List[Path]] = {}
        if not self._data_dir.is_dir():
            return result
        for species_dir in sorted(self._data_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            wavs = sorted(species_dir.glob("*.wav"))
            if len(wavs) >= 5:
                result[species_dir.name] = wavs
        return result

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_all_embeddings(
        self,
        backbone: BaseAudioEmbeddingBackbone,
        species_map: Dict[str, List[Path]],
    ) -> Dict[str, np.ndarray]:
        """Return {species_id: (N, D) float32 embedding matrix}."""
        emb_map: Dict[str, np.ndarray] = {}
        for sp, wavs in species_map.items():
            vecs = []
            for wav in wavs:
                try:
                    vec = backbone.embed(wav)
                    vecs.append(vec)
                except Exception:
                    continue
            if vecs:
                emb_map[sp] = np.stack(vecs, axis=0).astype(np.float32)
        return emb_map

    # ------------------------------------------------------------------
    # Exp 2: Few-shot accuracy curves
    # ------------------------------------------------------------------

    def _exp2_few_shot(
        self,
        backbone_name: str,
        embeddings_map: Dict[str, np.ndarray],
    ) -> List[FewShotResult]:
        results = []
        species_list = [s for s, e in embeddings_map.items() if len(e) >= 6]
        if len(species_list) < 2:
            return results

        for k in self._k_shots_list:
            eligible = [s for s in species_list if len(embeddings_map[s]) > k]
            if len(eligible) < 2:
                continue

            fold_accuracies = []
            for fold in range(self._n_folds):
                correct = total = 0
                clf = PrototypicalClassifier(threshold=0.0)  # no rejection for accuracy

                for sp in eligible:
                    embs = embeddings_map[sp]
                    indices = list(range(len(embs)))
                    self._rng.shuffle(indices)
                    support_idx = indices[:k]
                    query_idx = indices[k:k + max(1, (len(embs) - k) // self._n_folds)]

                    support = [embs[i] for i in support_idx]
                    clf.enroll(sp, support)

                for sp in eligible:
                    embs = embeddings_map[sp]
                    indices = list(range(len(embs)))
                    self._rng.shuffle(indices)
                    query_idx = indices[k:k + max(1, (len(embs) - k) // self._n_folds)]

                    for qi in query_idx:
                        pred_label, _ = clf.predict(embs[qi])
                        correct += int(pred_label == sp)
                        total += 1

                acc = correct / total if total > 0 else 0.0
                fold_accuracies.append(acc)
                results.append(FewShotResult(
                    backbone=backbone_name,
                    k_shots=k,
                    fold=fold,
                    top1_accuracy=round(acc, 4),
                    n_queries=total,
                ))

            mean_acc = np.mean(fold_accuracies) if fold_accuracies else 0.0
            print(f"    K={k:2d} shots → mean top-1 acc: {mean_acc:.3f}")

        return results

    # ------------------------------------------------------------------
    # Exp 3: Cross-taxa generalization
    # ------------------------------------------------------------------

    def _exp3_cross_taxa(
        self,
        backbone_name: str,
        embeddings_map: Dict[str, np.ndarray],
        species_map: Dict[str, List[Path]],
    ) -> List[CrossTaxaResult]:
        """Compute per-taxon accuracy using taxon suffix heuristic in species_id."""
        results = []

        # Build taxon lookup from species_id naming convention: e.g. corvus_splendens → aves
        # We use the data/species catalog if available, else fall back to name heuristics.
        taxon_of = self._build_taxon_lookup(list(species_map.keys()))

        taxa_species: Dict[str, List[str]] = {}
        for sp in embeddings_map:
            taxon = taxon_of.get(sp, "unknown")
            taxa_species.setdefault(taxon, []).append(sp)

        for taxon, species_in_taxon in taxa_species.items():
            eligible = [s for s in species_in_taxon if len(embeddings_map[s]) >= 6]
            if len(eligible) < 2:
                continue

            for k in self._k_shots_list[:1]:  # Use K=5 for cross-taxa
                correct = total = 0
                clf = PrototypicalClassifier(threshold=0.0)

                for sp in eligible:
                    embs = embeddings_map[sp]
                    indices = list(range(len(embs)))
                    self._rng.shuffle(indices)
                    support = [embs[i] for i in indices[:k]]
                    clf.enroll(sp, support)

                for sp in eligible:
                    embs = embeddings_map[sp]
                    indices = list(range(len(embs)))
                    self._rng.shuffle(indices)
                    for qi in indices[k:]:
                        pred_label, _ = clf.predict(embs[qi])
                        correct += int(pred_label == sp)
                        total += 1

                acc = correct / total if total > 0 else 0.0
                results.append(CrossTaxaResult(
                    backbone=backbone_name,
                    taxon=taxon,
                    k_shots=k,
                    top1_accuracy=round(acc, 4),
                    n_species=len(eligible),
                    n_queries=total,
                ))
                print(f"    Taxon={taxon:10s} K={k} → acc={acc:.3f} ({len(eligible)} species)")

        return results

    def _build_taxon_lookup(self, species_ids: List[str]) -> Dict[str, str]:
        """Map species_id -> taxon string.

        Priority:
        1. SPECIES_CATALOG (main 33-species catalog)
        2. supplementary_taxa.json (extra downloaded species)
        3. Name-based heuristic fallback
        """
        lookup: Dict[str, str] = {}

        # 1. Main catalog (SPECIES_CATALOG is {sp_id: SpeciesRecord})
        try:
            from anycall.data.species import SPECIES_CATALOG
            for sp_id, sp_rec in SPECIES_CATALOG.items():
                lookup[sp_id] = sp_rec.taxon.value
        except Exception:
            pass

        # 2. Supplementary JSON (written by download_extra_species.py)
        supp_path = Path("data/supplementary_taxa.json")
        if supp_path.exists():
            try:
                import json
                with open(supp_path) as f:
                    supp = json.load(f)
                lookup.update(supp)
            except Exception:
                pass

        # 3. Heuristic fallback for anything still unmapped
        insecta_keys  = ["gryllus", "teleogryllus", "oecanthus", "mecopoda",
                         "cryptotympana", "purana", "euconocephalus", "gryllotalpa",
                         "cicada", "cricket", "katydid", "locust", "acrid", "hierodula",
                         "conocephalus", "homorocoryphus", "gampsocleis", "valanga",
                         "hieroglyphus", "acheta", "schistocerca"]
        amphibia_keys = ["hoplobatrachus", "duttaphrynus", "polypedates", "euphlyctis",
                         "hydrophylax", "microhyla", "kaloula", "sphaerotheca",
                         "fejervarya", "nyctibatrachus", "uperodon", "minervarya",
                         "raorchestes", "indirana", "nasikabatrachus", "xanthophryne"]
        mammalia_keys = ["funambulus", "macaca", "semnopithecus", "canis", "muntiacus",
                         "elephas", "panthera", "axis", "vulpes", "sus", "boselaphus",
                         "cervus", "rusa", "antilope", "bos", "bubalus", "herpestes",
                         "viverra", "felis", "prionailurus", "melursus", "hystrix"]

        for sp in species_ids:
            if sp in lookup:
                continue
            low = sp.lower()
            if any(k in low for k in amphibia_keys):
                lookup[sp] = "amphibia"
            elif any(k in low for k in insecta_keys):
                lookup[sp] = "insecta"
            elif any(k in low for k in mammalia_keys):
                lookup[sp] = "mammalia"
            else:
                lookup[sp] = "aves"

        return lookup

    # ------------------------------------------------------------------
    # Exp 4: Rejection quality ROC
    # ------------------------------------------------------------------

    def _exp4_roc(
        self,
        backbone_name: str,
        embeddings_map: Dict[str, np.ndarray],
    ) -> List[RocResult]:
        """Sweep θ and compute FAR/TAR for genuine vs. intruder queries."""
        results = []
        species_list = list(embeddings_map.keys())
        if len(species_list) < 3:
            return results

        # Split into enrolled and intruder species
        enrolled_species = species_list[:max(2, len(species_list) // 2)]
        intruder_species = species_list[len(enrolled_species):]
        if not intruder_species:
            intruder_species = enrolled_species[-1:]
            enrolled_species = enrolled_species[:-1]

        k = 5
        clf_base = PrototypicalClassifier(threshold=0.0)
        for sp in enrolled_species:
            embs = embeddings_map[sp]
            if len(embs) > k:
                clf_base.enroll(sp, list(embs[:k]))

        # Collect genuine scores (query from enrolled species)
        genuine_scores = []
        for sp in enrolled_species:
            embs = embeddings_map[sp]
            for e in embs[k:]:
                _, score = clf_base.predict(e)
                genuine_scores.append(score)

        # Collect intruder scores (query from intruder / unknown species)
        intruder_scores = []
        for sp in intruder_species:
            for e in embeddings_map[sp]:
                _, score = clf_base.predict(e)
                intruder_scores.append(score)

        if not genuine_scores or not intruder_scores:
            return results

        for theta in self._thresholds:
            # TAR: fraction of genuine queries correctly accepted (score >= θ)
            tar = sum(1 for s in genuine_scores if s >= theta) / len(genuine_scores)
            # FAR: fraction of intruder queries incorrectly accepted (score >= θ)
            far = sum(1 for s in intruder_scores if s >= theta) / len(intruder_scores)
            results.append(RocResult(
                backbone=backbone_name,
                threshold=round(theta, 4),
                far=round(far, 4),
                tar=round(tar, 4),
            ))

        # EER approximation
        eer_theta, eer_val = self._compute_eer(results[-len(self._thresholds):])
        print(f"    ROC: EER ≈ {eer_val:.3f} at θ={eer_theta:.3f}")
        return results

    @staticmethod
    def _compute_eer(roc_rows: List[RocResult]) -> Tuple[float, float]:
        """Return (threshold, EER) by finding the crossing of FAR and FRR."""
        best_diff = float("inf")
        eer_theta = 0.5
        eer_val = 0.5
        for r in roc_rows:
            frr = 1.0 - r.tar
            diff = abs(r.far - frr)
            if diff < best_diff:
                best_diff = diff
                eer_theta = r.threshold
                eer_val = (r.far + frr) / 2.0
        return eer_theta, eer_val

    # ------------------------------------------------------------------
    # Exp 5: Latency profiling
    # ------------------------------------------------------------------

    def _exp5_latency(
        self,
        backbone_name: str,
        backbone: BaseAudioEmbeddingBackbone,
        species_map: Dict[str, List[Path]],
    ) -> Optional[LatencyResult]:
        """Time backbone.embed() over N trials using real audio clips."""
        wavs: List[Path] = []
        for paths in species_map.values():
            wavs.extend(paths[:2])
        if not wavs:
            return None

        latencies_ms = []
        for i in range(self._n_latency_trials):
            wav = wavs[i % len(wavs)]
            try:
                gc.collect()
                t0 = time.perf_counter()
                backbone.embed(wav)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
            except Exception:
                continue

        if not latencies_ms:
            return None

        result = LatencyResult(
            backbone=backbone_name,
            mean_ms=round(float(np.mean(latencies_ms)), 2),
            std_ms=round(float(np.std(latencies_ms)), 2),
            min_ms=round(float(np.min(latencies_ms)), 2),
            max_ms=round(float(np.max(latencies_ms)), 2),
            n_trials=len(latencies_ms),
            embedding_dim=backbone.embedding_dim,
        )
        print(f"    Latency: {result.mean_ms:.1f} ± {result.std_ms:.1f} ms "
              f"(min={result.min_ms:.1f}, max={result.max_ms:.1f})")
        return result

    # ------------------------------------------------------------------
    # Exp 1: BirdNET stock failure analysis
    # ------------------------------------------------------------------

    def _exp1_birdnet_failure(
        self, species_map: Dict[str, List[Path]]
    ) -> List[BirdNetFailureResult]:
        """Compare stock BirdNET softmax top-1 vs AnyCall prototypical on Indian species."""
        results = []
        taxon_of = self._build_taxon_lookup(list(species_map.keys()))

        # Load BirdNET with logits access
        try:
            from anycall.embeddings.birdnet import BirdNetBackbone
            birdnet = BirdNetBackbone()
        except Exception as exc:
            print(f"  [WARN] BirdNET not available for Exp 1: {exc}")
            return results

        # Build AnyCall classifier using BirdNET embeddings
        clf = PrototypicalClassifier(threshold=0.5)
        k = 5
        embs_map: Dict[str, np.ndarray] = {}
        for sp, wavs in species_map.items():
            vecs = []
            for wav in wavs[:k + 5]:
                try:
                    vec = birdnet.embed(wav)
                    vecs.append(vec)
                except Exception:
                    continue
            if len(vecs) >= k + 1:
                embs_map[sp] = np.stack(vecs)
                clf.enroll(sp, list(embs_map[sp][:k]))

        for sp, embs in embs_map.items():
            taxon = taxon_of.get(sp, "aves")
            for e in embs[k:k + 3]:  # test on 3 held-out clips
                # AnyCall prediction
                anycall_pred, anycall_score = clf.predict(e)
                anycall_correct = anycall_pred == sp

                # Stock BirdNET: softmax top-1 (if logits available)
                try:
                    wav = species_map[sp][k]
                    _, logits = birdnet.extract_with_logits(wav)
                    if logits is not None:
                        top_idx = int(np.argmax(logits))
                        birdnet_score = float(logits[top_idx])
                        birdnet_top = f"class_{top_idx}"
                        # BirdNET only covers 6522 bird species — non-bird taxa always fail
                        birdnet_correct = (taxon == "aves") and (birdnet_score > 0.5)
                    else:
                        birdnet_score, birdnet_top, birdnet_correct = 0.0, "N/A", False
                except Exception:
                    birdnet_score, birdnet_top, birdnet_correct = 0.0, "N/A", False

                results.append(BirdNetFailureResult(
                    species_label=sp,
                    taxon=taxon,
                    birdnet_correct=birdnet_correct,
                    anycall_correct=anycall_correct,
                    birdnet_top_prediction=birdnet_top,
                    birdnet_score=round(birdnet_score, 4),
                    anycall_score=round(anycall_score, 4),
                ))

        if results:
            birdnet_acc = np.mean([r.birdnet_correct for r in results])
            anycall_acc = np.mean([r.anycall_correct for r in results])
            print(f"    BirdNET stock top-1: {birdnet_acc:.3f} | AnyCall: {anycall_acc:.3f}")

        return results

    # ------------------------------------------------------------------
    # Summary & I/O
    # ------------------------------------------------------------------

    def _build_summary(self, report: BenchmarkReport) -> Dict:
        summary: Dict = {}

        # Few-shot mean accuracy per backbone × K
        if report.exp2_few_shot:
            for row in report.exp2_few_shot:
                key = f"{row['backbone']}_K{row['k_shots']}_mean_acc"
                summary.setdefault(key, []).append(row["top1_accuracy"])
            summary = {k: round(float(np.mean(v)), 4) if isinstance(v, list) else v
                       for k, v in summary.items()}

        # Latency summary
        for row in report.exp5_latency:
            summary[f"{row['backbone']}_latency_mean_ms"] = row["mean_ms"]

        # BirdNET failure
        if report.exp1_birdnet_failure:
            birdnet_acc = np.mean([r["birdnet_correct"] for r in report.exp1_birdnet_failure])
            anycall_acc = np.mean([r["anycall_correct"] for r in report.exp1_birdnet_failure])
            summary["exp1_birdnet_stock_accuracy"] = round(float(birdnet_acc), 4)
            summary["exp1_anycall_accuracy"] = round(float(anycall_acc), 4)

        return summary

    def _write_outputs(
        self,
        report: BenchmarkReport,
        output_json: str | Path,
        output_csv: str | Path,
    ) -> None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(output_json, "w") as f:
            json.dump(asdict(report), f, indent=2)

        # CSV (flattened few-shot rows — main table for the paper)
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = report.exp2_few_shot or []
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
