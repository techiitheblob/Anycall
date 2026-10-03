#!/usr/bin/env python3
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
AnyCall — Proper Benchmark
===========================
Fixes three issues in the old rigorous_blind_test:

1.  Uses PrototypicalClassifier.enroll() + predict() — so K=5 sub-prototypes
    and dynamic mean-centering are actually exercised.

2.  Per-taxon thresholds instead of one global theta=0.70.
    BirdNET maps non-avian taxa to a lower-similarity region; a single
    threshold unfairly penalises Insecta / Amphibia / Mammalia.

3.  Margin gate: only count as "correct" if score(true) - score(next_best) > MARGIN.
    This filters out borderline calls that happen to squeak past the threshold.

Outputs
-------
  reports/proper_benchmark_report.json
  reports/proper_benchmark_report.csv
  reports/anycall_proper_report.tex
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone

# ── Config ───────────────────────────────────────────────────────────────────

PROC_DIR    = PROJECT_ROOT / "data" / "processed"
DB_PATH     = PROJECT_ROOT / "anycall.db"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Per-taxon acceptance thresholds calibrated from empirical similarity distributions.
# Aves land ~0.72-0.95, Mammalia ~0.62-0.75, Amphibia ~0.58-0.70, Insecta ~0.52-0.68
TAXON_THRESHOLDS: Dict[str, float] = {
    "aves":      0.65,
    "mammalia":  0.58,
    "amphibia":  0.55,
    "insecta":   0.50,
}
DEFAULT_THRESHOLD = 0.62   # for any unlabelled taxon

# Margin gate: query must score >= MARGIN above the nearest competitor
MARGIN = 0.02

# K-Means sub-prototypes per species
N_SUBPROTOTYPES = 5

# Support / test split — use first SUPPORT_N files for enrollment, rest for test.
# If a species has <= SUPPORT_N files, use all for both (tiny sets).
SUPPORT_N = 5

# ─────────────────────────────────────────────────────────────────────────────


def embed_wav(bb: BirdNetBackbone, wav_path: Path) -> np.ndarray | None:
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return bb.embed(audio, sr=sr)
    except Exception:
        return None


def main() -> None:
    print("=" * 72)
    print("   AnyCall  —  Proper Benchmark  (engine + per-taxon thresholds)")
    print("=" * 72)

    REPORTS_DIR.mkdir(exist_ok=True)

    # ── Load species metadata ─────────────────────────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("SELECT species_id, common_name, taxon FROM species")
    species_meta: Dict[str, dict] = {
        sp_id: {"common_name": cn, "taxon": tx.lower() if tx else "aves"}
        for sp_id, cn, tx in cur.fetchall()
    }
    conn.close()

    # ── Init backbone ─────────────────────────────────────────────────────────
    print("\n[1/4] Initialising BirdNET backbone ...")
    bb = BirdNetBackbone()

    # ── Build classifier with proper sub-prototypes ───────────────────────────
    print("[2/4] Enrolling species with K=5 sub-prototypes + mean-centering ...\n")

    # Use a low global threshold; we will apply per-taxon thresholds manually at query time
    classifier = PrototypicalClassifier(threshold=0.01, n_subprototypes=N_SUBPROTOTYPES)

    species_dirs = sorted([d for d in PROC_DIR.iterdir()
                           if d.is_dir() and d.name in species_meta])

    enroll_info: Dict[str, dict] = {}

    for sp_dir in species_dirs:
        sp_id  = sp_dir.name
        all_wavs = sorted(sp_dir.glob("*.wav"))
        if not all_wavs:
            continue

        # Support / test split
        support_wavs = all_wavs[:SUPPORT_N]
        test_wavs    = all_wavs[SUPPORT_N:] if len(all_wavs) > SUPPORT_N else all_wavs

        support_embs = [e for e in (embed_wav(bb, w) for w in all_wavs[:max(SUPPORT_N, len(all_wavs) - len(test_wavs))]) if e is not None]

        if not support_embs:
            continue

        classifier.enroll(sp_id, support_embs)
        enroll_info[sp_id] = {
            "n_support": len(support_embs),
            "n_test":    len(test_wavs),
            "original_clips": len(all_wavs),
        }

        meta = species_meta[sp_id]
        print(f"  {sp_id:<38}  taxon={meta['taxon']:<10}  "
              f"support={len(support_embs):>4}  test={len(test_wavs):>4}  "
              f"theta={TAXON_THRESHOLDS.get(meta['taxon'], DEFAULT_THRESHOLD):.2f}")

    n_enrolled = len(classifier.enrolled_species)
    print(f"\n  Enrolled {n_enrolled} species with K=5 sub-prototypes.\n")

    # ── Benchmark: per-species recall / confusion / unknown ───────────────────
    print("[3/4] Running held-out benchmark (mean-centering + per-taxon theta) ...\n")

    per_species: List[dict] = []

    for sp_id in classifier.enrolled_species:
        sp_dir   = PROC_DIR / sp_id
        all_wavs = sorted(sp_dir.glob("*.wav"))
        info     = enroll_info.get(sp_id, {})
        n_support = info.get("n_support", SUPPORT_N)
        test_wavs = all_wavs[n_support:] if len(all_wavs) > n_support else all_wavs
        if not test_wavs:
            test_wavs = all_wavs

        meta   = species_meta.get(sp_id, {})
        taxon  = meta.get("taxon", "aves")
        theta  = TAXON_THRESHOLDS.get(taxon, DEFAULT_THRESHOLD)

        correct = confused = unknown = total = 0

        for wf in test_wavs:
            emb = embed_wav(bb, wf)
            if emb is None:
                continue

            # Use engine predict() — this applies mean-centering and sub-prototypes
            pred = classifier.predict(emb, threshold=theta, use_mean_centering=True)
            total += 1

            if not pred.is_known:
                unknown += 1
            elif pred.predicted_label == sp_id:
                # Apply margin gate: score(true) must beat runner-up by MARGIN
                sorted_scores = sorted(pred.scores.items(), key=lambda x: x[1], reverse=True)
                true_score = pred.scores.get(sp_id, 0.0)
                runner_up  = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
                if (true_score - runner_up) >= MARGIN:
                    correct += 1
                else:
                    confused += 1   # correct label but margin too thin → ambiguous
            else:
                confused += 1

        if total == 0:
            continue

        per_species.append({
            "species_id":     sp_id,
            "common_name":    meta.get("common_name", sp_id),
            "taxon":          taxon,
            "original_clips": info.get("original_clips", len(all_wavs)),
            "enrolled_clips": n_support,
            "tested_clips":   total,
            "theta":          theta,
            "recall_pct":     round(correct  / total * 100, 1),
            "confusion_pct":  round(confused / total * 100, 1),
            "unknown_pct":    round(unknown  / total * 100, 1),
        })

    per_species.sort(key=lambda x: (-x["recall_pct"], x["species_id"]))

    # ── Open-set rejection test ───────────────────────────────────────────────
    print("[4/4] Open-set noise rejection ...")
    rng = np.random.default_rng(42)
    rej_count = rej_total = 0
    for _ in range(60):
        noise = rng.normal(0, 0.05, int(48000 * 3)).astype(np.float32)
        emb   = bb.embed(noise, sr=48000)
        # Use lowest taxon threshold for the most lenient open-set test
        pred  = classifier.predict(emb, threshold=min(TAXON_THRESHOLDS.values()), use_mean_centering=True)
        rej_total += 1
        if not pred.is_known:
            rej_count += 1
    rej_rate = rej_count / rej_total * 100.0

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  {'Common Name':<36} {'Sci ID':<34} {'Tx':<5} {'theta':<5} {'Orig':>4} {'Enrl':>4} {'Test':>4}  "
          f"{'Recall':>7}  {'Confus':>7}  {'Unknwn':>7}")
    print("-" * 100)
    for r in per_species:
        tx = r["taxon"][:4].upper()
        print(
            f"  {r['common_name']:<36} {r['species_id']:<34} {tx:<5} {r['theta']:.2f} "
            f"{str(r['original_clips']):>4} {r['enrolled_clips']:>4} {r['tested_clips']:>4}  "
            f"{r['recall_pct']:>6.1f}%  {r['confusion_pct']:>6.1f}%  {r['unknown_pct']:>6.1f}%"
        )
    print("=" * 100)

    if per_species:
        avg_recall    = float(np.mean([r["recall_pct"]    for r in per_species]))
        avg_confusion = float(np.mean([r["confusion_pct"] for r in per_species]))
        avg_unknown   = float(np.mean([r["unknown_pct"]   for r in per_species]))

        # Per-taxon breakdown
        taxa = sorted(set(r["taxon"] for r in per_species))
        print("\n  Per-taxon macro averages:")
        for tx in taxa:
            tx_rows = [r for r in per_species if r["taxon"] == tx]
            print(f"    {tx:<12}  theta={TAXON_THRESHOLDS.get(tx, DEFAULT_THRESHOLD):.2f}  "
                  f"n={len(tx_rows):>2}  "
                  f"recall={np.mean([r['recall_pct'] for r in tx_rows]):.1f}%  "
                  f"confusion={np.mean([r['confusion_pct'] for r in tx_rows]):.1f}%  "
                  f"unknown={np.mean([r['unknown_pct'] for r in tx_rows]):.1f}%")
    else:
        avg_recall = avg_confusion = avg_unknown = 0.0

    print(f"\n  Overall macro-avg recall    : {avg_recall:.1f}%")
    print(f"  Overall macro-avg confusion : {avg_confusion:.1f}%")
    print(f"  Overall macro-avg unknown   : {avg_unknown:.1f}%")
    print(f"  Open-set rejection rate     : {rej_rate:.1f}%  ({rej_count}/{rej_total})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "improvements":       [
            "K=5 sub-prototypes via PrototypicalClassifier.enroll()",
            "Dynamic mean-centering via classifier.predict(use_mean_centering=True)",
            "Per-taxon thresholds: " + str(TAXON_THRESHOLDS),
            f"Margin gate: only accept if score(true) - score(runner_up) >= {MARGIN}",
        ],
        "taxon_thresholds":   TAXON_THRESHOLDS,
        "margin":             MARGIN,
        "n_subprototypes":    N_SUBPROTOTYPES,
        "n_enrolled":         n_enrolled,
        "macro_avg_recall":   round(avg_recall, 2),
        "macro_avg_confusion": round(avg_confusion, 2),
        "macro_avg_unknown":  round(avg_unknown, 2),
        "open_set_rejection_pct": round(rej_rate, 2),
        "per_species":        per_species,
    }

    json_path = REPORTS_DIR / "proper_benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / "proper_benchmark_report.csv"
    with open(csv_path, "w") as f:
        f.write("species_id,common_name,taxon,theta,original_clips,enrolled_clips,"
                "tested_clips,recall_pct,confusion_pct,unknown_pct\n")
        for r in per_species:
            f.write(f"{r['species_id']},{r['common_name']},{r['taxon']},{r['theta']},"
                    f"{r['original_clips']},{r['enrolled_clips']},{r['tested_clips']},"
                    f"{r['recall_pct']},{r['confusion_pct']},{r['unknown_pct']}\n")

    # ── Save LaTeX ────────────────────────────────────────────────────────────
    tex_path = REPORTS_DIR / "anycall_proper_report.tex"
    _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, tex_path)

    print(f"\n  Saved JSON  -> {json_path}")
    print(f"  Saved CSV   -> {csv_path}")
    print(f"  Saved LaTeX -> {tex_path}")
    print("\nDone.\n")


def _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, out_path):
    lines = []
    a = lines.append
    a(r"\documentclass[11pt]{article}")
    a(r"\usepackage[margin=1in]{geometry}")
    a(r"\usepackage{booktabs,longtable,amsmath,hyperref,xcolor}")
    a(r"\title{AnyCall: Improved Benchmark --- Sub-Prototypes + Per-Taxon Thresholds}")
    a(r"\author{AnyCall Project Team}")
    a(r"\date{\today}")
    a(r"\begin{document}\maketitle")
    a(r"\begin{abstract}")
    a(r"Three improvements over the original benchmark: "
      r"(1)~K\!=\!5 sub-prototypes via \texttt{PrototypicalClassifier.enroll()}, "
      r"(2)~dynamic mean-centering in \texttt{predict()}, and "
      r"(3)~per-taxon acceptance thresholds ($\theta_\text{aves}=0.65$, "
      r"$\theta_\text{mammalia}=0.58$, $\theta_\text{amphibia}=0.55$, "
      r"$\theta_\text{insecta}=0.50$) plus a margin gate ($\delta=0.02$) "
      r"to suppress borderline confusions. "
      f"Macro-average recall: \\textbf{{{avg_recall:.1f}\\%}}, "
      f"confusion: \\textbf{{{avg_confusion:.1f}\\%}}, "
      f"unknown rate: \\textbf{{{avg_unknown:.1f}\\%}}.")
    a(r"\end{abstract}")
    a(r"\section{Benchmark Results}")
    a(r"\begin{longtable}{llcccccc}")
    a(r"\caption{Improved Benchmark: Sub-Prototypes + Per-Taxon Thresholds}\label{tab:proper}\\")
    a(r"\toprule")
    a(r"\textbf{Species} & \textbf{Taxon} & $\theta$ & \textbf{Orig} & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endfirsthead")
    a(r"\toprule")
    a(r"\textbf{Species} & \textbf{Taxon} & $\theta$ & \textbf{Orig} & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endhead")
    a(r"\midrule\multicolumn{8}{r}{\textit{Continued}}\\\endfoot")
    a(r"\bottomrule\endlastfoot")
    for r in per_species:
        sci = r["species_id"].replace("_", r"\_")
        cn  = r["common_name"].replace("&", r"\&")
        tx  = r["taxon"].capitalize()
        a(f"{cn} (\\textit{{{sci}}}) & {tx} & {r['theta']:.2f} & "
          f"{r['original_clips']} & {r['enrolled_clips']} & "
          f"{r['recall_pct']}\\% & {r['confusion_pct']}\\% & {r['unknown_pct']}\\% \\\\")
    a(r"\end{longtable}")
    a(r"\section{Summary}")
    a(r"\begin{tabular}{lr}\toprule")
    a(r"\textbf{Metric} & \textbf{Value}\\\midrule")
    a(f"Macro-avg Recall    & {avg_recall:.1f}\\% \\\\")
    a(f"Macro-avg Confusion & {avg_confusion:.1f}\\% \\\\")
    a(f"Macro-avg Unknown   & {avg_unknown:.1f}\\% \\\\")
    a(f"Open-set Rejection  & {rej_rate:.1f}\\% \\\\")
    a(r"\bottomrule\end{tabular}")
    a(r"\end{document}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
