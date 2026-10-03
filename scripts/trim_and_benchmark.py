#!/usr/bin/env python3
"""
AnyCall — Trim-to-50 Benchmark Script
======================================
For each enrolled species, keep only the 50 embeddings closest to the
(sub-)cluster centroid, then re-run the recall / confusion / unknown test.

Strategy
--------
* Species WITHOUT subclustering  -> keep 50 clips closest to global centroid.
* Species WITH  subclustering (K=5 sub-prototypes) -> keep 50 clips per
  sub-cluster (250 clips max), each chosen by proximity to that sub-centroid.

Output
------
* Prints per-species recall / confusion / unknown rate table.
* Saves reports/trim50_benchmark_report.json and .csv.
* Generates reports/anycall_trim50_report.tex with the updated LaTeX table.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone

# Configuration
DB_PATH       = PROJECT_ROOT / "anycall.db"
PROC_DIR      = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
N_KEEP        = 50
N_SUBCLUSTERS = 5
THRESHOLD     = 0.70
HELD_OUT_START = 3


def l2(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.ones_like(v) / float(np.sqrt(len(v)))
    return v / norm


def kmeans_l2(stack: np.ndarray, k: int, max_iter: int = 80, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(stack), size=k, replace=False)
    centroids = stack[idx].copy()
    for _ in range(max_iter):
        sims = stack @ centroids.T
        assign = np.argmax(sims, axis=1)
        new_c = np.zeros_like(centroids)
        changed = False
        for ci in range(k):
            members = stack[assign == ci]
            if len(members) == 0:
                new_c[ci] = centroids[ci]
            else:
                mv = members.mean(axis=0)
                norm = np.linalg.norm(mv)
                nc = mv / norm if norm > 1e-12 else centroids[ci]
                if not np.allclose(nc, centroids[ci], atol=1e-6):
                    changed = True
                new_c[ci] = nc
        centroids = new_c
        if not changed:
            break
    return centroids


def embed_dir(bb: BirdNetBackbone, sp_dir: Path, limit: int | None = None) -> List[np.ndarray]:
    wavs = sorted(sp_dir.glob("*.wav"))
    if limit is not None:
        wavs = wavs[:limit]
    out: List[np.ndarray] = []
    for wf in wavs:
        try:
            audio, sr = sf.read(str(wf), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            out.append(bb.embed(audio, sr=sr))
        except Exception:
            pass
    return out


def select_trimmed(embs: List[np.ndarray], n_keep: int, k_sub: int) -> Tuple[List[np.ndarray], int]:
    if not embs:
        return [], 0
    stack = np.stack([l2(e) for e in embs], axis=0)
    if len(stack) <= n_keep:
        return list(stack), 1

    use_sub = len(stack) >= k_sub * 2
    if not use_sub:
        centroid = l2(stack.mean(axis=0))
        sims = stack @ centroid
        top_idx = np.argsort(sims)[::-1][:n_keep]
        return [stack[i] for i in top_idx], 1

    centroids = kmeans_l2(stack, k=k_sub)
    assign = np.argmax(stack @ centroids.T, axis=1)
    kept: List[np.ndarray] = []
    for ci in range(k_sub):
        midx = np.where(assign == ci)[0]
        if len(midx) == 0:
            continue
        sub_sims = stack[midx] @ centroids[ci]
        top_local = np.argsort(sub_sims)[::-1][:n_keep]
        for li in top_local:
            kept.append(stack[midx[li]])
    return kept, k_sub


def main() -> None:
    print("=" * 72)
    print("   AnyCall  Trim-to-50 Benchmark  (Recall / Confusion / Unknown)")
    print("=" * 72)

    REPORTS_DIR.mkdir(exist_ok=True)

    # Load DB metadata
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("SELECT species_id, common_name, taxon FROM species")
    species_meta = {sp_id: {"common_name": cn, "taxon": tx} for sp_id, cn, tx in cur.fetchall()}
    conn.close()

    print("\n[1/4] Initialising BirdNET backbone ...")
    bb = BirdNetBackbone()

    print("[2/4] Embedding, trimming, and enrolling species ...\n")
    classifier = PrototypicalClassifier(threshold=THRESHOLD, n_subprototypes=N_SUBCLUSTERS)
    enroll_stats: Dict[str, dict] = {}

    species_dirs = sorted([d for d in PROC_DIR.iterdir() if d.is_dir() and d.name in species_meta])

    for sp_dir in species_dirs:
        sp_id = sp_dir.name
        all_wavs = sorted(sp_dir.glob("*.wav"))
        n_test = max(1, len(all_wavs) - HELD_OUT_START)
        n_support_pool = len(all_wavs) - n_test

        # Embed the support pool (all clips except the held-out test tail)
        support_embs = embed_dir(bb, sp_dir, limit=n_support_pool if n_support_pool > 0 else None)
        test_wavs    = all_wavs[n_support_pool:] if n_support_pool < len(all_wavs) else all_wavs

        if not support_embs:
            print(f"  [SKIP] {sp_id} — no embeddable audio")
            continue

        trimmed, actual_k = select_trimmed(support_embs, N_KEEP, N_SUBCLUSTERS)
        if not trimmed:
            print(f"  [SKIP] {sp_id} — trim empty")
            continue

        classifier.enroll(sp_id, trimmed)
        enroll_stats[sp_id] = {
            "original_clips": len(all_wavs),
            "support_pool":   len(support_embs),
            "enrolled":       len(trimmed),
            "subclusters":    actual_k,
            "test_wavs":      [w.name for w in test_wavs],
        }
        print(
            f"  {sp_id:<38}  original={len(all_wavs):>4}  "
            f"pool={len(support_embs):>4}  enrolled={len(trimmed):>4}  k={actual_k}"
        )

    print(f"\n  Enrolled {len(classifier.enrolled_species)} species.\n")

    # Benchmark
    print("[3/4] Running held-out recall / confusion / unknown benchmark ...\n")
    per_species: List[dict] = []

    for sp_id in classifier.enrolled_species:
        sp_dir   = PROC_DIR / sp_id
        stats    = enroll_stats.get(sp_id, {})
        all_wavs = sorted(sp_dir.glob("*.wav"))
        n_pool   = stats.get("support_pool", HELD_OUT_START)
        test_wavs = all_wavs[n_pool:] if n_pool < len(all_wavs) else all_wavs
        if not test_wavs:
            test_wavs = all_wavs

        correct = confused = unknown = total = 0
        for wf in test_wavs:
            try:
                audio, sr = sf.read(str(wf), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                emb  = bb.embed(audio, sr=sr)
                pred = classifier.predict(emb, threshold=THRESHOLD)
                total += 1
                if not pred.is_known:
                    unknown += 1
                elif pred.predicted_label == sp_id:
                    correct += 1
                else:
                    confused += 1
            except Exception:
                pass

        if total == 0:
            continue

        meta = species_meta.get(sp_id, {})
        per_species.append({
            "species_id":     sp_id,
            "common_name":    meta.get("common_name", sp_id),
            "taxon":          meta.get("taxon", ""),
            "original_clips": stats.get("original_clips", "?"),
            "enrolled_clips": stats.get("enrolled", "?"),
            "tested_clips":   total,
            "recall_pct":     round(correct  / total * 100, 1),
            "confusion_pct":  round(confused / total * 100, 1),
            "unknown_pct":    round(unknown  / total * 100, 1),
        })

    per_species.sort(key=lambda x: (-x["recall_pct"], x["species_id"]))

    # Open-set rejection
    print("[4/4] Open-set noise rejection test ...")
    rng = np.random.default_rng(42)
    rej_count = rej_total = 0
    for _ in range(40):
        noise = rng.normal(0, 0.05, int(48000 * 3)).astype(np.float32)
        pred  = classifier.predict(bb.embed(noise, sr=48000), threshold=THRESHOLD)
        rej_total += 1
        if not pred.is_known:
            rej_count += 1
    rej_rate = rej_count / rej_total * 100.0

    # Print table
    print("\n" + "=" * 95)
    header = f"{'Common Name':<36} {'Sci ID':<34} {'Orig':>4} {'Enrl':>4} {'Test':>4}  {'Recall':>7}  {'Confus':>7}  {'Unknwn':>7}"
    print(header)
    print("-" * 95)
    for r in per_species:
        print(
            f"{r['common_name']:<36} {r['species_id']:<34} "
            f"{str(r['original_clips']):>4} {str(r['enrolled_clips']):>4} "
            f"{r['tested_clips']:>4}  "
            f"{r['recall_pct']:>6.1f}%  "
            f"{r['confusion_pct']:>6.1f}%  "
            f"{r['unknown_pct']:>6.1f}%"
        )
    print("=" * 95)

    avg_recall    = float(np.mean([r["recall_pct"]    for r in per_species])) if per_species else 0
    avg_confusion = float(np.mean([r["confusion_pct"] for r in per_species])) if per_species else 0
    avg_unknown   = float(np.mean([r["unknown_pct"]   for r in per_species])) if per_species else 0

    print(f"\n  Macro-avg recall    : {avg_recall:.1f}%")
    print(f"  Macro-avg confusion : {avg_confusion:.1f}%")
    print(f"  Macro-avg unknown   : {avg_unknown:.1f}%")
    print(f"  Open-set rejection  : {rej_rate:.1f}%  ({rej_count}/{rej_total} rejected)")

    # Save JSON
    report = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy":         f"trim_to_{N_KEEP}_per_subcentroid",
        "n_keep":           N_KEEP,
        "n_subclusters":    N_SUBCLUSTERS,
        "threshold":        THRESHOLD,
        "n_enrolled":       len(classifier.enrolled_species),
        "macro_avg_recall": round(avg_recall, 2),
        "macro_avg_confusion": round(avg_confusion, 2),
        "macro_avg_unknown": round(avg_unknown, 2),
        "open_set_rejection_pct": round(rej_rate, 2),
        "per_species":      per_species,
    }
    json_path = REPORTS_DIR / "trim50_benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved JSON  -> {json_path}")

    # Save CSV
    csv_path = REPORTS_DIR / "trim50_benchmark_report.csv"
    with open(csv_path, "w") as f:
        f.write("species_id,common_name,taxon,original_clips,enrolled_clips,"
                "tested_clips,recall_pct,confusion_pct,unknown_pct\n")
        for r in per_species:
            f.write(
                f"{r['species_id']},{r['common_name']},{r['taxon']},"
                f"{r['original_clips']},{r['enrolled_clips']},{r['tested_clips']},"
                f"{r['recall_pct']},{r['confusion_pct']},{r['unknown_pct']}\n"
            )
    print(f"  Saved CSV   -> {csv_path}")

    # Generate LaTeX
    tex_path = REPORTS_DIR / "anycall_trim50_report.tex"
    _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, tex_path)
    print(f"  Saved LaTeX -> {tex_path}")
    print("\nDone.\n")


def _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, out_path):
    lines = []
    a = lines.append
    a(r"\documentclass[11pt]{article}")
    a(r"\usepackage[margin=1in]{geometry}")
    a(r"\usepackage{booktabs}")
    a(r"\usepackage{hyperref}")
    a(r"\usepackage{longtable}")
    a(r"\usepackage{amsmath}")
    a(r"\title{AnyCall: Trim-to-50 Few-Shot Benchmark Report}")
    a(r"\author{AnyCall Project Team}")
    a(r"\date{\today}")
    a(r"\begin{document}")
    a(r"\maketitle")
    a(r"\begin{abstract}")
    a(
        r"Each species' support set was pruned to the "
        r"\textbf{50 embeddings nearest their (sub-)cluster centroid}. "
        r"For species using $K=5$ dynamic sub-clustering, up to $50\times5=250$ clips "
        r"are retained. "
        f"Macro-average recall: \\textbf{{{avg_recall:.1f}\\%}}, "
        f"confusion: \\textbf{{{avg_confusion:.1f}\\%}}, "
        f"unknown rate: \\textbf{{{avg_unknown:.1f}\\%}}."
    )
    a(r"\end{abstract}")
    a(r"\section{Benchmark Results}")
    a(r"\begin{longtable}{llccccc}")
    a(r"\caption{Trim-to-50 Results}\label{tab:trim50}\\")
    a(r"\toprule")
    a(r"\textbf{Species} & \textbf{Sci. ID} & \textbf{Orig} & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endfirsthead")
    a(r"\toprule")
    a(r"\textbf{Species} & \textbf{Sci. ID} & \textbf{Orig} & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endhead")
    a(r"\midrule\multicolumn{7}{r}{\textit{Continued}}\\\endfoot")
    a(r"\bottomrule\endlastfoot")
    for r in per_species:
        sci = r["species_id"].replace("_", r"\_")
        cn  = r["common_name"].replace("&", r"\&")
        a(f"{cn} & \\textit{{{sci}}} & {r['original_clips']} & {r['enrolled_clips']} & "
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
