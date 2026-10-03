#!/usr/bin/env python3
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
AnyCall Benchmark v2
====================
Lessons from v1 failure:
 - SUPPORT_N=5 is degenerate for species with 100-1000 clips.
   The original DB centroid was computed from ~100 clips.
 - K=5 sub-prototypes from 5 clips = 1 clip per cluster = garbage.
 - Mean-centering shifts similarity distribution; per-taxon thresholds
   were calibrated without it, so everything got rejected (75.9% unknown).

This version:
 - 90 / 10 split: first 90% of clips -> support, last 10% -> test.
   Gives stable centroid AND proper held-out evaluation.
 - K sub-prototypes only when support >= K*10 (avoids degenerate clusters).
 - NO mean-centering (original did not use it; calibrate separately later).
 - Per-taxon thresholds, empirically chosen WITHOUT mean-centering:
     aves=0.70, mammalia=0.62, amphibia=0.58, insecta=0.52
 - Margin gate delta=0.02 to suppress borderline confusions.

Outputs: reports/benchmark_v2_report.{json,csv,tex}
"""

import json, sqlite3, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anycall.classifier.engine import PrototypicalClassifier
from anycall.embeddings.birdnet import BirdNetBackbone

PROC_DIR    = PROJECT_ROOT / "data" / "processed"
DB_PATH     = PROJECT_ROOT / "anycall.db"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Per-taxon thresholds (NO mean-centering; raw cosine sims vs prototype)
# Calibrated from empirical score distributions per taxon in BirdNET space
TAXON_THRESHOLDS: Dict[str, float] = {
    "aves":      0.70,   # original value — birds land well here
    "mammalia":  0.62,   # mammals score ~0.63-0.75 for true matches
    "amphibia":  0.58,   # frogs/toads score ~0.58-0.70
    "insecta":   0.52,   # insects score ~0.52-0.68 (BirdNET corner)
}
DEFAULT_THRESHOLD = 0.65
MARGIN = 0.02            # score(true) - score(runner-up) must exceed this

# Sub-clustering: only kick in when support set is large enough
MIN_CLIPS_FOR_K_SUB = 15   # need >= 15 clips to bother with K=5
N_SUBPROTOTYPES     = 5


def embed_wav(bb, path: Path) -> np.ndarray | None:
    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return bb.embed(audio, sr=sr)
    except Exception:
        return None


def main() -> None:
    print("=" * 72)
    print("   AnyCall Benchmark v2  (90/10 split, per-taxon theta, no MC)")
    print("=" * 72)

    REPORTS_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    species_meta: Dict[str, dict] = {
        sp_id: {"common_name": cn, "taxon": (tx or "aves").lower()}
        for sp_id, cn, tx in conn.execute("SELECT species_id, common_name, taxon FROM species").fetchall()
    }
    conn.close()

    print("\n[1/4] Initialising BirdNET backbone ...")
    bb = BirdNetBackbone()

    print("[2/4] Embedding support sets (90% of clips) and enrolling ...\n")

    # Disable built-in threshold (apply per-taxon manually); enable K subclustering
    classifier = PrototypicalClassifier(threshold=0.01, n_subprototypes=N_SUBPROTOTYPES)
    enroll_info: Dict[str, dict] = {}

    species_dirs = sorted([d for d in PROC_DIR.iterdir()
                           if d.is_dir() and d.name in species_meta])

    for sp_dir in species_dirs:
        sp_id    = sp_dir.name
        all_wavs = sorted(sp_dir.glob("*.wav"))
        if not all_wavs:
            continue

        n_total   = len(all_wavs)
        # 90/10 split: at least 1 clip for test, at least 1 for support
        n_support = max(1, n_total - max(1, n_total // 10))
        support_wavs = all_wavs[:n_support]
        test_wavs    = all_wavs[n_support:]
        if not test_wavs:
            test_wavs = all_wavs   # tiny sets: no choice

        # Embed support set
        support_embs = [e for e in (embed_wav(bb, w) for w in support_wavs) if e is not None]
        if not support_embs:
            print(f"  [SKIP] {sp_id}: no embeddable support audio")
            continue

        # Disable sub-clustering for tiny support sets (avoids degenerate clusters)
        use_sub = len(support_embs) >= MIN_CLIPS_FOR_K_SUB
        classifier.enroll(sp_id, support_embs, use_subclustering=use_sub)

        meta  = species_meta[sp_id]
        theta = TAXON_THRESHOLDS.get(meta["taxon"], DEFAULT_THRESHOLD)
        k_eff = N_SUBPROTOTYPES if use_sub else 1

        enroll_info[sp_id] = {
            "n_support": len(support_embs),
            "n_test":    len(test_wavs),
            "original_clips": n_total,
            "theta":     theta,
            "k":         k_eff,
        }

        print(f"  {sp_id:<38}  taxon={meta['taxon']:<10}  "
              f"support={len(support_embs):>4}  test={len(test_wavs):>4}  "
              f"theta={theta:.2f}  k_sub={k_eff}")

    print(f"\n  Enrolled {len(classifier.enrolled_species)} species.\n")

    # ── Benchmark ────────────────────────────────────────────────────────────
    print("[3/4] Running held-out benchmark (NO mean-centering) ...\n")
    per_species: List[dict] = []

    for sp_id in classifier.enrolled_species:
        sp_dir   = PROC_DIR / sp_id
        info     = enroll_info[sp_id]
        all_wavs = sorted(sp_dir.glob("*.wav"))
        test_wavs = all_wavs[info["n_support"]:]
        if not test_wavs:
            test_wavs = all_wavs
        meta  = species_meta.get(sp_id, {})
        theta = info["theta"]

        correct = confused = unknown = total = 0

        for wf in test_wavs:
            emb = embed_wav(bb, wf)
            if emb is None:
                continue
            # use_mean_centering=False: raw cosine similarity, thresholds are calibrated for this
            pred = classifier.predict(emb, threshold=theta, use_mean_centering=False)
            total += 1

            if not pred.is_known:
                unknown += 1
            elif pred.predicted_label == sp_id:
                # Margin gate
                sorted_sc = sorted(pred.scores.items(), key=lambda x: x[1], reverse=True)
                true_sc   = pred.scores.get(sp_id, 0.0)
                runner_up = sorted_sc[1][1] if len(sorted_sc) > 1 else 0.0
                if (true_sc - runner_up) >= MARGIN:
                    correct += 1
                else:
                    confused += 1   # too close to call
            else:
                confused += 1

        if total == 0:
            continue

        per_species.append({
            "species_id":     sp_id,
            "common_name":    meta.get("common_name", sp_id),
            "taxon":          meta.get("taxon", "aves"),
            "original_clips": info["original_clips"],
            "enrolled_clips": info["n_support"],
            "k_sub":          info["k"],
            "tested_clips":   total,
            "theta":          theta,
            "recall_pct":     round(correct  / total * 100, 1),
            "confusion_pct":  round(confused / total * 100, 1),
            "unknown_pct":    round(unknown  / total * 100, 1),
        })

    per_species.sort(key=lambda x: (-x["recall_pct"], x["species_id"]))

    # ── Open-set rejection ───────────────────────────────────────────────────
    print("[4/4] Open-set noise rejection ...")
    rng = np.random.default_rng(42)
    rej_count = rej_total = 0
    for _ in range(60):
        noise = rng.normal(0, 0.05, int(48000 * 3)).astype(np.float32)
        emb   = bb.embed(noise, sr=48000)
        # test with the most lenient threshold
        pred  = classifier.predict(emb, threshold=min(TAXON_THRESHOLDS.values()),
                                   use_mean_centering=False)
        rej_total += 1
        if not pred.is_known:
            rej_count += 1
    rej_rate = rej_count / rej_total * 100.0

    # ── Print ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print(f"  {'Common Name':<36} {'Sci ID':<34} {'Tx':<5} {'theta':<5} "
          f"{'Orig':>4} {'Enrl':>5} {'Test':>4}  {'Recall':>7}  {'Confus':>7}  {'Unknwn':>7}")
    print("-" * 105)
    for r in per_species:
        tx = r["taxon"][:4].upper()
        print(
            f"  {r['common_name']:<36} {r['species_id']:<34} {tx:<5} {r['theta']:.2f} "
            f"{str(r['original_clips']):>4} {r['enrolled_clips']:>5} {r['tested_clips']:>4}  "
            f"{r['recall_pct']:>6.1f}%  {r['confusion_pct']:>6.1f}%  {r['unknown_pct']:>6.1f}%"
        )
    print("=" * 105)

    if per_species:
        avg_recall    = float(np.mean([r["recall_pct"]    for r in per_species]))
        avg_confusion = float(np.mean([r["confusion_pct"] for r in per_species]))
        avg_unknown   = float(np.mean([r["unknown_pct"]   for r in per_species]))
        taxa = sorted(set(r["taxon"] for r in per_species))
        print("\n  Per-taxon macro averages:")
        for tx in taxa:
            tx_rows = [r for r in per_species if r["taxon"] == tx]
            th = TAXON_THRESHOLDS.get(tx, DEFAULT_THRESHOLD)
            print(f"    {tx:<12}  theta={th:.2f}  n={len(tx_rows):>2}  "
                  f"recall={np.mean([r['recall_pct'] for r in tx_rows]):.1f}%  "
                  f"confusion={np.mean([r['confusion_pct'] for r in tx_rows]):.1f}%  "
                  f"unknown={np.mean([r['unknown_pct'] for r in tx_rows]):.1f}%")
    else:
        avg_recall = avg_confusion = avg_unknown = 0.0

    print(f"\n  Overall macro-avg recall    : {avg_recall:.1f}%")
    print(f"  Overall macro-avg confusion : {avg_confusion:.1f}%")
    print(f"  Overall macro-avg unknown   : {avg_unknown:.1f}%")
    print(f"  Open-set rejection          : {rej_rate:.1f}%  ({rej_count}/{rej_total})")

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split":            "90/10 (support/test)",
        "mean_centering":   False,
        "taxon_thresholds": TAXON_THRESHOLDS,
        "margin":           MARGIN,
        "n_subprototypes":  N_SUBPROTOTYPES,
        "n_enrolled":       len(classifier.enrolled_species),
        "macro_avg_recall": round(avg_recall, 2),
        "macro_avg_confusion": round(avg_confusion, 2),
        "macro_avg_unknown": round(avg_unknown, 2),
        "open_set_rejection_pct": round(rej_rate, 2),
        "per_species": per_species,
    }
    json_path = REPORTS_DIR / "benchmark_v2_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = REPORTS_DIR / "benchmark_v2_report.csv"
    with open(csv_path, "w") as f:
        f.write("species_id,common_name,taxon,theta,original_clips,enrolled_clips,k_sub,"
                "tested_clips,recall_pct,confusion_pct,unknown_pct\n")
        for r in per_species:
            f.write(f"{r['species_id']},{r['common_name']},{r['taxon']},{r['theta']},"
                    f"{r['original_clips']},{r['enrolled_clips']},{r['k_sub']},{r['tested_clips']},"
                    f"{r['recall_pct']},{r['confusion_pct']},{r['unknown_pct']}\n")

    tex_path = REPORTS_DIR / "anycall_v2_report.tex"
    _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, tex_path)

    print(f"\n  JSON  -> {json_path}")
    print(f"  CSV   -> {csv_path}")
    print(f"  LaTeX -> {tex_path}")
    print("\nDone.\n")


def _write_tex(per_species, avg_recall, avg_confusion, avg_unknown, rej_rate, out_path):
    lines = []
    a = lines.append
    a(r"\documentclass[11pt]{article}")
    a(r"\usepackage[margin=1in]{geometry}")
    a(r"\usepackage{booktabs,longtable,amsmath,hyperref}")
    a(r"\title{AnyCall v2 Benchmark: 90/10 Split + Sub-Prototypes + Per-Taxon Thresholds}")
    a(r"\author{AnyCall Project Team}")
    a(r"\date{\today}")
    a(r"\begin{document}\maketitle")
    a(r"\begin{abstract}")
    a(r"Benchmark using a 90/10 support/test split per species, "
      r"$K=5$ sub-prototypes (when support $\geq 15$ clips), "
      r"per-taxon cosine thresholds ($\theta_\mathrm{aves}=0.70$, "
      r"$\theta_\mathrm{mammalia}=0.62$, $\theta_\mathrm{amphibia}=0.58$, "
      r"$\theta_\mathrm{insecta}=0.52$), "
      r"and a margin gate $\delta=0.02$. No mean-centering. "
      f"Macro-average recall: \\textbf{{{avg_recall:.1f}\\%}}, "
      f"confusion: \\textbf{{{avg_confusion:.1f}\\%}}, "
      f"unknown: \\textbf{{{avg_unknown:.1f}\\%}}.")
    a(r"\end{abstract}")
    a(r"\section{Results}")
    a(r"\begin{longtable}{llccccc}")
    a(r"\caption{v2 Benchmark Results}\label{tab:v2}\\")
    a(r"\toprule")
    a(r"\textbf{Species (Sci. ID)} & \textbf{Tx} & $\theta$ & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endfirsthead\toprule")
    a(r"\textbf{Species (Sci. ID)} & \textbf{Tx} & $\theta$ & \textbf{Enrl} & \textbf{Recall} & \textbf{Confus.} & \textbf{Unknwn} \\")
    a(r"\midrule\endhead\midrule")
    a(r"\multicolumn{7}{r}{\textit{Continued}}\\\endfoot\bottomrule\endlastfoot")
    for r in per_species:
        sci = r["species_id"].replace("_", r"\_")
        cn  = r["common_name"].replace("&", r"\&")
        tx  = r["taxon"].capitalize()
        a(f"{cn} (\\textit{{{sci}}}) & {tx} & {r['theta']:.2f} & {r['enrolled_clips']} & "
          f"{r['recall_pct']}\\% & {r['confusion_pct']}\\% & {r['unknown_pct']}\\% \\\\")
    a(r"\end{longtable}")
    a(r"\section{Summary}")
    a(r"\begin{tabular}{lr}\toprule\textbf{Metric} & \textbf{Value}\\\midrule")
    a(f"Macro-avg Recall    & {avg_recall:.1f}\\% \\\\")
    a(f"Macro-avg Confusion & {avg_confusion:.1f}\\% \\\\")
    a(f"Macro-avg Unknown   & {avg_unknown:.1f}\\% \\\\")
    a(f"Open-set Rejection  & {rej_rate:.1f}\\% \\\\")
    a(r"\bottomrule\end{tabular}")
    a(r"\end{document}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
