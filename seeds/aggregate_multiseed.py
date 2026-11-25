#!/usr/bin/env python3
"""Summarise the historical all-active evaluation across seeds (mean and sample standard deviation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M8 = Path(__file__).resolve().parent
M4 = ROOT / "evaluation"

sys.path.insert(0, str(M4))
from unified_physical_evaluation import evaluate_models  # noqa: E402

sys.path.insert(0, str(ROOT / "Baseline"))

METRIC_COLS = [
    "Recovery Rate (%)",
    "Mod_Ratio (%)",
    "Avg-MCE",
    "MCE Violation Rate (%)",
    "EOM-Flow Correlation (Pearson)",
]


def evaluate_seed(seed: int, manifest: Path) -> pd.Series:
    cleaned = M8 / "runs" / f"seed_{seed}" / "stage3" / "flow_5min_cleaned.csv"
    if not cleaned.exists():
        raise FileNotFoundError(f"Missing cleaned CSV for seed {seed}: {cleaned}")

    models = {
        "Raw(Imputed)": ROOT / "Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv",
        f"PI-GAT_seed{seed}": cleaned,
    }
    # Evaluate only the raw reference and the current seed in this call.
    import unified_physical_evaluation as upe

    orig = upe.TABLE1_MODELS.copy()
    upe.TABLE1_MODELS = models
    try:
        df, _ = evaluate_models(
            root_dir=ROOT,
            manifest_path=manifest,
            report_set="all_active",
            threshold_set="calibration",
            k_sigma=3.0,
            table1_only=True,
            legacy_all_active=False,
        )
    finally:
        upe.TABLE1_MODELS = orig

    row = df[df["Model"] == f"PI-GAT_seed{seed}"].iloc[0]
    row = row.copy()
    row["seed"] = seed
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    parser.add_argument("--manifest", type=Path, default=M4 / "outputs" / "evaluation_split_manifest.json")
    parser.add_argument("--out", type=Path, default=M8 / "outputs" / "multiseed_metrics.csv")
    args = parser.parse_args()

    rows = []
    for s in args.seeds:
        print(f"Evaluating seed {s} ...")
        rows.append(evaluate_seed(s, args.manifest))

    per_seed = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out, index=False)

    summary_rows = []
    for col in METRIC_COLS:
        vals = per_seed[col].astype(float)
        summary_rows.append({
            "Metric": col,
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 2),
            "n_seeds": len(vals),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = args.out.parent / "multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "seeds": args.seeds,
        "protocol": "M4 all_active report, calibration tau",
        "per_seed_csv": str(args.out),
    }
    with open(args.out.parent / "multiseed_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Done] per-seed: {args.out}")
    print(f"[Done] summary:  {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
