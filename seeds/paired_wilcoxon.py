#!/usr/bin/env python3
"""
Run the historical paired Wilcoxon comparison of seed-42 PI-GAT and its baselines.

Use the all-active report pool and take the error tolerance from the calibration set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
M4 = ROOT / "evaluation"
M8 = Path(__file__).resolve().parent

sys.path.insert(0, str(M4))
from unified_physical_evaluation import (  # noqa: E402
    BRANCHES,
    TOTAL_NAME,
    compute_threshold_mce,
    load_bias_params,
    load_lambda,
    load_manifest,
    _t_mask,
)

sys.path.insert(0, str(ROOT / "Baseline"))
from effective_value_filtering import build_active_mask  # noqa: E402

BASELINE_CSV = {
    "IsolationForest": ROOT / "Baseline/IsolationForest/cleaning_results/flow_5min_cleaned_if.csv",
    "Multi-variate_Z-score": ROOT / "Baseline/Multi-variate_Z-score/cleaning_results/flow_5min_cleaned_mvz.csv",
    "GAT-only": ROOT / "Baseline/GAT-only/cleaning_results/flow_5min_cleaned_gat.csv",
    "Kalman": ROOT / "Baseline/Kalman/cleaning_results/flow_5min_cleaned_kalman.csv",
    "EPANET-WLS": ROOT / "Baseline/EPANET-WLS/cleaning_results/flow_5min_cleaned_epanet_wls.csv",
    "PI-GAT_without_bias": ROOT / "ablation_study/flow_5min_cleaned_without_bias.csv",
}


def mce_at_row(row: pd.Series, lam: float, biases: dict[str, float]) -> float:
    q_t = float(row[TOTAL_NAME])
    sum_b = sum(float(row[b]) - biases.get(b, 0.0) for b in BRANCHES)
    return abs(q_t * (1.0 - lam) - sum_b)


def paired_mce_series(
    df: pd.DataFrame,
    imputed_index: pd.Index,
    report_mask: np.ndarray,
    lam: float,
    biases: dict[str, float],
) -> np.ndarray:
    df = df.reindex(imputed_index)
    out = []
    cols = [TOTAL_NAME] + BRANCHES
    for i in range(len(imputed_index)):
        if not report_mask[i]:
            continue
        row = df.iloc[i][cols]
        if row.isna().any():
            continue
        out.append(mce_at_row(row, lam, biases))
    return np.asarray(out, dtype=float)


def load_df(path: Path, imputed_index: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.reindex(imputed_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pigat-csv",
        type=Path,
        default=ROOT / "data_clean/outputs/flow_5min_cleaned.csv",
        help="PI-GAT seed=42 cleaned CSV",
    )
    parser.add_argument("--out", type=Path, default=M8 / "outputs" / "wilcoxon_paired.csv")
    args = parser.parse_args()

    imputed_path = ROOT / "Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv"
    manifest_path = M4 / "outputs" / "evaluation_split_manifest.json"
    bias_file = ROOT / "PI-GAT/bias_calibration/calibration_params_final.json"

    lam = load_lambda(ROOT)
    pigat_biases = load_bias_params(bias_file, BRANCHES)

    df_imp = pd.read_csv(imputed_path, index_col="date", parse_dates=True)
    active_mask = build_active_mask(df_imp)
    manifest = load_manifest(manifest_path)
    cal_mask = _t_mask(len(df_imp), manifest["calibration_t_index"], active_mask)
    report_mask = active_mask.to_numpy()

    sum_br = df_imp[BRANCHES].sum(axis=1)
    imputed_mce = (df_imp[TOTAL_NAME] * (1 - lam) - sum_br).abs()
    tau = compute_threshold_mce(imputed_mce, cal_mask, 3.0)

    df_pigat = load_df(args.pigat_csv, df_imp.index)
    biases_zero = {b: 0.0 for b in BRANCHES}

    results = []
    cols = [TOTAL_NAME] + BRANCHES
    pigat_aligned = df_pigat.reindex(df_imp.index)

    for name, path in BASELINE_CSV.items():
        if not path.exists():
            print(f"Skip {name}, missing {path}")
            continue
        df_b = load_df(path, df_imp.index)
        a_list, b_list = [], []
        for i in range(len(df_imp)):
            if not report_mask[i]:
                continue
            rp = pigat_aligned.iloc[i][cols]
            rb = df_b.iloc[i][cols]
            if rp.isna().any() or rb.isna().any():
                continue
            a_list.append(mce_at_row(rp, lam, pigat_biases))
            b_list.append(mce_at_row(rb, lam, biases_zero))
        a = np.asarray(a_list, dtype=float)
        b = np.asarray(b_list, dtype=float)
        diff = a - b
        try:
            stat, p = wilcoxon(a, b, alternative="less", zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, np.nan
        results.append({
            "Comparison": f"PI-GAT_vs_{name}",
            "n_pairs": len(a),
            "median_diff_MCE_m3h": round(float(np.median(diff)), 2),
            "mean_PI-GAT_MCE": round(float(np.mean(a)), 2),
            "mean_baseline_MCE": round(float(np.mean(b)), 2),
            "wilcoxon_stat": stat,
            "p_value": p,
            "PI-GAT_better_p_lt_0.05": bool(p < 0.05) if p == p else False,
            "lambda": round(lam, 4),
            "tau_MCE": round(tau, 2),
        })

    out_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))
    print(f"\n[Done] {args.out}")


if __name__ == "__main__":
    main()
