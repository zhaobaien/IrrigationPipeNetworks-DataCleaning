"""
Run the historical physical evaluation of saved flow series.

The default evaluates all active timesteps and estimates the tolerance from
calibration endpoints. The optional test mode restricts the report pool, while
legacy all-active mode estimates the tolerance from all high-credibility steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Load the shared active-timestep mask from the baseline module.
_ROOT = Path(__file__).resolve().parent.parent
_BASELINE = _ROOT / "Baseline"
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))
from effective_value_filtering import build_active_mask  # noqa: E402

BRANCHES = ["B01", "B02", "B03", "B04", "B05", "B06"]
TOTAL_NAME = "Total"

TABLE1_MODELS = {
    "Raw(Imputed)": "imputed",
    "IsolationForest": _ROOT / "Baseline/IsolationForest/cleaning_results/flow_5min_cleaned_if.csv",
    "Multi-variate_Z-score": _ROOT / "Baseline/Multi-variate_Z-score/cleaning_results/flow_5min_cleaned_mvz.csv",
    "GAT-only": _ROOT / "Baseline/GAT-only/cleaning_results/flow_5min_cleaned_gat.csv",
    "Kalman": _ROOT / "Baseline/Kalman/cleaning_results/flow_5min_cleaned_kalman.csv",
    "EPANET-WLS": _ROOT / "Baseline/EPANET-WLS/cleaning_results/flow_5min_cleaned_epanet_wls.csv",
    "PI-GAT_without_bias": _ROOT / "ablation_study/flow_5min_cleaned_without_bias.csv",
    "PI-GAT": _ROOT / "data_clean/outputs/flow_5min_cleaned.csv",
}


def load_lambda(root_dir: Path) -> float:
    """Read the Stage-I leakage fraction from the calibration JSON when available."""
    json_path = root_dir / "PI-GAT/bias_calibration/calibration_params_final.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rate = data.get("leakage", {}).get("rate")
        if rate is not None:
            lam = float(rate)
            print(f"[Info] lambda from calibration_params_final.json = {lam:.4f}")
            return lam
    print("[Warning] leakage.rate not found; using lambda = 0.10")
    return 0.10


def load_bias_params(bias_file: Path, branches: list[str]) -> dict[str, float]:
    biases = {b: 0.0 for b in branches}
    if not bias_file.exists():
        return biases
    with open(bias_file, "r", encoding="utf-8") as f:
        calib = json.load(f)
    for b in branches:
        if b in calib:
            biases[b] = float(np.mean(calib[b].get("biases", [0.0])))
    return biases


def get_eom_effective_opening(od_series: pd.Series, valve_name: str, eom_data: dict) -> np.ndarray:
    if valve_name not in eom_data:
        return np.zeros(len(od_series))
    cfg = eom_data[valve_name]
    x_knots = np.array(cfg["x_knots_pct"])
    y_knots = np.array(cfg.get("y_knots_pct", x_knots))
    return np.interp(od_series.fillna(0), x_knots, y_knots)


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _t_mask(n_rows: int, t_indices: list[int], active_mask: pd.Series) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    for t in t_indices:
        ti = int(t)
        if 0 <= ti < n_rows and bool(active_mask.iloc[ti]):
            mask[ti] = True
    return mask


def compute_threshold_mce(
    imputed_mce_full: pd.Series,
    calibration_mask: np.ndarray,
    k_sigma: float,
) -> float:
    cal_mce = imputed_mce_full.iloc[calibration_mask]
    if len(cal_mce) == 0:
        raise RuntimeError("Calibration set empty after active mask intersection.")
    return float(k_sigma * cal_mce.std())


def evaluate_models(
    *,
    root_dir: Path,
    manifest_path: Path,
    report_set: str,
    threshold_set: str,
    k_sigma: float,
    table1_only: bool,
    legacy_all_active: bool,
) -> tuple[pd.DataFrame, dict]:
    imputed_path = root_dir / "Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv"
    bias_file = root_dir / "PI-GAT/bias_calibration/calibration_params_final.json"
    eom_file = root_dir / "PI-GAT/opening_mapping/Debiased_eom_full.json"
    lam = load_lambda(root_dir)
    pi_gat_biases = load_bias_params(bias_file, BRANCHES)

    with open(eom_file, "r", encoding="utf-8") as f:
        eom_data = json.load(f)

    df_imputed_full = pd.read_csv(imputed_path, index_col="date", parse_dates=True)
    active_mask = build_active_mask(df_imputed_full)
    n_rows = len(df_imputed_full)

    manifest = load_manifest(manifest_path)
    cal_t = manifest["calibration_t_index"]
    test_t = manifest["test_t_index"]

    calibration_mask = _t_mask(n_rows, cal_t, active_mask)
    test_mask = _t_mask(n_rows, test_t, active_mask)
    active_np = active_mask.to_numpy(dtype=bool)

    sum_branches = df_imputed_full[BRANCHES].sum(axis=1)
    imputed_mce_full = (df_imputed_full[TOTAL_NAME] * (1 - lam) - sum_branches).abs()

    if legacy_all_active:
        threshold_mask = np.zeros(n_rows, dtype=bool)
        cred_path = root_dir / "PI-GAT/preprocessing/credibility_windows.csv"
        df_cred = pd.read_csv(cred_path)
        high_t = df_cred[df_cred["credibility"] == "high"]["t_index"].astype(int).values
        for t in high_t:
            if 0 <= t < n_rows:
                threshold_mask[t] = True
        threshold_mask &= active_np
        report_mask = active_np
        protocol_label = "legacy_all_active_high_cred_threshold"
    elif threshold_set == "calibration":
        threshold_mask = calibration_mask
        protocol_label = "M4_calibration_threshold"
    else:
        raise ValueError(f"Unknown threshold_set: {threshold_set}")

    if report_set == "test":
        report_mask = test_mask
    elif report_set == "all_active":
        report_mask = active_np
    else:
        raise ValueError(f"Unknown report_set: {report_set}")

    threshold_mce = compute_threshold_mce(imputed_mce_full, threshold_mask, k_sigma)

    meta = {
        "protocol": protocol_label,
        "report_set": report_set,
        "threshold_set": threshold_set,
        "k_sigma": k_sigma,
        "lambda": lam,
        "threshold_mce": threshold_mce,
        "n_calibration_active": int(calibration_mask.sum()),
        "n_test_active": int(test_mask.sum()),
        "n_report": int(report_mask.sum()),
        "manifest": str(manifest_path.resolve()),
    }
    print(f"[*] Protocol: {protocol_label}")
    print(f"    calibration active steps: {meta['n_calibration_active']}")
    print(f"    test active steps:        {meta['n_test_active']}")
    print(f"    report steps:             {meta['n_report']}")
    print(f"    tau_MCE (k={k_sigma} sigma): {threshold_mce:.4f}")

    models: dict[str, Path] = {}
    if table1_only:
        for name, spec in TABLE1_MODELS.items():
            if spec == "imputed":
                models[name] = imputed_path
            else:
                models[name] = Path(spec)
    else:
        # Include saved parameter-sweep outputs from the sensitivity folders.
        models = {k: (imputed_path if v == "imputed" else Path(v)) for k, v in TABLE1_MODELS.items()}
        sensitivity_dir = root_dir / "sensitivity_analysis"
        if sensitivity_dir.exists():
            for d in sensitivity_dir.iterdir():
                if not d.is_dir() or not d.name.startswith("sensitivity_analysis_"):
                    continue
                val_str = d.name.split("_")[-1]
                if "-Sigma" in val_str:
                    for fname, prefix in [
                        ("flow_5min_cleaned.csv", "PI-GAT"),
                        ("flow_5min_cleaned_gat.csv", "GAT"),
                        ("flow_5min_cleaned_if.csv", "IF"),
                        ("flow_5min_cleaned_mvz.csv", "MVZ"),
                    ]:
                        p = d / fname
                        if p.exists():
                            models[f"{prefix}_{val_str}"] = p
                else:
                    p = d / "flow_5min_cleaned.csv"
                    if p.exists():
                        models[f"Leakage_{val_str}"] = p

    report_idx = df_imputed_full.index[report_mask]
    df_imputed_report = df_imputed_full.loc[report_idx]
    df_mean = df_imputed_report[BRANCHES].mean()

    results = []
    for model_name, path in models.items():
        if not path.exists():
            print(f"Skipping {model_name}, missing: {path}")
            continue

        df = pd.read_csv(path, index_col="date", parse_dates=True)
        df = df.reindex(df_imputed_full.index).loc[report_idx]

        recovery_rate = df[BRANCHES + [TOTAL_NAME]].notna().mean().mean()

        mod_counts = 0
        valid_comp_points = 0
        for b in BRANCHES:
            diff = (df[b] - df_imputed_report[b]).abs()
            thres = 0.02 * df_mean[b]
            mod_counts += int((diff > thres).sum())
            valid_comp_points += int(diff.notna().sum())
        mod_ratio = mod_counts / max(valid_comp_points, 1)

        df_clean_mce = df[[TOTAL_NAME] + BRANCHES].dropna()
        if len(df_clean_mce) == 0:
            avg_mce = np.nan
            violation_rate = np.nan
        else:
            current_lam = lam
            bias_dict = {b: 0.0 for b in BRANCHES}
            if model_name == "PI-GAT" or model_name.startswith("PI-GAT_"):
                bias_dict = pi_gat_biases
            elif model_name.startswith("Leakage_"):
                sens_bias = path.parent / "calibration_params_final.json"
                if sens_bias.exists():
                    bias_dict = load_bias_params(sens_bias, BRANCHES)
                    with open(sens_bias, "r", encoding="utf-8") as f:
                        cal_data = json.load(f)
                    if "leakage" in cal_data and "rate" in cal_data["leakage"]:
                        current_lam = float(cal_data["leakage"]["rate"])

            sum_b = pd.Series(0.0, index=df_clean_mce.index)
            for b in BRANCHES:
                sum_b += df_clean_mce[b] - bias_dict.get(b, 0.0)
            model_mce = np.abs(df_clean_mce[TOTAL_NAME] * (1 - current_lam) - sum_b)
            avg_mce = float(model_mce.mean())
            violation_rate = float((model_mce > threshold_mce).mean())

        physically_valid = (avg_mce < threshold_mce) if pd.notna(avg_mce) else False

        corr_list = []
        for b in BRANCHES:
            if f"OD_{b}" in df.columns:
                od_col = df[f"OD_{b}"]
            elif f"OD_{b}" in df_imputed_report.columns:
                od_col = df_imputed_report[f"OD_{b}"]
            else:
                continue
            od_eff = get_eom_effective_opening(od_col, b, eom_data)
            q_clean = np.asarray(df[b].values, dtype=float)
            od_eff_arr = np.asarray(od_eff, dtype=float)
            valid = ~np.isnan(q_clean) & ~np.isnan(od_eff_arr)
            if valid.sum() > 1 and np.std(q_clean[valid]) > 1e-6 and np.std(od_eff_arr[valid]) > 1e-6:
                corr, _ = pearsonr(q_clean[valid], od_eff_arr[valid])
                corr_list.append(corr)
        eom_corr_mean = float(np.nanmean(corr_list)) if corr_list else np.nan

        results.append(
            {
                "Model": model_name,
                "Recovery Rate (%)": recovery_rate * 100,
                "Mod_Ratio (%)": mod_ratio * 100,
                "Avg-MCE": avg_mce,
                "MCE Violation Rate (%)": violation_rate * 100,
                "Is Physically Valid": "Yes" if physically_valid else "No",
                "EOM-Flow Correlation (Pearson)": eom_corr_mean,
                "tau_MCE": threshold_mce,
                "n_report_steps": int(report_mask.sum()),
            }
        )

    return pd.DataFrame(results), meta


def main() -> None:
    m4_dir = Path(__file__).resolve().parent
    root_dir = m4_dir.parent
    default_manifest = m4_dir / "outputs" / "evaluation_split_manifest.json"
    default_out = m4_dir / "outputs" / "evaluation_metrics.csv"

    parser = argparse.ArgumentParser(description="M4 held-out physical evaluation (Table 1).")
    parser.add_argument("--split-manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--report-set",
        choices=["test", "all_active"],
        default="all_active",
        help="Table 1 default: all_active (26,472 steps); use test for held-out-only reporting.",
    )
    parser.add_argument("--threshold-set", choices=["calibration"], default="calibration")
    parser.add_argument("--k-sigma", type=float, default=3.0)
    parser.add_argument("--table1-only", action="store_true", default=True)
    parser.add_argument("--all-models", action="store_true", help="Include sensitivity analysis models.")
    parser.add_argument(
        "--legacy-all-active",
        action="store_true",
        help="Reproduce pre-M4 logic: threshold on all high-cred active, report on all active.",
    )
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--write-meta", type=Path, default=None)
    args = parser.parse_args()

    if not args.split_manifest.exists():
        print(f"[Error] Split manifest not found: {args.split_manifest}")
        print("Run: python evaluation/split_manifest.py")
        raise SystemExit(1)

    if args.legacy_all_active:
        report_set = "all_active"
        threshold_set = "calibration"
        # The legacy mode estimates the tolerance from all high-credibility active timesteps.
    else:
        report_set = args.report_set
        threshold_set = args.threshold_set

    res_df, meta = evaluate_models(
        root_dir=root_dir,
        manifest_path=args.split_manifest,
        report_set=report_set,
        threshold_set=threshold_set,
        k_sigma=args.k_sigma,
        table1_only=not args.all_models,
        legacy_all_active=args.legacy_all_active,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out, index=False)
    print(f"\n[Done] Saved: {args.out}")
    try:
        print(res_df.to_markdown(index=False))
    except ImportError:
        print(res_df.to_string(index=False))

    meta_path = args.write_meta or (args.out.parent / "evaluation_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Done] Meta: {meta_path}")


if __name__ == "__main__":
    main()
