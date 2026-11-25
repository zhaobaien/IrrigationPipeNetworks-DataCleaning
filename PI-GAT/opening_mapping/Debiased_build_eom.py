from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# bin0: [0, 5%] → no bias; bin1: (5%, 33.33%]; bin2: (33.33%, 66.67%]; bin3: (66.67%, 100%]
_BIN_THRESHOLDS = (5.0, 33.3333, 66.6667)


def _od_to_bin(od_pct: float) -> int:
    """Map mechanical opening in percent to the calibration bin, 0–3."""
    if od_pct <= _BIN_THRESHOLDS[0]:
        return 0
    elif od_pct <= _BIN_THRESHOLDS[1]:
        return 1
    elif od_pct <= _BIN_THRESHOLDS[2]:
        return 2
    else:
        return 3


def _load_scale_factors(calib_json: Path) -> dict[str, list[float]]:
    """Load the three open-bin scale factors per valve.

    Bin 0, at or below 5% opening, always uses unit scale."""
    with open(calib_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    valves_cfg = obj.get("bias", {}).get("valves", {})
    result: dict[str, list[float]] = {}
    for vname, cfg in valves_cfg.items():
        scale_list = cfg.get("scale", [])
        if len(scale_list) != 3:
            raise ValueError(
                f"Valve '{vname}' must have three scales. Found {len(scale_list)}"
            )
        result[vname] = [float(s) for s in scale_list]
    return result


def _apply_debias(df: pd.DataFrame, scale_factors: dict[str, list[float]], valves: list[str]) -> pd.DataFrame:
    """Divide each observed branch flow by the scale for its opening bin.

    Bin 0 uses a scale of 1. The three open bins use their calibrated scales."""
    df = df.copy()
    b0, b1, b2 = _BIN_THRESHOLDS  # 5.0, 33.3333, 66.6667

    for v in valves:
        od_col = f"OD_{v}"
        q_col = v
        if od_col not in df.columns or q_col not in df.columns:
            continue
        if v not in scale_factors:
            continue

        s1, s2, s3 = scale_factors[v]
        od = pd.to_numeric(df[od_col], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)

        scale_arr = np.where(
            od <= b0, 1.0,
            np.where(
                od <= b1, s1,
                np.where(od <= b2, s2, s3)
            )
        )
        scale_arr = np.where(scale_arr > 1e-6, scale_arr, 1.0)

        q_debiased = q / scale_arr
        df[q_col] = q_debiased

    return df


def _load_high_credibility_indices(credibility_csv: Path) -> list[int]:
    """Read the indices of high-credibility time steps."""
    indices: list[int] = []
    with open(credibility_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("credibility", "")).strip() == "high":
                try:
                    indices.append(int(row["t_index"]))
                except (KeyError, ValueError):
                    pass
    return indices


def main() -> None:
    here = Path(__file__).resolve()
    pi_gat_dir = here.parent.parent          # .../PI-GAT
    repo_root = pi_gat_dir.parent

    if str(pi_gat_dir) not in sys.path:
        sys.path.insert(0, str(pi_gat_dir))

    try:
        from opening_mapping.eom import (
            learn_effective_opening_mapping_from_flow,
            load_mapping_json,
            save_mapping_json,
            compare_eom_three_way,
            _filter_valid_windows,
            _parse_segments as parse_segments,
        )
    except Exception:  # pragma: no cover
        from eom import (
            learn_effective_opening_mapping_from_flow,
            load_mapping_json,
            save_mapping_json,
            compare_eom_three_way,
            _filter_valid_windows,
            _parse_segments as parse_segments,
        )

    default_calib = str(pi_gat_dir / "bias_calibration" / "calibration_params_final.json")
    default_cred  = str(pi_gat_dir / "preprocessing" / "credibility_windows.csv")
    default_flow  = str(repo_root / "Preprocess_Timeseries" / "outputs" / "preprocessed" / "flow_5min_imputed.csv")

    parser = argparse.ArgumentParser(
        description=(
            "Fit EOM to bias-corrected data and compare it with the high-credibility and uncorrected all-window mappings.\n"
            "Default outputs: Debiased_eom_full.json and new_eom_high_vs_full.json."
        )
    )
    parser.add_argument("--flow",                  type=str, default=default_flow)
    parser.add_argument("--calib_json",            type=str, default=default_calib,
                        help="Path to calibration_params_final.json")
    parser.add_argument("--credibility_windows",   type=str, default=default_cred,
                        help="Path to credibility_windows.csv")
    parser.add_argument("--eom_full_json",         type=str,
                        default=str(here.parent / "eom_full.json"),
                        help="Previously fitted all-window EOM JSON (eom_full.json)")
    parser.add_argument("--eom_high_json",         type=str,
                        default=str(here.parent / "eom_high.json"),
                        help="Previously fitted high-credibility EOM JSON (eom_high.json)")
    parser.add_argument("--valves",                type=str, default="B01,B02,B03,B04,B05,B06")
    parser.add_argument("--segments",             type=str, default="0-20,21-40,41-60,61-80,81-100")
    parser.add_argument("--cap_quantile",          type=float, default=0.95)
    parser.add_argument("--min_samples_per_segment", type=int, default=30)
    parser.add_argument("--delta_quantile",        type=float, default=0.8)
    parser.add_argument("--max_shift_pct",         type=float, default=60.0)
    parser.add_argument("--out",                   type=str,
                        default=str(here.parent / "Debiased_eom_full.json"),
                        help="Output path for debiased full EOM.")
    parser.add_argument("--out_comparison",        type=str, default=None,
                        help="Output path for 3-way comparison JSON "
                             "(default: new_eom_high_vs_full.json next to --out).")

    args = parser.parse_args()

    flow         = Path(str(args.flow))
    calib_json   = Path(str(args.calib_json))
    cred_csv     = Path(str(args.credibility_windows))
    eom_full_path = Path(str(args.eom_full_json))
    eom_high_path = Path(str(args.eom_high_json))
    out          = Path(str(args.out))
    out_comparison = Path(
        args.out_comparison if args.out_comparison
        else str(out.parent / "new_eom_high_vs_full.json")
    )

    if not flow.exists():
        raise FileNotFoundError(f"Flow CSV not found: {flow}")
    if not calib_json.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calib_json}")

    valves = [v.strip() for v in str(args.valves).split(",") if v.strip()]
    if not valves:
        raise ValueError("No valves specified")

    segments = parse_segments(str(args.segments)) if str(args.segments).strip() else None

    cap_quantile          = float(args.cap_quantile)
    min_samples_per_segment = int(args.min_samples_per_segment)
    delta_quantile        = float(args.delta_quantile)
    max_shift_pct         = float(args.max_shift_pct)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print("Loading valve bias scales ...")
    scale_factors = _load_scale_factors(calib_json)
    print(f"  Valves loaded: {list(scale_factors.keys())}")
    for vname, scales in scale_factors.items():
        print(f"    {vname}: bin1={scales[0]:.6f}, bin2={scales[1]:.6f}, bin3={scales[2]:.6f}")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("\nLoading and filtering all input data ...")
    required_columns = ["Total", "B01", "B02", "B03", "B04", "B05", "B06"]
    needed = ["date"]
    for v in valves:
        needed.append(f"OD_{v}")
        needed.append(v)
    all_needed = list(set(needed + required_columns))

    df_head = pd.read_csv(flow, nrows=1)
    usecols = [c for c in all_needed if c in df_head.columns]
    df_raw = pd.read_csv(flow, usecols=usecols)
    df_filtered = _filter_valid_windows(df_raw, required_columns)
    print(f"  Valid rows after filtering: {len(df_filtered)}")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("\nApplying Q_debiased = Q_raw / Scale(OD) ...")
    df_debiased = _apply_debias(df_filtered, scale_factors, valves)
    for v in valves:
        if v not in df_filtered.columns:
            continue
        q_raw  = pd.to_numeric(df_filtered[v], errors="coerce")
        q_debi = pd.to_numeric(df_debiased[v],  errors="coerce")
        ratio  = (q_debi / q_raw.replace(0, np.nan)).mean()
        print(f"  {v}: mean(Q_debiased/Q_raw) = {ratio:.6f}")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("\n[Debiased] Fitting EOM to bias-corrected data ...")
    debiased_mapping = learn_effective_opening_mapping_from_flow(
        flow_csv=flow,
        valves=valves,
        segments=segments,
        cap_quantile=cap_quantile,
        min_samples_per_segment=min_samples_per_segment,
        delta_quantile=delta_quantile,
        max_shift_pct=max_shift_pct,
        df_override=df_debiased,
        data_mode_override="debiased_full",
    )
    save_mapping_json(debiased_mapping, out)
    print(f"Bias-corrected EOM saved to: {out}")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    if not eom_full_path.exists():
        print(f"\nWarning: eom_full.json missing. Skipping the three-way comparison: {eom_full_path}")
        return
    if not eom_high_path.exists():
        print(f"\nWarning: eom_high.json missing. Skipping the three-way comparison: {eom_high_path}")
        return

    print("\nLoading eom_full.json and eom_high.json ...")
    full_mapping = load_mapping_json(eom_full_path)
    high_mapping = load_mapping_json(eom_high_path)

    print("Creating the three-way comparison JSON ...")
    comparison = compare_eom_three_way(full_mapping, high_mapping, debiased_mapping)
    save_mapping_json(comparison, out_comparison)
    print(f"Three-way comparison JSON saved to: {out_comparison}")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("Three-way comparison of EOM curves")
    print(f"  Curve 1 (full)     : {eom_full_path.name}")
    print(f"  Curve 2 (high)     : {eom_high_path.name}")
    print(f"  Curve 3 (debiased) : {out.name}")
    print()
    hdr = f"{'Valve':6s}  {'high-full max':>14s}  {'high-full agr':>13s}  {'debi-full max':>14s}  {'debi-full agr':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for v in valves:
        if v not in comparison:
            continue
        interp = comparison[v]["interpretation"]
        hf = interp["high_vs_full"]
        df_ = interp["debiased_vs_full"]
        print(
            f"  {v:6s}"
            f"  {hf['max_abs_delta_pct']:>13.2f}%"
            f"  {hf['agreement']:>13s}"
            f"  {df_['max_abs_delta_pct']:>13.2f}%"
            f"  {df_['agreement']:>13s}"
        )
    print("=" * 80)
    print(
        "\nReading the comparison:\n"
        "  If max_abs_delta for debiased_vs_full is smaller than for high_vs_full\n"
        "    → Compare the bias-corrected curve directly with the high-credibility fit\n"
        "  If debiased_vs_full and high_vs_full are similar in agreement and direction\n"
        "    → The bias-corrected fit can cover a wider range of OD values\n"
        "  If debiased_vs_full still shows poor agreement\n"
        "    → Inspect possible sources of drift beyond instrument scaling"
    )
    print(f"\nSee {out_comparison.name} for all three curves and their differences by valve.")
    print(f"     Pass {out.name} to train_pi_gat.py via --effective_opening_mapping_json.")


if __name__ == "__main__":
    main()
