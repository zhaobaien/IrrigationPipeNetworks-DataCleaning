from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
DATA_DIR      = ROOT_DIR / "data_clean" / "outputs"
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"

SENSOR_COLS = ["B01", "B02", "B03", "B04", "B05", "B06"]

# FLAG codes (must match data_cleaning.py constants)
FLAG_DRIFT            =  1
FLAG_SPIKE_FIXED      =  2
FLAG_TOTAL_ADJUSTED   =  3
FLAG_ZERO             =  5


def _run_lengths(arr: np.ndarray, target_val: int) -> list[int]:
    """Return list of consecutive run lengths where arr == target_val."""
    runs: list[int] = []
    i = 0
    while i < len(arr):
        if arr[i] == target_val:
            j = i
            while j < len(arr) and arr[j] == target_val:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs


def extract_spike_profile(flags: pd.DataFrame, raw: pd.DataFrame, cln: pd.DataFrame) -> dict:
    """
    For each sensor, find FLAG_SPIKE_FIXED rows and compute raw/clean ratio.
    Returns per-sensor stats + a global median used as injection multiplier.
    """
    per_sensor: dict = {}
    all_ratios: list[float] = []
    for s in SENSOR_COLS:
        mask = flags[s] == FLAG_SPIKE_FIXED
        n = int(mask.sum())
        if n == 0:
            per_sensor[s] = {"n": 0, "median_ratio": 2.5, "p5": 1.5, "p95": 4.0}
            continue
        cln_vals = cln.loc[mask, s].replace(0.0, np.nan)
        ratios = (raw.loc[mask, s] / cln_vals).dropna()
        ratios = ratios[(ratios > 0.01) & (ratios < 1000)]   # clip pathological outliers
        med = float(np.median(ratios)) if len(ratios) else 2.5
        per_sensor[s] = {
            "n": n,
            "median_ratio": round(med, 3),
            "p5":  round(float(np.quantile(ratios, 0.05)) if len(ratios) else 1.5, 3),
            "p95": round(float(np.quantile(ratios, 0.95)) if len(ratios) else 4.0, 3),
        }
        all_ratios.extend(ratios.tolist())

    global_median = round(float(np.median(all_ratios)) if all_ratios else 2.5, 3)
    print(f"  [Spike] global_median_ratio = {global_median}")
    return {"per_sensor": per_sensor, "global_median_ratio": global_median}


def extract_drop_profile(flags: pd.DataFrame) -> dict:
    """
    For each sensor, compute run-length statistics of FLAG_ZERO runs (valve-forced
    zero) and FLAG_SPIKE_FIXED=0 patterns.  We treat prolonged FLAG_ZERO as proxy
    for sensor drop (same pattern: sustained zero reading).
    Returns p50 and p75 run lengths per sensor.
    """
    per_sensor: dict = {}
    for s in SENSOR_COLS:
        runs = _run_lengths(np.asarray(flags[s], dtype=int), FLAG_ZERO)
        # keep only runs >= 2 steps (single-step zeros are valve events, not drops)
        runs = [r for r in runs if r >= 2]
        if not runs:
            per_sensor[s] = {"n_runs": 0, "p50_steps": 24, "p75_steps": 86}
            continue
        arr = np.array(runs)
        per_sensor[s] = {
            "n_runs":    len(arr),
            "p50_steps": int(np.percentile(arr, 50)),
            "p75_steps": int(np.percentile(arr, 75)),
            "max_steps": int(arr.max()),
        }
        print(f"  [Drop/{s}] n_runs={len(arr)}, p50={per_sensor[s]['p50_steps']}, "
              f"p75={per_sensor[s]['p75_steps']} steps")
    # recommended injection length = p75 of B03 (primary demo sensor), min 24 max 144
    B03_p75 = per_sensor.get("B03", {}).get("p75_steps", 86)
    recommended_drop_len = int(np.clip(B03_p75, 24, 144))
    print(f"  [Drop] recommended_injection_steps = {recommended_drop_len}")
    return {"per_sensor": per_sensor, "recommended_injection_steps": recommended_drop_len}


def extract_drift_profile(flags: pd.DataFrame, raw: pd.DataFrame, cln: pd.DataFrame) -> dict:
    """
    Derive ramp-drift parameters from FLAG_DRIFT (static-bias) rows.
    The static bias correction applied Q_clean = Q_raw / scale, so the
    underlying gain error = scale - 1.0.  We approximate the slope as:

        median_gain_error / median_run_length  (gain error per step)

    and set the plateau target = 1 + median_gain_error so the ramp
    asymptotically reaches the same distortion as real sensors had.
    """
    per_sensor: dict = {}
    for s in SENSOR_COLS:
        mask = (flags[s] == FLAG_DRIFT) & (cln[s].replace(0, np.nan).notna())
        n = int(mask.sum())
        if n < 10:
            per_sensor[s] = {"ramp_slope_per_step": 0.001, "plateau_ratio": 1.15,
                             "n_drift_rows": n}
            continue
        cln_vals = cln.loc[mask, s].replace(0.0, np.nan)
        gain_errors = (raw.loc[mask, s] - cln_vals).abs() / cln_vals.abs()
        gain_errors = gain_errors.dropna()
        gain_errors = gain_errors[gain_errors < 2.0]   # clip wild outliers

        median_gain = float(np.median(gain_errors)) if len(gain_errors) else 0.10

        # run-length of drift patches to estimate typical duration
        runs = _run_lengths(np.asarray(flags[s], dtype=int), FLAG_DRIFT)
        runs = [r for r in runs if r >= 5]
        median_run = float(np.median(runs)) if runs else 288.0

        slope = round(median_gain / median_run, 6)
        plateau = round(float(np.clip(1.0 + median_gain, 1.05, 1.40)), 4)
        per_sensor[s] = {
            "n_drift_rows":       n,
            "median_gain_error":  round(median_gain, 4),
            "median_run_steps":   round(median_run, 1),
            "ramp_slope_per_step": slope,
            "plateau_ratio":      plateau,
        }
        print(f"  [Drift/{s}] gain={round(median_gain*100,1)}%  "
              f"run={round(median_run,0)} steps  slope={slope:.6f}  plateau={plateau}")

    # Use B04 as B3 reference (well-sampled drift sensor)
    ref = per_sensor.get("B04", {})
    return {
        "per_sensor":          per_sensor,
        "recommended_ramp_steps": 144,          # 12 h ramp-up
        "recommended_slope":   ref.get("ramp_slope_per_step", 0.001),
        "recommended_plateau": ref.get("plateau_ratio", 1.15),
    }


def extract_total_bias_profile(flags: pd.DataFrame, raw: pd.DataFrame, cln: pd.DataFrame) -> dict:
    """
    Derive root-node (Total meter) chronic bias from FLAG_TOTAL_ADJUSTED rows.
    Clamp to [0.16, 0.22] to stay inside the PI-GAT Scenario-B activation corridor
    (τ_MCE_REL=15% lower bound, SCENARIO_B_FUSE=30% upper bound).
    """
    mask = flags["Total"] == FLAG_TOTAL_ADJUSTED
    n = int(mask.sum())
    if n < 5:
        bias = 0.18
    else:
        cln_vals = cln.loc[mask, "Total"].replace(0.0, np.nan)
        rel_err = (raw.loc[mask, "Total"] - cln_vals).abs() / cln_vals.abs()
        rel_err = rel_err.dropna()
        bias = float(np.clip(np.median(rel_err), 0.16, 0.22)) if len(rel_err) else 0.18

    recommended_multiplier = round(1.0 + bias, 3)
    print(f"  [TotalBias] n_rows={n}, bias={round(bias*100,1)}%, "
          f"recommended_multiplier={recommended_multiplier}")
    return {
        "n_adjusted_rows":         n,
        "median_relative_bias":    round(bias, 4),
        "recommended_multiplier":  recommended_multiplier,
    }


def main() -> None:
    print("=== Empirical Fault Profile Extraction ===\n")

    flags = pd.read_csv(str(DATA_DIR / "cleaning_flags.csv"))
    raw   = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_imputed.csv"))
    cln   = pd.read_csv(str(DATA_DIR / "flow_5min_cleaned.csv"))

    # Align index (all three share row-for-row correspondence)
    assert len(flags) == len(raw) == len(cln), \
        "Row count mismatch between flags / raw / cleaned CSVs!"

    print("[1/4] Extracting Spike profile (FLAG=2)...")
    spike_profile = extract_spike_profile(flags, raw, cln)

    print("\n[2/4] Extracting Sensor-Drop profile (FLAG=5 runs)...")
    drop_profile = extract_drop_profile(flags)

    print("\n[3/4] Extracting Gain-Drift profile (FLAG=1)...")
    drift_profile = extract_drift_profile(flags, raw, cln)

    print("\n[4/4] Extracting Total-Meter-Bias profile (FLAG=3)...")
    bias_profile = extract_total_bias_profile(flags, raw, cln)

    # ── Assemble and write ────────────────────────────────────────────────
    profiles = {
        "_description": (
            "Empirical fault parameters extracted from Stage-I/II PI-GAT cleaning artefacts. "
            "Used by run_cascading_injection.py to seed physically realistic faults."
        ),
        "spike":        spike_profile,
        "sensor_drop":  drop_profile,
        "gain_drift":   drift_profile,
        "total_bias":   bias_profile,
    }

    out_path = SCRIPT_DIR / "fault_profiles.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)

    print(f"\n=== Done!  Profiles saved to {out_path} ===")


if __name__ == "__main__":
    main()
