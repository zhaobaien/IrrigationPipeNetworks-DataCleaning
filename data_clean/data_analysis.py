from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SCRIPT_DIR     = Path(__file__).parent
ROOT_DIR       = SCRIPT_DIR.parent
PI_GAT_DIR     = ROOT_DIR / "PI-GAT"
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"
OUTPUT_DIR     = SCRIPT_DIR / "outputs"
FIGURE_DIR     = SCRIPT_DIR / "analysis_results"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SENSOR_COLS   = ["B01", "B02", "B03", "B04", "B05", "B06"]
TOTAL_COL     = "Total"
ALL_FLOW_COLS = SENSOR_COLS + [TOTAL_COL]

FLAG_IGNORE           = -2
FLAG_SPINUP           = -1
FLAG_NORMAL           =  0
FLAG_DRIFT            =  1
FLAG_SPIKE_FIXED      =  2
FLAG_TOTAL_ADJUSTED   =  3
FLAG_DYNAMIC_DRIFT    =  4
FLAG_ZERO             =  5
FLAG_VALVE_TRANSITION =  6
FLAG_SYSTEM_ANOMALY   =  7
FLAG_FULL_RECONSTRUCT =  8

FLAG_NAMES: Dict[int, str] = {
    FLAG_IGNORE:           "Ignore",
    FLAG_SPINUP:           "Spinup",
    FLAG_NORMAL:           "Normal",
    FLAG_DRIFT:            "Static Drift",
    FLAG_SPIKE_FIXED:      "Spike Fixed",
    FLAG_TOTAL_ADJUSTED:   "Total Adjusted",
    FLAG_DYNAMIC_DRIFT:    "Dynamic Drift",
    FLAG_ZERO:             "Valve Closed",
    FLAG_VALVE_TRANSITION: "Valve Transition",
    FLAG_SYSTEM_ANOMALY:   "System Anomaly",
    FLAG_FULL_RECONSTRUCT: "Full Reconstruct",
}

BRANCH_COLORS = {
    "B01":  "#1f77b4",
    "B02":"#ff7f0e",
    "B03":  "#2ca02c",
    "B04":  "#d62728",
    "B05":  "#9467bd",
    "B06":  "#8c564b",
}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _build_scale_matrix_for_analysis(
    df_raw: pd.DataFrame,
    calib: dict,
) -> pd.DataFrame:
    """Rebuild the Stage-II scales used to divide raw branch flows for diagnostics."""
    scale_df = pd.DataFrame(1.0, index=df_raw.index, columns=SENSOR_COLS, dtype=float)

    sensors_calib = calib.get("sensors", {})
    for sensor in SENSOR_COLS:
        od_col = f"OD_{sensor}"
        if sensor not in sensors_calib or od_col not in df_raw.columns:
            continue
        s_calib = sensors_calib[sensor]
        bins    = s_calib.get("bins", {})
        od_vals = df_raw[od_col].clip(0.0, 100.0).fillna(100.0)

        for bin_key, scale_val in bins.items():
            lo, hi = (float(x) for x in bin_key.split("-"))
            mask   = (od_vals >= lo) & (od_vals <= hi)
            if mask.any():
                scale_df.loc[mask, sensor] = float(scale_val)

    return scale_df


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_all_data() -> Tuple[
    pd.DataFrame,   # flag_df
    pd.DataFrame,   # df_raw
    pd.DataFrame,   # df_cleaned
    Dict,           # sensor_profile
    Dict,           # cleaning_summary
    dict,           # calib
    pd.Series,      # is_active_row
]:
    flags_path   = OUTPUT_DIR / "cleaning_flags.csv"
    cleaned_path = OUTPUT_DIR / "flow_5min_cleaned.csv"
    profile_path = OUTPUT_DIR / "sensor_profile.json"
    summary_path = OUTPUT_DIR / "cleaning_summary.json"
    raw_path     = PREPROCESS_DIR / "flow_5min_imputed.csv"
    calib_path   = PI_GAT_DIR / "bias_calibration" / "calibration_params_final.json"
    cred_path    = PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"

    for p in [flags_path, cleaned_path, raw_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required output file not found: {p}")

    flag_df    = pd.read_csv(flags_path,   index_col=0, parse_dates=True)
    df_cleaned = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df_raw     = pd.read_csv(raw_path,     index_col=0, parse_dates=True)

    df_raw     = df_raw.reindex(flag_df.index)
    df_cleaned = df_cleaned.reindex(flag_df.index)

    sensor_profile: Dict = {}
    if profile_path.exists():
        with open(profile_path, encoding="utf-8") as f:
            sensor_profile = json.load(f)

    cleaning_summary: Dict = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            cleaning_summary = json.load(f)

    calib: dict = {}
    if calib_path.exists():
        with open(calib_path, encoding="utf-8") as f:
            calib = json.load(f)

    n_total    = len(flag_df)
    is_active  = pd.Series(False, index=flag_df.index)
    if cred_path.exists():
        cred_df  = pd.read_csv(cred_path, usecols=["t_index"])
        act_arr  = cred_df["t_index"].astype(int).to_numpy()
        in_range = (act_arr >= 0) & (act_arr < n_total)
        is_active.iloc[act_arr[in_range]] = True

    return flag_df, df_raw, df_cleaned, sensor_profile, cleaning_summary, calib, is_active


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def print_summary_table(flag_df: pd.DataFrame, is_active: pd.Series) -> None:
    valid_mask = flag_df[SENSOR_COLS[0]].ne(FLAG_SPINUP) & flag_df[SENSOR_COLS[0]].ne(FLAG_IGNORE)
    active_valid = is_active & valid_mask

    header = f"{'Sensor':<8s}  " + "  ".join(f"{FLAG_NAMES[k]:<16s}" for k in sorted(FLAG_NAMES))
    print("  " + header)
    print("  " + "-" * 120)
    for sensor in ALL_FLOW_COLS:
        if sensor not in flag_df.columns:
            continue
        row = f"{sensor:<8s}  "
        fv  = flag_df.loc[active_valid, sensor]
        for fk in sorted(FLAG_NAMES):
            cnt = int((fv == fk).sum())
            row += f"{cnt:<18d}"
        print("  " + row)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_radar_chart(sensor_profile: Dict, cleaning_summary: Dict) -> None:
    """Plot four sensor diagnostics: spike rate, drift count, opening–residual
    correlation and contribution to mass-balance violations."""
    AXES = [
        "Spike\nRate",
        "Drift\nFreq",
        "OD-R\nCorr",
        "Root\nCause",
    ]
    N = len(AXES)

    raw_vals: Dict[str, List[float]] = {s: [] for s in SENSOR_COLS}
    for sensor in SENSOR_COLS:
        prof = sensor_profile.get(sensor, {})
        raw_vals[sensor] = [
            float(prof.get("spike_rate_per_1k_windows", 0.0)),
            float(prof.get("dynamic_drift_event_count", 0)),
            float(prof.get("od_residual_pearson_corr", 0.0)),
            float(prof.get("root_cause_probability_pct", 0.0)),
        ]

    arr = np.array([raw_vals[s] for s in SENSOR_COLS])  # [6, 4]
    axis_min = arr.min(axis=0)
    axis_max = arr.max(axis=0)
    axis_rng = np.where(axis_max - axis_min > 1e-9, axis_max - axis_min, 1.0)
    norm_arr = (arr - axis_min) / axis_rng  # [6, 4], all in [0,1]

    angles   = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for idx, sensor in enumerate(SENSOR_COLS):
        vals = norm_arr[idx].tolist() + [norm_arr[idx, 0]]
        color = BRANCH_COLORS.get(sensor, f"C{idx}")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=sensor, markersize=4)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
    ax.set_title("Sensor Health Radar\n(0 = best, 1 = worst per axis)", fontsize=13, pad=20)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.12),
        fontsize=10,
        framealpha=0.9,
    )

    info_lines = ["Raw values (before normalization):"]
    for sensor in SENSOR_COLS:
        v = raw_vals[sensor]
        info_lines.append(
            f"  {sensor}: spike={v[0]:.2f}/1k  drift_evts={v[1]:.0f}"
            f"  OD-corr={v[2]:.3f}  RC%={v[3]:.1f}"
        )

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig1_sensor_health_radar.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 1 saved: {out_path.name}")

    txt_path = FIGURE_DIR / "fig1_radar_raw_values.txt"
    txt_path.write_text("\n".join(info_lines), encoding="utf-8")
    print(f"  [OK] Raw radar values: {txt_path.name}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_total_correction(
    flag_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    leakage_rate: float,
    is_active: pd.Series,
) -> None:
    """Compare raw and cleaned trunk flow with the branch sum divided by 1 - lambda.
    Shading identifies trunk corrections and flagged system anomalies."""
    mask = is_active & (flag_df[TOTAL_COL] != FLAG_SPINUP)
    idx  = df_raw.index[mask]

    if len(idx) == 0:
        print("  [SKIP] Fig 2: no active rows found")
        return

    raw_total     = df_raw.loc[mask, TOTAL_COL]
    clean_total   = df_cleaned.loc[mask, TOTAL_COL] if TOTAL_COL in df_cleaned.columns else raw_total.copy()
    branch_sum    = df_cleaned.loc[mask, SENSOR_COLS].sum(axis=1)
    reconstructed = branch_sum / max(1.0 - leakage_rate, 1e-9)

    flag_total = flag_df.loc[mask, TOTAL_COL]

    step = max(1, len(idx) // 4000)
    x_raw = np.arange(len(idx))[::step]
    x_lbl = idx[::step]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x_raw, raw_total.to_numpy(dtype=float)[::step],     color="#1f77b4", lw=1.2,
            label="Q_total (raw)", alpha=0.85)
    ax.plot(x_raw, clean_total.to_numpy(dtype=float)[::step],   color="#d62728", lw=1.2,
            label="Q_total (cleaned)", alpha=0.75)
    ax.plot(x_raw, reconstructed.to_numpy(dtype=float)[::step], color="#2ca02c", lw=1.0,
            linestyle="--", label="Σ branches/(1-λ)", alpha=0.70)

    adj_flags  = np.asarray(flag_total, dtype=np.int16) == FLAG_TOTAL_ADJUSTED
    anom_flags = np.asarray(flag_total, dtype=np.int16) == FLAG_SYSTEM_ANOMALY
    for i in range(len(adj_flags) - 1):
        xi = np.searchsorted(x_raw, i, side="right") - 1
        if xi < 0 or xi >= len(x_raw) - 1:
            continue
        if bool(adj_flags[i]):
            ax.axvspan(x_raw[xi], x_raw[min(xi + 1, len(x_raw) - 1)],
                       color="#ffe066", alpha=0.4)
        elif bool(anom_flags[i]):
            ax.axvspan(x_raw[xi], x_raw[min(xi + 1, len(x_raw) - 1)],
                       color="#ff4444", alpha=0.3)

    adj_patch  = mpatches.Patch(color="#ffe066", alpha=0.6, label="Total Adjusted (Scenario B)")
    anom_patch = mpatches.Patch(color="#ff4444", alpha=0.5, label="System Anomaly (Scenario B fused)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [adj_patch, anom_patch], labels + [adj_patch.get_label(), anom_patch.get_label()],
              fontsize=9, loc="upper right", framealpha=0.85)

    ax.set_xlabel("Timestep (active rows)", fontsize=11)
    ax.set_ylabel("Flow  (m³/h)", fontsize=11)
    ax.set_title("Total Meter: Raw vs Cleaned vs Branch Sum / (1-λ)", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig2_total_correction_timeseries.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 2 saved: {out_path.name}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _compute_mce_series(
    df_branch: pd.DataFrame,
    df_total:  pd.DataFrame,
    leakage_rate: float,
    mask: pd.Series,
) -> pd.Series:
    """
    MCE = Total*(1-λ) - Σ branches
    """
    branch_sum = df_branch.loc[mask, SENSOR_COLS].sum(axis=1)
    total      = df_total.loc[mask, TOTAL_COL]
    mce        = total * (1.0 - leakage_rate) - branch_sum
    return mce


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_mce_timeseries(
    df_raw: pd.DataFrame,
    scale_df: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    flag_df: pd.DataFrame,
    leakage_rate: float,
    is_active: pd.Series,
) -> None:
    """Plot mass-closure error for the raw, scaled-only and cleaned branch flows."""
    mask = is_active & (flag_df[SENSOR_COLS[0]] != FLAG_SPINUP)

    # Q_debiased
    Q_debiased = df_raw.copy()
    for sensor in SENSOR_COLS:
        if sensor in scale_df.columns:
            Q_debiased[sensor] = df_raw[sensor] / scale_df[sensor].replace(0, 1.0)

    mce_raw      = _compute_mce_series(df_raw,       df_raw, leakage_rate, mask)
    mce_debiased = _compute_mce_series(Q_debiased,   df_raw, leakage_rate, mask)
    mce_cleaned  = _compute_mce_series(df_cleaned,   df_raw, leakage_rate, mask)

    step  = max(1, len(mce_raw) // 4000)
    x_arr = np.arange(len(mce_raw))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x_arr[::step], mce_raw.to_numpy(dtype=float)[::step],      color="#1f77b4", lw=0.9,
            alpha=0.7, label=f"Raw  (μ={float(mce_raw.mean()):+.1f} m³/h)")
    ax.plot(x_arr[::step], mce_debiased.to_numpy(dtype=float)[::step], color="#ff7f0e", lw=0.9,
            alpha=0.7, label=f"Debiased-only  (μ={float(mce_debiased.mean()):+.1f} m³/h)")
    ax.plot(x_arr[::step], mce_cleaned.to_numpy(dtype=float)[::step],  color="#2ca02c", lw=1.1,
            label=f"Cleaned  (μ={float(mce_cleaned.mean()):+.1f} m³/h)")

    ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.6, label="MCE = 0 (perfect conservation)")

    ax.set_xlabel("Timestep (active rows)", fontsize=11)
    ax.set_ylabel("MCE = Total·(1-λ) - ΣBranch  (m³/h)", fontsize=11)
    ax.set_title("Mass Conservation Error — 3-way Ablation (Raw / Debiased-only / Cleaned)", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.88)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig3_mce_timeseries_ablation.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 3 saved: {out_path.name}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_mce_kde(
    df_raw: pd.DataFrame,
    scale_df: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    flag_df: pd.DataFrame,
    leakage_rate: float,
    is_active: pd.Series,
) -> None:
    mask = is_active & (flag_df[SENSOR_COLS[0]] != FLAG_SPINUP)

    Q_debiased = df_raw.copy()
    for sensor in SENSOR_COLS:
        if sensor in scale_df.columns:
            Q_debiased[sensor] = df_raw[sensor] / scale_df[sensor].replace(0, 1.0)

    mce_raw      = _compute_mce_series(df_raw,     df_raw, leakage_rate, mask).dropna()
    mce_debiased = _compute_mce_series(Q_debiased, df_raw, leakage_rate, mask).dropna()
    mce_cleaned  = _compute_mce_series(df_cleaned, df_raw, leakage_rate, mask).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    entries = [
        (mce_raw,      "#1f77b4", "Raw"),
        (mce_debiased, "#ff7f0e", "Debiased-only"),
        (mce_cleaned,  "#2ca02c", "Cleaned"),
    ]

    for mce_s, color, label in entries:
        arr = np.asarray(mce_s, dtype=float)
        if len(arr) < 5:
            continue
        p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        arr_clip = arr[(arr >= p1) & (arr <= p99)]
        if len(arr_clip) < 5:
            arr_clip = arr

        try:
            kde  = gaussian_kde(arr_clip, bw_method="scott")
            xs   = np.linspace(float(arr_clip.min()), float(arr_clip.max()), 800)
            dens = kde(xs)
            ax.plot(xs, dens, lw=2.0, color=color,
                    label=f"{label}  μ={float(arr.mean()):+.1f}  σ={float(arr.std()):.1f} m³/h")
            ax.fill_between(xs, dens, alpha=0.10, color=color)
        except Exception:
            pass

    ax.axvline(0, color="black", lw=1.1, linestyle="--", alpha=0.7, label="MCE = 0 (ideal)")

    ax.set_xlabel("MCE = Total·(1-λ) - ΣBranch  (m³/h)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        "Mass Conservation Error KDE — 3-way Ablation\n(1%–99% clipped)",
        fontsize=13,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig4_mce_kde_ablation.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 4 saved: {out_path.name}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage III — Data Analysis & Visualization")
    print("=" * 60)

    print("\n[Step 1/4] Loading data...")
    (flag_df, df_raw, df_cleaned,
     sensor_profile, cleaning_summary, calib, is_active) = load_all_data()

    print(f"  flag_df   : {flag_df.shape}")
    print(f"  df_raw    : {df_raw.shape}")
    print(f"  df_cleaned: {df_cleaned.shape}")
    print(f"  is_active : {int(is_active.sum())} active rows")
    print(f"  sensor_profile keys: {list(sensor_profile.keys())}")

    leakage_rate: float = float(
        calib.get("leakage", {}).get("rate", 0.0)
        if calib
        else cleaning_summary.get("leakage_rate", 0.0)
    )
    print(f"  leakage_rate = {leakage_rate:.6f}")

    print("\n[Step 2/4] Printing flag summary table...")
    print_summary_table(flag_df, is_active)

    print("\n[Step 3/4] Building scale matrix for debiased MCE...")
    scale_df = _build_scale_matrix_for_analysis(df_raw, calib)

    print("\n[Step 4/4] Generating figures...")
    if sensor_profile:
        plot_radar_chart(sensor_profile, cleaning_summary)
    else:
        print("  [SKIP] Fig 1: sensor_profile.json not found (run data_cleaning.py first)")

    plot_total_correction(flag_df, df_raw, df_cleaned, leakage_rate, is_active)
    plot_mce_timeseries(df_raw, scale_df, df_cleaned, flag_df, leakage_rate, is_active)
    plot_mce_kde(df_raw, scale_df, df_cleaned, flag_df, leakage_rate, is_active)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
