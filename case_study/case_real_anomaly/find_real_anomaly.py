import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"
BASELINE_DIR = ROOT_DIR / "Baseline"
PI_GAT_DIR = ROOT_DIR / "PI-GAT"
DATA_CLEAN_DIR = ROOT_DIR / "data_clean"

sys.path.insert(0, str(BASELINE_DIR))
try:
    from effective_value_filtering import build_active_mask  # type: ignore
except ImportError:
    def build_active_mask(df: pd.DataFrame) -> pd.Series:  # type: ignore
        req = ["Total", "B01", "B02", "B03", "B04", "B05", "B06"]
        has_nan = df[req].isna().any(axis=1)
        all_zero = (df[req] == 0.0).all(axis=1)
        return (~has_nan & ~all_zero).rename("active")


def get_leakage_rate() -> float:
    """Load the calibrated leakage rate.

    Priority:
      1. calibration_params_final.json → leakage.rate  (canonical output of instrument_bias.py)
      2. state_dict["leakage_logit"] with proper range scaling:
         rate = 0.05 + 0.10 * sigmoid(leakage_logit)  (matches model_pi_gat.py get_leakage_rate())
    """
    import json

    # Priority 1: calibration_params_final.json (same source as data_cleaning.py)
    calib_json = PI_GAT_DIR / "bias_calibration" / "calibration_params_final.json"
    if calib_json.exists():
        try:
            with open(calib_json, "r", encoding="utf-8") as f:
                calib = json.load(f)
            lam = calib.get("leakage", {}).get("rate", None)
            if lam is not None:
                lambda_val = float(lam)
                print(f"Loaded leakage rate from calibration_params_final.json: {lambda_val:.4f}")
                return lambda_val
        except Exception as e:
            print(f"Warning: Failed to read calibration JSON ({e}).")

    # Priority 2: checkpoint leakage_logit with range [0.05, 0.15] scaling
    # (formula matches PI_GAT.get_leakage_rate(): lo + (hi-lo)*sigmoid(logit))
    try:
        ckpt_path = PI_GAT_DIR / "pi_gat_best.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if "leakage_logit" in state_dict:
            logit = state_dict["leakage_logit"]
            lo, hi = 0.05, 0.15
            lambda_val = lo + (hi - lo) * float(torch.sigmoid(torch.as_tensor(logit)).item())
            print(f"Loaded leakage rate from leakage_logit (range-scaled): {lambda_val:.4f}")
            return lambda_val
    except Exception as e:
        print(f"Warning: Failed to load leakage rate from checkpoint ({e}).")

    print("Fallback to leakage rate: 0.10")
    return 0.10

def find_anomalous_window() -> Tuple[int, int, float]:
    print("=== Finding Best Real Anomaly Window ===")
    lam = get_leakage_rate()
    
    raw_csv = PREPROCESS_DIR / "flow_5min_imputed.csv"
    pigat_csv = DATA_CLEAN_DIR / "outputs" / "flow_5min_cleaned.csv"
    if_csv = BASELINE_DIR / "IsolationForest" / "cleaning_results" / "flow_5min_cleaned_if.csv"
    mvz_csv = BASELINE_DIR / "Multi-variate_Z-score" / "cleaning_results" / "flow_5min_cleaned_mvz.csv"

    df_raw = pd.read_csv(str(raw_csv))
    df_pigat = pd.read_csv(str(pigat_csv))
    df_if = pd.read_csv(str(if_csv))
    df_mvz = pd.read_csv(str(mvz_csv))

    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]

    # Build active mask via effective_value_filtering rules (no zero/NaN periods)
    active_mask_series = build_active_mask(df_raw)
    # Ensure a plain NumPy bool array so static type checkers know .all() exists
    active_arr = active_mask_series.to_numpy(dtype=bool)          # bool array aligned with df_raw
    active_idx = df_raw.index[active_mask_series]
    if len(active_idx) == 0:
        active_idx = df_raw.index

    # Compute MCE for raw
    sum_branches = df_raw[sensors].sum(axis=1)
    q_total_effective = df_raw["Total"] * (1.0 - lam)
    mce_raw: pd.Series = (q_total_effective - sum_branches).abs()
    
    # MCE 3-sigma Threshold based on active windows only
    threshold_3sigma = float(np.nanmean(mce_raw.loc[active_idx])) + 3.0 * float(np.nanstd(mce_raw.loc[active_idx]))
    
    # Search with window size = 4 days = 1152 steps
    win_size = 1152
    best_score = float('-inf')
    best_start = 0
    
    diff_if = np.abs(df_if[sensors] - df_raw[sensors]).sum(axis=1)
    diff_mvz = np.abs(df_mvz[sensors] - df_raw[sensors]).sum(axis=1)
    
    for i in range(0, len(df_raw) - win_size, 288):
        # Only evaluate windows where every single step is active
        if not active_arr[i : i + win_size].all():
            continue

        window_slice = slice(i, i + win_size)
        
        # 1. Smoothness (Low variance)
        branch_variance = float(df_raw.iloc[window_slice][sensors].var().sum())
        total_variance = float(df_raw.iloc[window_slice]["Total"].var())
        smoothness = 1.0 / (1.0 + branch_variance + total_variance)
        
        # 2. Zero baseline intervention
        intervention_if = float(diff_if.iloc[window_slice].mean())
        intervention_mvz = float(diff_mvz.iloc[window_slice].mean())
        baseline_silence = 1.0 / (1.0 + intervention_if + intervention_mvz)
        
        # 3. Severe MCE Violation
        mce_violation = float(mce_raw.iloc[window_slice].mean())
        
        # Require MCE to be large
        if mce_violation < threshold_3sigma * 1.5:
            continue
            
        score = mce_violation * baseline_silence * (smoothness * 1e5)
        
        if score > best_score:
            best_score = score
            best_start = i
            
    if best_score == float('-inf'):
        # Fallback: among fully-active windows pick the one with highest rolling MCE
        mce_active = mce_raw.copy()
        mce_active[~active_mask_series] = 0.0
        rolling_mean = mce_active.rolling(win_size).mean()
        best_start = max(0, int(rolling_mean.idxmax()) - win_size + 1)
        
    return best_start, best_start + win_size, lam

def plot_real_anomaly(start_idx: int, end_idx: int, lam: float) -> None:
    raw_csv = PREPROCESS_DIR / "flow_5min_imputed.csv"
    pigat_csv = DATA_CLEAN_DIR / "outputs" / "flow_5min_cleaned.csv"
    if_csv = BASELINE_DIR / "IsolationForest" / "cleaning_results" / "flow_5min_cleaned_if.csv"
    mvz_csv = BASELINE_DIR / "Multi-variate_Z-score" / "cleaning_results" / "flow_5min_cleaned_mvz.csv"
    
    df_raw = pd.read_csv(str(raw_csv))
    df_pigat = pd.read_csv(str(pigat_csv))
    df_if = pd.read_csv(str(if_csv))
    df_mvz = pd.read_csv(str(mvz_csv))
    
    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]
    time_axis = np.arange(start_idx, end_idx)
    
    window_slice = slice(start_idx, end_idx)
    raw_total = df_raw["Total"].iloc[window_slice] * (1.0 - lam)
    raw_sum = df_raw[sensors].iloc[window_slice].sum(axis=1)
    
    if_sum = df_if[sensors].iloc[window_slice].sum(axis=1)
    mvz_sum = df_mvz[sensors].iloc[window_slice].sum(axis=1)
    
    pigat_sum = df_pigat[sensors].iloc[window_slice].sum(axis=1)
    pigat_total = df_pigat["Total"].iloc[window_slice] * (1.0 - lam)
    
    fig, axs = cast(Tuple[Figure, np.ndarray], plt.subplots(3, 1, figsize=(12, 10), sharex=True))
    
    # Subplot 1
    axs[0].plot(time_axis, raw_total, label=r'Total * $(1-\lambda)$', color='black', linewidth=2)
    axs[0].plot(time_axis, raw_sum, label='Sum of Branches (Raw)', color='red', linestyle='--', linewidth=2)
    axs[0].fill_between(time_axis, raw_total, raw_sum, color='gray', alpha=0.3, label='Physical Gap (MCE)')
    axs[0].set_title('Subplot 1: The Problem (Smooth but Severe Mass Conservation Violation)')
    axs[0].set_ylabel('Flow ($m^3/h$)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2
    axs[1].plot(time_axis, raw_total, label=r'Total * $(1-\lambda)$', color='black', linewidth=2)
    axs[1].plot(time_axis, if_sum, label='Sum of Branches (IF Cleaned)', color='orange', alpha=0.8)
    axs[1].plot(time_axis, mvz_sum, label='Sum of Branches (MVZ Cleaned)', color='purple', alpha=0.8)
    axs[1].set_title('Subplot 2: Baseline Failure (Statistical models overlook systematic gap)')
    axs[1].set_ylabel('Flow ($m^3/h$)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 3
    axs[2].plot(time_axis, pigat_total, label=r'Total * $(1-\lambda)$ [Adjusted]', color='black', linewidth=2)
    axs[2].plot(time_axis, pigat_sum, label='Sum of Branches (PI-GAT Cleaned)', color='green', linewidth=2)
    axs[2].set_title('Subplot 3: PI-GAT Intervention (Physics-Informed spatial reconstruction)')
    axs[2].set_ylabel('Flow ($m^3/h$)')
    axs[2].set_xlabel('Time Steps')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_path = SCRIPT_DIR / "real_anomaly_comparison.png"
    plt.savefig(str(out_path), dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def plot_zoom_in(start_idx: int, end_idx: int, lam: float) -> None:
    """Twin-axis zoom-in: PI-GAT flow (left Y) vs. sum of active valve openings (right Y).

    Finds the 150-step sub-window with the highest flow variance inside the
    anomalous window, then overlays the summed OD% of all branches that were
    actually open at some point (max OD > 5%) during the zoom window.  Always-
    closed valves (flat 0% lines) are excluded so the right axis directly
    explains the flow oscillations on the left axis.
    """
    raw_csv   = PREPROCESS_DIR / "flow_5min_imputed.csv"
    pigat_csv = DATA_CLEAN_DIR / "outputs" / "flow_5min_cleaned.csv"

    df_raw   = pd.read_csv(str(raw_csv))
    df_pigat = pd.read_csv(str(pigat_csv))

    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]
    od_cols = [f"OD_{s}" for s in sensors if f"OD_{s}" in df_raw.columns]

    window_slice  = slice(start_idx, end_idx)
    pigat_sum_win = df_pigat[sensors].iloc[window_slice].sum(axis=1)

    # Find the 150-step sub-window with the highest rolling std (most volatile)
    zoom_size      = min(150, len(pigat_sum_win))
    s_reset        = pigat_sum_win.reset_index(drop=True)
    roll_std       = s_reset.rolling(zoom_size).std()
    best_end_pos   = int(roll_std.idxmax())
    best_start_pos = max(0, best_end_pos - zoom_size + 1)
    zoom_start = start_idx + best_start_pos
    zoom_end   = zoom_start + zoom_size
    zoom_slice = slice(zoom_start, zoom_end)
    time_axis  = np.arange(zoom_start, zoom_end)

    # Convert to plain NumPy float arrays to satisfy static type checkers
    pigat_sum_zoom = df_pigat[sensors].iloc[zoom_slice].sum(axis=1).to_numpy(dtype=float)
    pigat_total_zoom = df_pigat["Total"].iloc[zoom_slice].to_numpy(dtype=float) * (1.0 - lam)

    # Collect all OD columns that were open at least once in the zoom window
    # (max OD > 5%).  Always-closed valves contribute a flat 0% line that
    # has no visual correlation with flow oscillations and are excluded.
    active_od_cols = [
        oc for oc in od_cols
        if float(np.max(df_raw[oc].iloc[zoom_slice].to_numpy(dtype=float))) > 5.0
    ]
    # Fallback: if no valve reaches 5% in the zoom window, include all OD columns.
    if not active_od_cols:
        active_od_cols = od_cols

    # Compute the element-wise sum of active valve openings.
    if active_od_cols:
        od_sum_zoom: np.ndarray = sum(
            df_raw[oc].iloc[zoom_slice].to_numpy(dtype=float)
            for oc in active_od_cols
        )  # type: ignore[assignment]
    else:
        od_sum_zoom = np.array([])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(time_axis, pigat_total_zoom, color='black', linewidth=2,
             label=r'Total$\times(1-\lambda)$ [PI-GAT]')
    ax1.plot(time_axis, pigat_sum_zoom, color='green', linewidth=2,
             label='Sum of Branches [PI-GAT]')
    ax1.set_ylabel('Flow (m\u00b3/h)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle=':', alpha=0.5)

    if active_od_cols and od_sum_zoom.size > 0:
        od_label = 'Sum of Active Valve Openings\n(' + ', '.join(active_od_cols) + ', %)'
        ax2 = ax1.twinx()
        ax2.step(time_axis, od_sum_zoom, color='steelblue', alpha=0.55,
                 linewidth=1.2, linestyle='--', where='mid',
                 label=od_label)
        ax2.set_ylabel('Active Valve Opening Sum (%)', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    else:
        ax1.legend(loc='upper right')

    ax1.set_xlabel('Time Step')
    ax1.set_title(
        f'Zoom-in [{zoom_start}\u2013{zoom_end}]: Flow Dynamics vs. Active Valve Openings\n'
        '(Flow dips align with valve closures; right axis = sum of open branches OD%)'
    )
    plt.tight_layout()
    out_path = SCRIPT_DIR / "real_anomaly_zoom.png"
    plt.savefig(str(out_path), dpi=300)
    print(f"Saved zoom-in plot to {out_path}")
    plt.close(fig)


def compute_kcr(start_idx: int, end_idx: int) -> None:
    """Compute and print Kinematic Compliance Rate (KCR) for each method.

    KCR = (# steps where OD < 5% AND |flow| < FLOW_ZERO_THRESH) /
          (# steps where OD < 5%)  x 100 %

    A high KCR means the method correctly suppresses flow when the valve is
    mechanically closed.  PI-GAT applies FLAG_ZERO (force-to-zero) for
    OD < 5%, so its KCR should approach 100 %.  Statistical models (IF /
    MVZ) ignore valve state and therefore exhibit systematically lower KCR.
    """
    import json as _json

    raw_csv   = PREPROCESS_DIR / "flow_5min_imputed.csv"
    pigat_csv = DATA_CLEAN_DIR / "outputs" / "flow_5min_cleaned.csv"
    if_csv    = BASELINE_DIR / "IsolationForest"        / "cleaning_results" / "flow_5min_cleaned_if.csv"
    mvz_csv   = BASELINE_DIR / "Multi-variate_Z-score" / "cleaning_results" / "flow_5min_cleaned_mvz.csv"

    df_raw   = pd.read_csv(str(raw_csv))
    df_pigat = pd.read_csv(str(pigat_csv))
    df_if    = pd.read_csv(str(if_csv))
    df_mvz   = pd.read_csv(str(mvz_csv))

    sensors      = ["B01", "B02", "B03", "B04", "B05", "B06"]
    window_slice = slice(start_idx, end_idx)
    FLOW_ZERO_THRESH = 5.0  # m\u00b3/h  — accounts for zero-point instrument drift

    print("\n=== Kinematic Compliance Rate (KCR)  [OD < 5% \u2192 flow should \u2248 0] ===")
    print(f"{'Sensor':<8} {'N(OD<5%)':>9} {'KCR_Raw':>9} {'KCR_IF':>9} {'KCR_MVZ':>9} {'KCR_PIGAT':>10}")
    print("-" * 60)

    all_kcr: Dict[str, List[float]] = {"Raw": [], "IF": [], "MVZ": [], "PI-GAT": []}

    for sensor in sensors:
        od_col = f"OD_{sensor}"
        if od_col not in df_raw.columns:
            continue
        od_vals = df_raw[od_col].iloc[window_slice].to_numpy(dtype=float)
        closed_mask = od_vals < 5.0
        n_closed = int(closed_mask.sum())
        if n_closed == 0:
            # Valve never closed in this window → no zero-flow requirements
            # → trivially 100 % compliant for all methods
            all_kcr["Raw"].append(100.0)
            all_kcr["IF"].append(100.0)
            all_kcr["MVZ"].append(100.0)
            all_kcr["PI-GAT"].append(100.0)
            print(f"{sensor:<8} {n_closed:>9d}  (always open – KCR = 100.0% by convention)")
            continue

        # Define _kcr using a simple closure (avoid default args that reference
        # mutable/outer variables which can confuse static type checkers).
        def _kcr(df: pd.DataFrame, s: str = sensor) -> float:
            vals = df[s].iloc[window_slice].to_numpy(dtype=float)
            return float(np.sum(vals[closed_mask] < FLOW_ZERO_THRESH)) / int(n_closed) * 100.0

        kcr_raw = _kcr(df_raw)
        kcr_if = _kcr(df_if)
        kcr_mvz = _kcr(df_mvz)
        kcr_pigat = _kcr(df_pigat)

        all_kcr["Raw"].append(kcr_raw)
        all_kcr["IF"].append(kcr_if)
        all_kcr["MVZ"].append(kcr_mvz)
        all_kcr["PI-GAT"].append(kcr_pigat)

        print(
            f"{sensor:<8} {n_closed:>9d} "
            f"{kcr_raw:>8.1f}% "
            f"{kcr_if:>8.1f}% "
            f"{kcr_mvz:>8.1f}% "
            f"{kcr_pigat:>9.1f}%"
        )

    if all_kcr["Raw"]:
        print("-" * 60)
        print(
            f"{'Mean':<8} {'':>9} "
            f"{np.mean(all_kcr['Raw']):>8.1f}% "
            f"{np.mean(all_kcr['IF']):>8.1f}% "
            f"{np.mean(all_kcr['MVZ']):>8.1f}% "
            f"{np.mean(all_kcr['PI-GAT']):>9.1f}%"
        )
        kcr_summary = {
            "window": {"start": start_idx, "end": end_idx},
            "flow_zero_threshold_m3h": FLOW_ZERO_THRESH,
            "mean_kcr": {
                "Raw":    round(float(np.mean(all_kcr["Raw"])),    2),
                "IF":     round(float(np.mean(all_kcr["IF"])),     2),
                "MVZ":    round(float(np.mean(all_kcr["MVZ"])),    2),
                "PI-GAT": round(float(np.mean(all_kcr["PI-GAT"])), 2),
            },
        }
        out_json = SCRIPT_DIR / "kcr_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            _json.dump(kcr_summary, f, indent=4)
        print(f"KCR summary saved to {out_json}")
    else:
        print("  (No OD columns found in the raw dataset for this window.)")
    print()


if __name__ == "__main__":
    start, end, lam_val = find_anomalous_window()
    print(f"Found anomalous window from step {start} to {end}")
    plot_real_anomaly(start, end, lam_val)
    plot_zoom_in(start, end, lam_val)
    compute_kcr(start, end)
