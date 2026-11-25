from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SCRIPT_DIR     = Path(__file__).parent
ROOT_DIR       = SCRIPT_DIR.parent
PI_GAT_DIR     = ROOT_DIR / "PI-GAT"
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"

sys.path.insert(0, str(PI_GAT_DIR))

from graph_utils import build_graph_from_csv          # type: ignore
from model_pi_gat import PI_GAT                       # type: ignore
from timeseries_dataset import IrrigationTimeWindowDataset  # type: ignore

def _unwrap_pi_gat(model: torch.nn.Module) -> PI_GAT:
    """Unwrap torch.compile() wrapper so Pylance sees PI_GAT methods."""
    return cast(PI_GAT, getattr(model, "_orig_mod", model))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SENSOR_COLS   = ["B01", "B02", "B03", "B04", "B05", "B06"]
TOTAL_COL     = "Total"
ALL_FLOW_COLS = SENSOR_COLS + [TOTAL_COL]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
FLAG_IGNORE           = -2   # Identify prolonged missing runs as rest periods.
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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
WIN    = 12
BATCH  = 64
HEADS  = 8
HIDDEN = 64
INFEAT = 2

ΔOD_TRANSITION    = 5.0
VALVE_GUARD       = 1      # Number of protected steps on each side of a valve transition.
W_OD_STABLE       = 6
ΔOD_STABLE_THRESH = 10.0
VALVE_CLOSED_THRESH = 5.0  # Closed-valve threshold, in percentage points.

K_SPIKE             = 3.0
β_NORMAL            = 0.5
RELATIVE_NOISE_RATIO      = 0.10   # Lower bound proportional to the branch's mean flow.
GLOBAL_HYDRAULIC_DEADBAND = 80.0   # Global hydraulic deadband in m³/h.
MAX_THRESHOLD_WARN  = 200.0

τ_MCE_REL  = 0.15   # Relative network mass-closure threshold.
τ_MCE_ABS  = 50.0   # Absolute network mass-closure threshold.
MIN_DIAG_FOR_B = 2

SCENARIO_B_FUSE = 0.30

W_MIN_DRIFT   = 12    # Minimum duration of a persistent drift in time steps.
DRIFT_MULT    = 1.5
W_SMOOTH      = 12

OD_CORR_SIGNIFICANT = 0.6


# ===========================================================================
# ===========================================================================

def _load_zscore_stats(stats_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("mean", {}), obj.get("std", {})


def _load_calib_params(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_scale_matrix(df_raw: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """Build the per-record Stage-II scale matrix for the six branches.

    Divide raw branch flow by the scale. Unit scale applies through 5%
    opening; the three open bins use the values stored in the calibration JSON."""
    valves = calib.get("bias", {}).get("valves", {})
    scale_df = pd.DataFrame(1.0, index=df_raw.index, columns=SENSOR_COLS, dtype=float)

    for sensor in SENSOR_COLS:
        od_col = f"OD_{sensor}"
        if od_col not in df_raw.columns or sensor not in valves:
            continue

        scales_list: List[float] = valves[sensor].get("scale", [1.0, 1.0, 1.0])
        od_pct = df_raw[od_col].clip(0.0, 100.0).fillna(100.0)

        bin_idx = pd.Series(0, index=df_raw.index, dtype=int)
        bin_idx[od_pct > 5.0]      = 1
        bin_idx[od_pct > 33.3333]  = 2
        bin_idx[od_pct > 66.6667]  = 3

        s = pd.Series(1.0, index=df_raw.index, dtype=float)
        for b in [1, 2, 3]:
            mask = bin_idx == b
            if mask.any():
                s[mask] = float(scales_list[b - 1])

        scale_df[sensor] = s

    return scale_df


# ===========================================================================
# ===========================================================================

def run_full_inference(
    model: PI_GAT,
    dataset: IrrigationTimeWindowDataset,
    sensor2node_idx: Dict[str, int],
    n_nodes: int,
    device: str,
    batch_size: int = BATCH,
) -> pd.DataFrame:
    """Predict nodal flow over the valid time windows and undo standardisation.

    Return the branch predictions in m³/h, indexed by the original row number."""
    from typing import cast
    model.eval()

    loader = DataLoader(
        cast(Any, dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    pred_records: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch)

            node_hat_z    = out["node_hat"].cpu()
            node_mean     = batch.node_mean.cpu()
            node_std      = batch.node_std.cpu()
            node_hat_m3ph = node_hat_z * node_std + node_mean

            t_index_attr = batch.t_index
            if isinstance(t_index_attr, torch.Tensor):
                raw_list: List[int] = t_index_attr.cpu().tolist()
                t_index_list: List[int] = [
                    int(raw_list[k]) - k * n_nodes for k in range(len(raw_list))
                ]
            elif isinstance(t_index_attr, (list, tuple)):
                raw_list2 = [int(v) for v in t_index_attr]
                t_index_list = [
                    raw_list2[k] - k * n_nodes for k in range(len(raw_list2))
                ]
            else:
                t_index_list = [int(t_index_attr)]

            for g_idx in range(len(t_index_list)):
                t_val = t_index_list[g_idx]
                start = g_idx * n_nodes
                end   = start + n_nodes
                node_preds = node_hat_m3ph[start:end]

                rec: Dict[str, Any] = {"t_index": t_val}
                for sensor, ni in sensor2node_idx.items():
                    if sensor in SENSOR_COLS and ni < len(node_preds):
                        rec[sensor] = float(node_preds[ni].item())
                pred_records.append(rec)

    pred_df = (
        pd.DataFrame(pred_records)
        .set_index("t_index")
        .sort_index()
    )
    pred_df = pred_df[~pred_df.index.duplicated(keep="last")]
    for s in SENSOR_COLS:
        if s not in pred_df.columns:
            pred_df[s] = np.nan
    return pred_df[SENSOR_COLS]


# ===========================================================================
# ===========================================================================

def compute_valve_state_masks(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-branch masks for guarded valve transitions and stable opening.

    A transition is a change larger than DELTA_OD_TRANSITION with a VALVE_GUARD
    step on either side. Opening is stable when its range over the centred
    window is below DELTA_OD_STABLE_THRESH."""
    is_valve_transition = pd.DataFrame(False, index=df_raw.index, columns=SENSOR_COLS)
    is_od_stable        = pd.DataFrame(True,  index=df_raw.index, columns=SENSOR_COLS)

    for sensor in SENSOR_COLS:
        od_col = f"OD_{sensor}"
        if od_col not in df_raw.columns:
            continue
        od_vals   = df_raw[od_col].clip(0.0, 100.0).fillna(100.0)
        delta_od  = od_vals.diff().abs().fillna(0.0)
        raw_jump  = delta_od.gt(ΔOD_TRANSITION)
        is_valve_transition[sensor] = (
            raw_jump
            | raw_jump.shift(VALVE_GUARD,  fill_value=False)
            | raw_jump.shift(-VALVE_GUARD, fill_value=False)
        )
        win      = 2 * W_OD_STABLE + 1
        roll_max = od_vals.rolling(win, center=True, min_periods=1).max()
        roll_min = od_vals.rolling(win, center=True, min_periods=1).min()
        is_od_stable[sensor] = (roll_max - roll_min).lt(ΔOD_STABLE_THRESH)

    return is_valve_transition, is_od_stable


# ===========================================================================
# ===========================================================================

def compute_residual_baseline(
    Q_input: pd.DataFrame,
    pred_aligned: pd.DataFrame,
    is_valve_transition: pd.DataFrame,
    is_od_stable: pd.DataFrame,
    is_active_row: pd.Series,
    credibility_path: Optional[Path],
    n_total: int,
    valid_t_list: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Estimate each branch residual mean on stable high-credibility records.

    Select the largest of the statistical threshold, the local relative-flow
    threshold and the global hydraulic deadband."""
    valid_mask_row = pd.Series(False, index=Q_input.index)
    valid_mask_row.iloc[np.array(valid_t_list, dtype=int)] = True

    high_t_set: set = set()
    if credibility_path is not None and credibility_path.exists():
        cred_df = pd.read_csv(credibility_path)
        if "t_index" in cred_df.columns and "credibility" in cred_df.columns:
            high_t_set = set(
                cred_df.loc[cred_df["credibility"] == "high", "t_index"].astype(int).tolist()
            )
    if len(high_t_set) == 0:
        high_t_set = set(valid_t_list)

    high_mask = pd.Series(False, index=Q_input.index)
    high_t_arr = np.array(sorted(high_t_set), dtype=int)
    in_range   = (high_t_arr >= 0) & (high_t_arr < n_total)
    high_mask.iloc[high_t_arr[in_range]] = True

    R = pd.DataFrame(np.nan, index=Q_input.index, columns=SENSOR_COLS, dtype=float)
    for sensor in SENSOR_COLS:
        if sensor in Q_input.columns and sensor in pred_aligned.columns:
            R[sensor] = Q_input[sensor] - pred_aligned[sensor]

    μ_R:             Dict[str, float] = {}
    node_thresholds: Dict[str, float] = {}

    for sensor in SENSOR_COLS:
        baseline_mask = (
            high_mask
            & is_od_stable[sensor]
            & ~is_valve_transition[sensor]
            & is_active_row
            & valid_mask_row
            & R[sensor].notna()
        )
        R_base = R.loc[baseline_mask, sensor]

        if len(R_base) < 2:
            μ_R[sensor]            = 0.0
            node_thresholds[sensor] = GLOBAL_HYDRAULIC_DEADBAND
            continue

        mu_i    = float(R_base.mean())
        dev     = (R_base - mu_i).abs()
        mu_dev  = float(dev.mean())
        sig_dev = float(dev.std(ddof=1)) if len(dev) > 1 else 0.0
        stat_thr = float(mu_dev + K_SPIKE * sig_dev)

        # Global hydraulic deadband in m³/h.
        mu_q_series        = Q_input.loc[baseline_mask, sensor].dropna()
        mu_q               = float(mu_q_series.mean()) if len(mu_q_series) > 0 else 0.0
        local_relative_thr = RELATIVE_NOISE_RATIO * mu_q
        thr_i              = max(stat_thr, local_relative_thr, GLOBAL_HYDRAULIC_DEADBAND)

        μ_R[sensor]            = mu_i
        node_thresholds[sensor] = thr_i

        if thr_i > MAX_THRESHOLD_WARN:
            print(f"  [WARNING] Threshold[{sensor}] = {thr_i:.1f} m³/h > {MAX_THRESHOLD_WARN:.0f} m³/h!")

    return μ_R, node_thresholds, R


# ===========================================================================
# ===========================================================================

def detect_dynamic_drift(
    R: pd.DataFrame,
    μ_R: Dict[str, float],
    node_thresholds: Dict[str, float],
    is_valve_transition: pd.DataFrame,
    is_active_row: pd.Series,
    valid_mask_row: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[int, int, float]]]]:
    """Scan consecutive eligible records for sustained same-sign residual drift.

    Exclude inactive samples, valve transitions and missing predictions.
    Require the minimum duration and a mean absolute deviation above the
    configured multiple of the branch threshold."""
    n_total = len(R)
    drift_correction = pd.DataFrame(0.0, index=R.index, columns=SENSOR_COLS, dtype=float)
    drift_events: Dict[str, List[Tuple[int, int, float]]] = {s: [] for s in SENSOR_COLS}

    active_valid_pos = np.where(
        np.asarray(is_active_row, dtype=bool) & np.asarray(valid_mask_row, dtype=bool)
    )[0]

    for sensor in SENSOR_COLS:
        thr_c   = DRIFT_MULT * node_thresholds.get(sensor, GLOBAL_HYDRAULIC_DEADBAND)
        mu_i    = μ_R.get(sensor, 0.0)
        R_ex_np = (R[sensor] - mu_i).to_numpy(dtype=float)
        trans_np = is_valve_transition[sensor].to_numpy(dtype=bool)

        N_active = len(active_valid_pos)
        i = 0
        while i <= N_active - W_MIN_DRIFT:
            win_abs = active_valid_pos[i: i + W_MIN_DRIFT]

            if trans_np[win_abs].any():
                i += 1
                continue
            R_w = R_ex_np[win_abs]
            if not np.isfinite(R_w).all():
                i += 1
                continue

            all_pos  = bool((R_w > 0).all())
            all_neg  = bool((R_w < 0).all())
            mean_abs = float(np.abs(R_w.mean()))

            if (all_pos or all_neg) and mean_abs > thr_c:
                sign_dir = 1 if all_pos else -1
                j = i + W_MIN_DRIFT
                while j < N_active:
                    nxt = int(active_valid_pos[j])
                    if (not trans_np[nxt]
                            and np.isfinite(R_ex_np[nxt])
                            and sign_dir * R_ex_np[nxt] > 0):
                        j += 1
                    else:
                        break

                win_full = active_valid_pos[i:j]
                R_win    = R_ex_np[win_full]
                ser      = pd.Series(R_win, dtype=float)
                smooth   = ser.rolling(W_SMOOTH, center=True, min_periods=1).mean().to_numpy()

                drift_correction.loc[drift_correction.index[win_full], sensor] = smooth

                drift_events[sensor].append(
                    (int(win_full[0]), int(win_full[-1]), float(np.abs(smooth).max()))
                )
                i = j
            else:
                i += 1

    return drift_correction, drift_events


# ===========================================================================
# ===========================================================================

def apply_physics_assimilation(
    df_raw: pd.DataFrame,
    pred_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    calib: dict,
    node_thresholds: Dict[str, float],
    μ_R: Dict[str, float],
    drift_correction: pd.DataFrame,
    drift_events: Dict[str, List[Tuple[int, int, float]]],
    is_valve_transition: pd.DataFrame,
    is_od_stable: pd.DataFrame,
    valid_t_list: List[int],
    credibility_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Assimilate the branch records with the Stage-III correction rules.

    Compare the scale-adjusted observation and the inverse-standardised node
    prediction in m³/h. Apply drift correction before assessing the network
    imbalance and preserve transitions and closed-valve constraints."""
    n_total      = len(df_raw)
    leakage_rate = float(calib.get("leakage", {}).get("rate", 0.0))

    Q_input = df_raw.copy()
    for sensor in SENSOR_COLS:
        if sensor in df_raw.columns and sensor in scale_df.columns:
            Q_input[sensor] = df_raw[sensor] / scale_df[sensor].replace(0, 1.0)

    pred_aligned = pd.DataFrame(np.nan, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    pred_t_arr = np.array(pred_df.index.tolist(), dtype=int)
    in_range   = (pred_t_arr >= 0) & (pred_t_arr < n_total)
    if in_range.any():
        locs     = pred_t_arr[in_range]
        pred_sub = pred_df.iloc[np.where(in_range)[0]]
        for sensor in SENSOR_COLS:
            if sensor in pred_df.columns:
                pred_aligned.loc[pred_aligned.index[locs], sensor] = (
                    pred_sub[sensor].to_numpy(dtype=float)
                )

    valid_mask_row = pd.Series(False, index=df_raw.index)
    valid_mask_row.iloc[np.array(valid_t_list, dtype=int)] = True

    is_active_row = pd.Series(False, index=df_raw.index)
    if credibility_path is not None and credibility_path.exists():
        _cred_df    = pd.read_csv(credibility_path, usecols=["t_index"])
        _active_arr = _cred_df["t_index"].astype(int).to_numpy()
        _in_range   = (_active_arr >= 0) & (_active_arr < n_total)
        is_active_row.iloc[_active_arr[_in_range]] = True

    Q_input_c = Q_input.copy()
    for sensor in SENSOR_COLS:
        dc_mask = (
            drift_correction[sensor].ne(0)
            & is_active_row
            & ~is_valve_transition[sensor]
            & valid_mask_row
            & pred_aligned[sensor].notna()
        )
        if dc_mask.any():
            Q_input_c.loc[dc_mask, sensor] -= drift_correction.loc[dc_mask, sensor]

    MCE_c    = df_raw[TOTAL_COL] * (1.0 - leakage_rate) - Q_input_c[SENSOR_COLS].sum(axis=1)
    MCE_c_rel = MCE_c.abs() / df_raw[TOTAL_COL].clip(lower=1.0)

    R_c = pd.DataFrame(np.nan, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    for sensor in SENSOR_COLS:
        R_c[sensor] = Q_input_c[sensor] - pred_aligned[sensor]

    mu_arr    = np.array([μ_R.get(s, 0.0) for s in SENSOR_COLS], dtype=float)
    R_c_excess = R_c[SENSOR_COLS] - mu_arr

    flag_df = pd.DataFrame(
        FLAG_SPINUP,
        index=df_raw.index,
        columns=ALL_FLOW_COLS,
        dtype=np.int16,
    )
    flag_df.loc[valid_mask_row, :]                     = FLAG_NORMAL
    flag_df.loc[valid_mask_row & ~is_active_row, :]    = FLAG_IGNORE

    for sensor in SENSOR_COLS:
        dc_rows = (
            drift_correction[sensor].ne(0)
            & is_active_row
            & valid_mask_row
            & ~is_valve_transition[sensor]
        )
        if dc_rows.any():
            flag_df.loc[dc_rows, sensor] = FLAG_DYNAMIC_DRIFT

    Q_input_c_out = Q_input_c.copy()
    for sensor in SENSOR_COLS:
        od_col = f"OD_{sensor}"
        if od_col not in df_raw.columns:
            continue
        od_arr = df_raw[od_col].clip(0.0, 100.0).fillna(100.0)
        zero_mask = valid_mask_row & is_active_row & od_arr.lt(VALVE_CLOSED_THRESH)
        if zero_mask.any():
            Q_input_c_out.loc[zero_mask, sensor] = 0.0
            flag_df.loc[zero_mask, sensor] = FLAG_ZERO

    for sensor in SENSOR_COLS:
        trans_mask = (
            valid_mask_row & is_active_row & is_valve_transition[sensor]
            & (flag_df[sensor] != FLAG_ZERO)
        )
        if trans_mask.any():
            flag_df.loc[trans_mask, sensor] = FLAG_VALVE_TRANSITION

    diagnosable_mat = np.column_stack([
        (np.asarray(valid_mask_row, dtype=bool) & np.asarray(is_active_row, dtype=bool)
         & (np.asarray(flag_df[s], dtype=np.int16) != FLAG_ZERO)
         & (np.asarray(flag_df[s], dtype=np.int16) != FLAG_VALVE_TRANSITION))
        for s in SENSOR_COLS
    ])  # [T, 6]

    mce_violated = (
        np.asarray(valid_mask_row, dtype=bool)
        & np.asarray(is_active_row, dtype=bool)
        & MCE_c_rel.gt(τ_MCE_REL).to_numpy(dtype=bool)
        & MCE_c.abs().gt(τ_MCE_ABS).to_numpy(dtype=bool)
    )

    thr_arr      = np.array([node_thresholds.get(s, GLOBAL_HYDRAULIC_DEADBAND) for s in SENSOR_COLS])
    excess_mat   = R_c_excess[SENSOR_COLS].abs().to_numpy(dtype=float)
    pred_ok_mat  = pred_aligned[SENSOR_COLS].notna().to_numpy(dtype=bool)
    od_stab_mat  = is_od_stable[SENSOR_COLS].to_numpy(dtype=bool)
    no_trans_mat = (~is_valve_transition[SENSOR_COLS]).to_numpy(dtype=bool)

    branch_normal_mat = (excess_mat < β_NORMAL * thr_arr) & diagnosable_mat
    branch_spike_mat  = (
        (excess_mat > thr_arr) & od_stab_mat & no_trans_mat & pred_ok_mat & diagnosable_mat
    )

    n_diagnosable = diagnosable_mat.sum(axis=1)
    n_normal      = branch_normal_mat.sum(axis=1)
    n_spikes      = branch_spike_mat.sum(axis=1)

    mce_violated_rows = np.where(mce_violated)[0]
    root_cause_counts = {s: 0 for s in SENSOR_COLS}
    if len(mce_violated_rows) > 0:
        excess_mce     = excess_mat[mce_violated_rows]
        argmax_sensors = np.argmax(excess_mce, axis=1)
        for ci, sensor in enumerate(SENSOR_COLS):
            root_cause_counts[sensor] = int((argmax_sensors == ci).sum())

    # ──────────── Scenario B ────────────
    all_diag_normal = (
        mce_violated
        & (n_normal == n_diagnosable)
        & (n_diagnosable >= MIN_DIAG_FOR_B)
    )

    Q_sum_branches = Q_input_c_out[SENSOR_COLS].sum(axis=1)
    Q_total_new    = Q_sum_branches / max(1.0 - leakage_rate, 1e-9)
    rel_diff_b     = (Q_total_new - df_raw[TOTAL_COL]).abs() / df_raw[TOTAL_COL].clip(lower=1.0)

    sub_b_apply   = all_diag_normal & rel_diff_b.le(SCENARIO_B_FUSE).values
    sub_b_anomaly = all_diag_normal & rel_diff_b.gt(SCENARIO_B_FUSE).values

    if sub_b_apply.any():
        Q_input_c_out.loc[sub_b_apply, TOTAL_COL] = Q_total_new[sub_b_apply]
        flag_df.loc[sub_b_apply, TOTAL_COL] = FLAG_TOTAL_ADJUSTED

    if sub_b_anomaly.any():
        flag_df.loc[sub_b_anomaly, TOTAL_COL] = FLAG_SYSTEM_ANOMALY

    # ──────────── Scenario A ────────────
    remaining    = mce_violated & ~all_diag_normal
    n_non_spike  = (diagnosable_mat & ~branch_spike_mat).sum(axis=1)
    scenario_a   = remaining & (n_spikes > 0) & (n_non_spike > 0)

    if scenario_a.any():
        for ci, sensor in enumerate(SENSOR_COLS):
            spike_rows = scenario_a & branch_spike_mat[:, ci]
            if spike_rows.any():
                Q_input_c_out.loc[spike_rows, sensor] = pred_aligned.loc[spike_rows, sensor]
                flag_df.loc[spike_rows, sensor] = FLAG_SPIKE_FIXED

    fallback = remaining & ~scenario_a

    if fallback.any():
        pred_np      = pred_aligned[SENSOR_COLS].to_numpy(dtype=float)
        net_total_np = np.asarray(df_raw[TOTAL_COL], dtype=float) * (1.0 - leakage_rate)
        zero_mat     = np.column_stack([
            (flag_df[s].values == FLAG_ZERO) | (flag_df[s].values == FLAG_VALVE_TRANSITION)
            for s in SENSOR_COLS
        ])

        for row_pos in np.where(fallback)[0]:
            pred_row = pred_np[row_pos].copy()
            pred_row[zero_mat[row_pos]] = 0.0
            pred_row[~np.isfinite(pred_row)] = 0.0
            pred_sum = float(pred_row.sum())
            if pred_sum > 0:
                weights = np.divide(pred_row, pred_sum)
                recon   = weights * float(net_total_np[row_pos])
                for ci, sensor in enumerate(SENSOR_COLS):
                    if not zero_mat[row_pos, ci]:
                        row_label = Q_input_c_out.index[row_pos]
                        Q_input_c_out.loc[row_label, sensor] = float(recon[ci])
                        flag_df.loc[row_label, sensor] = FLAG_FULL_RECONSTRUCT

    for sensor in SENSOR_COLS:
        if sensor not in scale_df.columns:
            continue
        drift_rows = (
            valid_mask_row
            & is_active_row
            & (pd.Series(flag_df[sensor].values == FLAG_NORMAL, index=df_raw.index))
            & scale_df[sensor].ne(1.0)
        )
        if drift_rows.any():
            flag_df.loc[drift_rows, sensor] = FLAG_DRIFT

    assimilation_info: Dict = {
        "leakage_rate":         leakage_rate,
        "n_mce_violated":       int(mce_violated.sum()),
        "n_scenario_b":         int(sub_b_apply.sum()),
        "n_scenario_b_anomaly": int(sub_b_anomaly.sum()),
        "n_scenario_a":         int(scenario_a.sum()),
        "n_fallback":           int(fallback.sum()),
        "root_cause_counts":    root_cause_counts,
        "mce_violated_count":   int(mce_violated.sum()),
    }

    return Q_input_c_out, flag_df, assimilation_info


# ===========================================================================
# ===========================================================================

def build_sensor_profile(
    df_raw: pd.DataFrame,
    R_c: pd.DataFrame,
    μ_R: Dict[str, float],
    node_thresholds: Dict[str, float],
    flag_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    drift_events: Dict[str, List[Tuple[int, int, float]]],
    is_valve_transition: pd.DataFrame,
    is_active_row: pd.Series,
    valid_mask_row: pd.Series,
    assimilation_info: Dict,
) -> Dict:
    """Summarise each meter's correction flags and residual diagnostics."""
    profile: Dict[str, Any] = {}
    mce_violated_count = assimilation_info.get("mce_violated_count", 1)
    root_cause_counts  = assimilation_info.get("root_cause_counts", {})

    active_valid = is_active_row & valid_mask_row
    active_count = int(active_valid.sum())

    for sensor in SENSOR_COLS:
        f_active = flag_df.loc[active_valid, sensor]
        n_active = max(active_count, 1)

        spike_c = int((f_active == FLAG_SPIKE_FIXED).sum())
        drift_c = int((f_active == FLAG_DYNAMIC_DRIFT).sum())
        zero_c  = int((f_active == FLAG_ZERO).sum())
        trans_c = int((f_active == FLAG_VALVE_TRANSITION).sum())
        recon_c = int((f_active == FLAG_FULL_RECONSTRUCT).sum())

        both = active_valid & ~is_valve_transition[sensor] & R_c[sensor].notna()
        od_residual_corr = 0.0
        od_col = f"OD_{sensor}"
        if od_col in df_raw.columns and both.sum() > 10:
            try:
                od_v = np.asarray(df_raw.loc[both, od_col], dtype=float)
                r_v  = np.asarray(R_c.loc[both, sensor].abs(), dtype=float)
                corr_mat  = np.corrcoef(od_v, r_v)
                corr_val  = float(corr_mat[0, 1])
                od_residual_corr = float(abs(corr_val)) if np.isfinite(corr_val) else 0.0
            except Exception:
                od_residual_corr = 0.0

        evts = drift_events.get(sensor, [])
        max_drift_dur  = max((e[1] - e[0] + 1 for e in evts), default=0)
        max_drift_amp  = max((e[2] for e in evts), default=0.0)

        root_cause_prob = 100.0 * root_cause_counts.get(sensor, 0) / max(mce_violated_count, 1)

        profile[sensor] = {
            "spike_count":                    spike_c,
            "spike_rate_per_1k_windows":      round(1000.0 * spike_c / n_active, 3),
            "dynamic_drift_event_count":      len(evts),
            "dynamic_drift_total_steps":      drift_c,
            "dynamic_drift_max_consecutive":  max_drift_dur,
            "dynamic_drift_max_amplitude_m3ph": round(max_drift_amp, 3),
            "od_residual_pearson_corr":        round(od_residual_corr, 4),
            "od_corr_significant":             bool(od_residual_corr >= OD_CORR_SIGNIFICANT),
            "root_cause_probability_pct":      round(root_cause_prob, 2),
            "zero_count":                      zero_c,
            "valve_transition_count":          trans_c,
            "full_reconstruct_count":          recon_c,
            "static_scale_mean":               round(
                float(scale_df[sensor][active_valid].mean())
                if sensor in scale_df.columns else 1.0, 6
            ),
            "threshold_m3ph":                  round(node_thresholds.get(sensor, 0.0), 3),
            "mu_residual_m3ph":                round(μ_R.get(sensor, 0.0), 3),
        }

    n_total_adj   = int((flag_df[TOTAL_COL] == FLAG_TOTAL_ADJUSTED).sum())
    n_sys_anomaly = int((flag_df[TOTAL_COL] == FLAG_SYSTEM_ANOMALY).sum())
    profile[TOTAL_COL] = {
        "total_adjusted_count":       n_total_adj,
        "system_anomaly_count":       n_sys_anomaly,
        "total_adjusted_rate_per_1k": round(1000.0 * n_total_adj / max(active_count, 1), 3),
        "mce_violated_count":         mce_violated_count,
    }

    return profile


# ===========================================================================
# ===========================================================================

def global_conservation_check(
    df_raw: pd.DataFrame,
    Q_input: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    leakage_rate: float = 0.0,
) -> Dict[str, float]:
    branch_cols = [c for c in SENSOR_COLS if c in df_cleaned.columns]
    result: Dict[str, float] = {}

    for tag, df in [
        ("raw_original", df_raw),
        ("raw_debiased",  Q_input),
        ("cleaned",       df_cleaned),
    ]:
        if TOTAL_COL not in df.columns or not branch_cols:
            result[f"{tag}_residual_mean"] = float("nan")
            continue
        total      = df[TOTAL_COL]
        branch_sum = df[branch_cols].sum(axis=1)
        valid_mask = total.notna() & branch_sum.notna() & total.gt(0)
        residual   = (total * (1.0 - leakage_rate) - branch_sum)[valid_mask]
        if len(residual) == 0:
            result[f"{tag}_residual_mean"] = float("nan")
            continue
        result[f"{tag}_residual_mean"] = float(residual.mean())
        result[f"{tag}_residual_std"]  = float(residual.std())
        result[f"{tag}_residual_n"]    = int(valid_mask.sum())

    return result


# ===========================================================================
# ===========================================================================

def _load_eom_and_set(model: PI_GAT, eom_json: Path, data_graph: Any, device: str) -> None:
    if not eom_json.exists():
        print(f"  [WARNING] EOM JSON not found, skipping: {eom_json}")
        return
    try:
        sys.path.insert(0, str(PI_GAT_DIR / "opening_mapping"))
        from eom import load_mapping_json  # type: ignore
        mapping = load_mapping_json(eom_json)
    except Exception as e:
        print(f"  [WARNING] Failed to load EOM JSON: {e}")
        return

    assert data_graph.edge_index is not None
    E_base  = int(data_graph.edge_index.size(1))
    K       = 6
    y_knots = torch.ones(E_base, K, dtype=torch.float32)

    sensor2edge = getattr(data_graph, "sensor2edge_idx", {}) or {}
    for sensor, ei in sensor2edge.items():
        if sensor not in mapping or not isinstance(mapping[sensor], dict):
            continue
        y_pct = mapping[sensor].get("y_knots_pct", None)
        if y_pct is not None and len(y_pct) == K:
            y_knots[int(ei)] = torch.tensor(
                [v / 100.0 for v in y_pct], dtype=torch.float32
            )

    _unwrap_pi_gat(model).set_effective_opening_mapping(y_knots.to(device))
    print(f"  [OK] EOM injected from {eom_json.name} ({E_base} edges, K={K})")


# ===========================================================================
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI-GAT Stage III: State-Estimation Reverse Fault Diagnosis"
    )
    parser.add_argument("--device",    type=str, default=None)
    parser.add_argument("--window",    type=int, default=WIN)
    parser.add_argument("--batch",     type=int, default=BATCH)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PI_GAT_DIR / "stage1_results_high_eom" / "pi_gat_best.pth"),
    )
    parser.add_argument(
        "--eom-json",
        type=str,
        default=str(PI_GAT_DIR / "opening_mapping" / "eom_high.json"),
    )
    args = parser.parse_args()

    DEVICE     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    WINDOW     = args.window
    BATCH_SIZE = args.batch

    print("=" * 60)
    print("PI-GAT Stage III: State-Estimation Reverse Fault Diagnosis")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Window : {WINDOW}")
    print(f"  Batch  : {BATCH_SIZE}")

    TOPO_CSV        = ROOT_DIR / "network_topology.csv"
    ZSCORE_CSV      = PREPROCESS_DIR / "flow_5min_zscore.csv"
    RESTMASK_CSV    = PREPROCESS_DIR / "mask_rest.csv"
    ZSCORE_STATS    = PREPROCESS_DIR / "zscore_stats.json"
    OPEN_CSV        = PREPROCESS_DIR / "flow_5min_imputed.csv"
    CKPT_PATH       = Path(args.ckpt)
    EOM_JSON        = Path(args.eom_json) if args.eom_json else None
    CREDIBILITY_CSV = PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"

    OUT_CLEANED = SCRIPT_DIR / "flow_5min_cleaned_without_bias.csv"
    OUT_FLAGS   = SCRIPT_DIR / "cleaning_flags_without_bias.csv"
    OUT_PROFILE = SCRIPT_DIR / "sensor_profile_without_bias.json"
    OUT_SUMMARY = SCRIPT_DIR / "cleaning_summary_without_bias.json"

    print("\n[Step 1/7] Loading external resources")

    if not ZSCORE_STATS.exists():
        raise FileNotFoundError(f"Z-score stats not found: {ZSCORE_STATS}")
    zscore_means, _ = _load_zscore_stats(ZSCORE_STATS)
    print(f"  [OK] Z-score stats: {len(zscore_means)} columns")

    if not OPEN_CSV.exists():
        raise FileNotFoundError(f"Flow CSV not found: {OPEN_CSV}")
    df_raw = pd.read_csv(OPEN_CSV, index_col=0, parse_dates=True)
    print(f"  [OK] Raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    print("\n[Step 2/7] Building graph and dataset")
    data_graph, node2idx, pipe2idx, sensor2edge_idx = build_graph_from_csv(
        TOPO_CSV, weight_mode="conductance", add_self_loops=True, verbose=False
    )
    sensor2node_idx: Dict[str, int] = data_graph.sensor2node_idx
    # Pylance: data_graph.num_nodes / data_graph.edge_index may be Optional — guard them
    if data_graph.num_nodes is None:
        raise RuntimeError("Graph object has no 'num_nodes' defined")
    N_nodes = int(data_graph.num_nodes)
    if getattr(data_graph, "edge_index", None) is None:
        raise RuntimeError("Graph object has no 'edge_index' defined")
    from typing import cast
    edge_index = cast("torch.Tensor", data_graph.edge_index)
    n_edges = int(edge_index.size(1))
    print(f"  [OK] Graph: {N_nodes} nodes, {n_edges} edges")

    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=ZSCORE_CSV,
        restmask_path=RESTMASK_CSV,
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WINDOW,
        random_mask_range=None,
        open_paths=OPEN_CSV,
        zscore_stats_path=ZSCORE_STATS,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None),
        verbose=False,
    )
    valid_t_list: List[int] = dataset.valid_t
    print(f"  [OK] Dataset: {len(valid_t_list)} valid timesteps (window={WINDOW})")

    if valid_t_list and max(valid_t_list) >= len(df_raw):
        print(f"  [WARNING] max valid_t={max(valid_t_list)} >= n_rows={len(df_raw)}, truncating.")
        valid_t_list = [t for t in valid_t_list if t < len(df_raw)]

    print("\n[Step 3/7] Loading model")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model = PI_GAT(
        win=WINDOW,
        num_nodes=N_nodes,
        hidden=HIDDEN,
        heads=HEADS,
        dropout=0.0,
        in_feat=INFEAT,
    ).to(DEVICE)

    ckpt  = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_state: Dict = {}
    for k, v in state.items():
        nk = k
        for prefix in ("_orig_mod.", "module."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    print(f"  [OK] Model loaded: {CKPT_PATH.name}")

    leakage_rate = float(_unwrap_pi_gat(model).get_leakage_rate().detach().cpu().item())
    print(f"  [OK] Leakage rate (from model): {leakage_rate:.6f}")

    if EOM_JSON is not None:
        _load_eom_and_set(model, EOM_JSON, data_graph, DEVICE)
    else:
        print("  [INFO] No EOM JSON provided, using model's built-in EOM.")

    print("\n[Step 4/7] Full inference (Debiased Physical Space: node_hat)")
    pred_df = run_full_inference(
        model=model,
        dataset=dataset,
        sensor2node_idx=sensor2node_idx,
        n_nodes=N_nodes,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )
    print(f"  [OK] Inference done: {len(pred_df)} timesteps")

    # ── Step 5: Scale Matrix / Valve Masks / Residual Baseline / Drift Scan ──
    print("\n[Step 5/7] Pre-processing: scale → valve masks → residual baseline → drift scan")

    scale_df = pd.DataFrame(1.0, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    print("  [OK] Scale matrix built (ablation: all-ones, no bias correction)")

    is_valve_transition, is_od_stable = compute_valve_state_masks(df_raw)
    print(f"  [OK] Valve transition rows  : {int(is_valve_transition.any(axis=1).sum())}")
    print(f"  [OK] All-OD-stable rows     : {int(is_od_stable.all(axis=1).sum())}")

    n_total = len(df_raw)

    Q_input_tmp = df_raw.copy()
    for sensor in SENSOR_COLS:
        if sensor in df_raw.columns and sensor in scale_df.columns:
            Q_input_tmp[sensor] = df_raw[sensor] / scale_df[sensor].replace(0, 1.0)

    pred_aligned_tmp = pd.DataFrame(np.nan, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    pred_t_arr = np.array(pred_df.index.tolist(), dtype=int)
    in_range   = (pred_t_arr >= 0) & (pred_t_arr < n_total)
    if in_range.any():
        locs = pred_t_arr[in_range]
        psub = pred_df.iloc[np.where(in_range)[0]]
        for sensor in SENSOR_COLS:
            if sensor in pred_df.columns:
                pred_aligned_tmp.loc[pred_aligned_tmp.index[locs], sensor] = (
                    psub[sensor].to_numpy(dtype=float)
                )

    is_active_tmp = pd.Series(False, index=df_raw.index)
    if CREDIBILITY_CSV.exists():
        _cred_tmp   = pd.read_csv(CREDIBILITY_CSV, usecols=["t_index"])
        _active_arr = _cred_tmp["t_index"].astype(int).to_numpy()
        _in_range   = (_active_arr >= 0) & (_active_arr < n_total)
        is_active_tmp.iloc[_active_arr[_in_range]] = True

    μ_R, node_thresholds, R_base = compute_residual_baseline(
        Q_input=Q_input_tmp,
        pred_aligned=pred_aligned_tmp,
        is_valve_transition=is_valve_transition,
        is_od_stable=is_od_stable,
        is_active_row=is_active_tmp,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None),
        n_total=n_total,
        valid_t_list=valid_t_list,
    )

    print("  Node-specific thresholds (m³/h):")
    for sensor in SENSOR_COLS:
        print(f"    {sensor:5s}: μ_R = {μ_R[sensor]:+7.2f},  Threshold = {node_thresholds[sensor]:.2f}")

    valid_mask_tmp = pd.Series(False, index=df_raw.index)
    valid_mask_tmp.iloc[np.array(valid_t_list, dtype=int)] = True

    print("  Scanning for Scenario C dynamic drift...")
    drift_correction, drift_events = detect_dynamic_drift(
        R=R_base,
        μ_R=μ_R,
        node_thresholds=node_thresholds,
        is_valve_transition=is_valve_transition,
        is_active_row=is_active_tmp,
        valid_mask_row=valid_mask_tmp,
    )
    for sensor in SENSOR_COLS:
        evts = drift_events.get(sensor, [])
        if evts:
            print(f"    {sensor}: {len(evts)} event(s), "
                  f"max {max(e[1]-e[0]+1 for e in evts)} steps, "
                  f"max amp {max(e[2] for e in evts):.1f} m³/h")
        else:
            print(f"    {sensor}: no drift events")

    print("\n[Step 6/7] Physics assimilation (Scenario A / B / C + Fallback)")
    df_cleaned, flag_df, assim_info = apply_physics_assimilation(
        df_raw=df_raw,
        pred_df=pred_df,
        scale_df=scale_df,
        calib={"leakage": {"rate": leakage_rate}},
        node_thresholds=node_thresholds,
        μ_R=μ_R,
        drift_correction=drift_correction,
        drift_events=drift_events,
        is_valve_transition=is_valve_transition,
        is_od_stable=is_od_stable,
        valid_t_list=valid_t_list,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None),
    )

    print(f"  MCE violated rows    : {assim_info['n_mce_violated']}")
    print(f"  Scenario B applied   : {assim_info['n_scenario_b']}")
    print(f"  System anomaly: {assim_info['n_scenario_b_anomaly']}")
    print(f"  Scenario A applied   : {assim_info['n_scenario_a']}")
    print(f"  Fallback reconstruct : {assim_info['n_fallback']}")

    is_active_final  = pd.Series(False, index=df_raw.index)
    valid_mask_final = pd.Series(False, index=df_raw.index)
    valid_mask_final.iloc[np.array(valid_t_list, dtype=int)] = True
    if CREDIBILITY_CSV.exists():
        _ca  = pd.read_csv(CREDIBILITY_CSV, usecols=["t_index"])
        _arr = _ca["t_index"].astype(int).to_numpy()
        _ir  = (_arr >= 0) & (_arr < n_total)
        is_active_final.iloc[_arr[_ir]] = True

    active_valid_final = is_active_final & valid_mask_final
    FLAG_NAMES = {
        FLAG_IGNORE: "IGNORE", FLAG_SPINUP: "SPINUP",
        FLAG_NORMAL: "NORMAL", FLAG_DRIFT: "DRIFT",
        FLAG_SPIKE_FIXED: "SPIKE_FIXED", FLAG_TOTAL_ADJUSTED: "TOTAL_ADJUSTED",
        FLAG_DYNAMIC_DRIFT: "DYNAMIC_DRIFT", FLAG_ZERO: "ZERO",
        FLAG_VALVE_TRANSITION: "VALVE_TRANSITION",
        FLAG_SYSTEM_ANOMALY: "SYSTEM_ANOMALY",
        FLAG_FULL_RECONSTRUCT: "FULL_RECONSTRUCT",
    }
    print("\n  Cleaning flag summary (active rows only):")
    for sensor in ALL_FLOW_COLS:
        if sensor not in flag_df.columns:
            continue
        f_active = flag_df.loc[active_valid_final, sensor]
        counts_str = "  ".join(
            f"{fn}={int((f_active == fv).sum())}"
            for fv, fn in FLAG_NAMES.items()
            if int((f_active == fv).sum()) > 0
        )
        print(f"    {sensor:6s}: {counts_str}")

    print("\n[Step 7/7] Conservation check & output")

    Q_input_final = df_raw.copy()
    for sensor in SENSOR_COLS:
        if sensor in df_raw.columns and sensor in scale_df.columns:
            Q_input_final[sensor] = df_raw[sensor] / scale_df[sensor].replace(0, 1.0)

    conservation = global_conservation_check(
        df_raw=df_raw,
        Q_input=Q_input_final,
        df_cleaned=df_cleaned,
        leakage_rate=leakage_rate,
    )
    for k, v in conservation.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")

    Q_input_c_for_profile = Q_input_final.copy()
    pred_aligned_final = pd.DataFrame(np.nan, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    if in_range.any():
        psub_f = pred_df.iloc[np.where(in_range)[0]]
        for sensor in SENSOR_COLS:
            if sensor in pred_df.columns:
                pred_aligned_final.loc[pred_aligned_final.index[pred_t_arr[in_range]], sensor] = (
                    psub_f[sensor].to_numpy(dtype=float)
                )

    for sensor in SENSOR_COLS:
        dc_mask = (
            drift_correction[sensor].ne(0)
            & is_active_final
            & ~is_valve_transition[sensor]
            & valid_mask_final
            & pred_aligned_final[sensor].notna()
        )
        if dc_mask.any():
            Q_input_c_for_profile.loc[dc_mask, sensor] -= drift_correction.loc[dc_mask, sensor]

    R_c_final = pd.DataFrame(np.nan, index=df_raw.index, columns=SENSOR_COLS, dtype=float)
    for sensor in SENSOR_COLS:
        R_c_final[sensor] = Q_input_c_for_profile[sensor] - pred_aligned_final[sensor]

    sensor_profile = build_sensor_profile(
        df_raw=df_raw,
        R_c=R_c_final,
        μ_R=μ_R,
        node_thresholds=node_thresholds,
        flag_df=flag_df,
        scale_df=scale_df,
        drift_events=drift_events,
        is_valve_transition=is_valve_transition,
        is_active_row=is_active_final,
        valid_mask_row=valid_mask_final,
        assimilation_info=assim_info,
    )

    df_cleaned.to_csv(OUT_CLEANED, encoding="utf-8")
    print(f"  [OK] Cleaned data  : {OUT_CLEANED}")

    flag_df.to_csv(OUT_FLAGS, encoding="utf-8")
    print(f"  [OK] Flags         : {OUT_FLAGS}")

    with open(OUT_PROFILE, "w", encoding="utf-8") as f:
        json.dump(sensor_profile, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Sensor profile: {OUT_PROFILE}")

    cleaning_counts: Dict[str, Dict[str, int]] = {}
    for sensor in ALL_FLOW_COLS:
        if sensor not in flag_df.columns:
            continue
        cleaning_counts[sensor] = {
            fn: int(np.sum(np.asarray(flag_df[sensor], dtype=np.int16) == int(fv)))
            for fv, fn in FLAG_NAMES.items()
        }

    summary = {
        "stage":     "III — State-Estimation Reverse Fault Diagnosis",
        "eom_used":  str(EOM_JSON) if EOM_JSON else "model_default",
        "leakage_rate": leakage_rate,
        "total_rows":   len(df_raw),
        "spinup_rows":  int(WINDOW - 1),
        "valid_rows":   len(valid_t_list),
        "active_rows":  int(is_active_final.sum()),
        "algorithm_params": {
            "k_spike":                K_SPIKE,
            "beta_normal":            β_NORMAL,
            "tau_mce_rel":            τ_MCE_REL,
            "tau_mce_abs":            τ_MCE_ABS,
            "scenario_b_fuse":        SCENARIO_B_FUSE,
            "w_min_drift":            W_MIN_DRIFT,
            "drift_criterion_mult":   DRIFT_MULT,
            "delta_od_transition":    ΔOD_TRANSITION,
            "delta_od_stable_thresh": ΔOD_STABLE_THRESH,
            "valve_closed_thresh":    VALVE_CLOSED_THRESH,
        },
        "node_thresholds_m3ph": {k: round(v, 4) for k, v in node_thresholds.items()},
        "mu_residuals_m3ph":    {k: round(v, 4) for k, v in μ_R.items()},
        "assimilation_stats":   assim_info,
        "cleaning_counts":      cleaning_counts,
        "conservation":         conservation,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Summary       : {OUT_SUMMARY}")

    print("\n" + "=" * 60)
    print("Stage III complete!")
    print("=" * 60)
    print(f"  Cleaned data  : {OUT_CLEANED.name}")
    print(f"  Flags         : {OUT_FLAGS.name}")
    print(f"  Sensor profile: {OUT_PROFILE.name}")
    print(f"  Summary       : {OUT_SUMMARY.name}")


if __name__ == "__main__":
    main()
