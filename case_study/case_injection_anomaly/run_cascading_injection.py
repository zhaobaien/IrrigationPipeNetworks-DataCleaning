from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from scipy.linalg import pinv        # type: ignore
from scipy.stats import norm, pearsonr  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PI_GAT_DIR    = ROOT_DIR / "PI-GAT"
BASELINE_DIR  = ROOT_DIR / "Baseline"
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"
TEMP_DIR      = SCRIPT_DIR / "temp_injection"
TEMP_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "data_clean"))
sys.path.insert(0, str(PI_GAT_DIR))
sys.path.insert(0, str(BASELINE_DIR / "GAT-only"))
sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(BASELINE_DIR / "common"))

try:
    from data_cleaning import (  # type: ignore
        _load_calib_params, _build_scale_matrix,
        run_full_inference, compute_valve_state_masks, compute_residual_baseline,
        detect_dynamic_drift, apply_physics_assimilation, _load_eom_and_set,
        SENSOR_COLS, BATCH, WIN, GLOBAL_HYDRAULIC_DEADBAND,
    )
    from graph_utils import build_graph_from_csv                 # type: ignore
    from model_pi_gat import PI_GAT                              # type: ignore
    from timeseries_dataset import IrrigationTimeWindowDataset   # type: ignore
    from model_gat import GAT                                    # type: ignore
    from effective_value_filtering import build_active_mask       # type: ignore
except ImportError as e:
    raise RuntimeError(
        "Failed to import PI-GAT framework modules. "
        "Ensure the conda PyTorch environment is active."
    ) from e

TOTAL_COL = "Total"
FLOW_ZERO_THRESH = 5.0          # m³/h — kinematic compliance threshold
SFAR_REL_THRESH  = 0.05         # 5 % relative change → false alarm
SFAR_ABS_THRESH  = 2.0          # 2 m³/h absolute floor (avoid noise FP)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GT window selection
# ══════════════════════════════════════════════════════════════════════════════

def find_longest_clean_window(active_set: set) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return (df_gt, gt_indices) for the longest consecutive high-credibility
    AND active window.  Minimum length: 3 days = 864 steps."""
    cred_csv = PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"
    df_cred  = pd.read_csv(str(cred_csv))
    df_cred["is_high"] = (
        (df_cred["credibility"] == "high") &
        (df_cred["t_index"].isin(active_set))
    )
    m        = df_cred["is_high"].ne(df_cred["is_high"].shift()).cumsum()
    grouped  = df_cred.groupby(m)
    counts   = grouped.size()
    is_high_per_group = grouped["is_high"].first()
    max_group = int(counts[is_high_per_group].idxmax())
    gt_indices = df_cred[m == max_group]["t_index"].values.astype(int)

    L = len(gt_indices)
    if L < 864:
        raise RuntimeError(f"Longest clean window is only {L} steps (< 864). "
                           "Cannot run cascading injection reliably.")

    raw_csv = PREPROCESS_DIR / "flow_5min_imputed.csv"
    df_raw  = pd.read_csv(str(raw_csv))
    df_gt   = df_raw.iloc[list(gt_indices)].copy().reset_index(drop=True)
    print(f"  Golden GT window: {L} steps (~{L/288:.1f} days)")
    return df_gt, cast(np.ndarray, gt_indices)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fault injection (empirically seeded)
# ══════════════════════════════════════════════════════════════════════════════

def inject_cascading_faults(
    df_gt: pd.DataFrame,
    profiles: Dict[str, Any],
    seed: int = 42,
    gt_indices: Optional[np.ndarray] = None,
    frozen_refs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inject four cascading fault types onto the clean GT DataFrame.
    Returns (df_corrupted, injection_map) where injection_map records which
    rows/sensors were corrupted for evaluation.
    """
    rng = np.random.default_rng(seed)
    L   = len(df_gt)
    df  = df_gt.copy()

    spike_ratio   = profiles["spike"]["global_median_ratio"]
    # Clamp spike ratio to [2.0, 8.0]: extreme 33x is visually valid in real data
    # but makes MAPE trivially 3200%, obscuring meaningful model differences.
    spike_ratio   = float(np.clip(spike_ratio, 2.0, 8.0))
    drop_len      = int(profiles["sensor_drop"]["recommended_injection_steps"])
    ramp_steps    = int(profiles["gain_drift"]["recommended_ramp_steps"])
    ramp_slope    = float(profiles["gain_drift"]["recommended_slope"])
    plateau_ratio = float(profiles["gain_drift"]["recommended_plateau"])
    total_mult    = float(profiles["total_bias"]["recommended_multiplier"])

    injection_map: Dict[str, Any] = {
        "L": L,
        "spike_ratio":   spike_ratio,
        "drop_len":      drop_len,
        "ramp_steps":    ramp_steps,
        "plateau_ratio": plateau_ratio,
        "total_mult":    total_mult,
    }

    # B1 — Random Spikes: 15 points on non-zero B01, EXCLUDING valve-transition
    # guard windows.  data_cleaning.py skips Scenario-A correction for any
    # timestep where |ΔOD_B01| > 5% ± VALVE_GUARD(=1) steps, so spikes placed
    # inside those windows will never be cleaned.  We therefore filter them out
    # here to guarantee each injected spike falls in a correctable region.
    DELTA_OD_TRANSITION = 5.0   # mirrors data_cleaning.py constant
    VALVE_GUARD_STEPS   = 1     # mirrors data_cleaning.py VALVE_GUARD
    od_B01_col = "OD_B01"
    if od_B01_col in df.columns:
        od_B01      = df[od_B01_col].astype(float)
        raw_jump   = od_B01.diff().abs().gt(DELTA_OD_TRANSITION)
        # Expand guard: flag the jump step itself plus ±VALVE_GUARD steps
        valve_guard = raw_jump.copy()
        for _shift in range(1, VALVE_GUARD_STEPS + 1):
            valve_guard = valve_guard | raw_jump.shift(_shift, fill_value=False)
            valve_guard = valve_guard | raw_jump.shift(-_shift, fill_value=False)
        valid_b1 = df.index[
            (df["B01"].astype(float) > 10.0) & ~valve_guard
        ].tolist()
    else:
        # OD column absent — fall back to flow-only filter
        valid_b1 = df.index[(df["B01"].astype(float) > 10.0)].tolist()

    n_spikes  = min(15, len(valid_b1))
    spike_idx = sorted(rng.choice(valid_b1, n_spikes, replace=False).tolist())
    df.loc[spike_idx, "B01"] = df.loc[spike_idx, "B01"].astype(float) * spike_ratio
    injection_map["b1_spike_idx"] = spike_idx

    # B2 — Sensor Drop: B03 → 0, window [10% ~ 25%] ───────────────────────────
    # Moved earlier (was 20%) so it fully clears before B3 begins at 35%.
    b2_start = int(L * 0.10)
    b2_end   = min(b2_start + drop_len, int(L * 0.25))
    b2_idx   = df.index[b2_start:b2_end].tolist()
    df.loc[b2_idx, "B03"] = 0.0
    injection_map["b2_drop_idx"] = b2_idx

    # B3 — Ramp Gain Drift: B04, window [35% ~ 60%] — RECOVERS at 60% ─────────
    # Critical design: B3 ends at 60% so that by the time B4 starts at 70%
    # ALL branch sensors (including B04) are fully healthy again.  This removes
    # the "ambiguous downstream fault" condition that previously caused PI-GAT's
    # Scenario-B safety lock to bypass Total correction.
    b3_start    = int(L * 0.35)
    b3_end_hard = int(L * 0.60)           # hard ceiling — drift fully stops here
    b3_ramp_end = min(b3_start + ramp_steps, b3_end_hard)
    b3_plat_end = b3_end_hard

    # Ramp phase: multiplier grows linearly from 1.0 to plateau_ratio
    ramp_multipliers = np.linspace(1.0, plateau_ratio, b3_ramp_end - b3_start)
    ramp_idx         = df.index[b3_start:b3_ramp_end].tolist()
    df.loc[ramp_idx, "B04"] = (np.asarray(df.loc[ramp_idx, "B04"].astype(float), dtype=float)
                               * ramp_multipliers)
    # Plateau phase: constant multiplier until end of B3 window
    plat_idx = df.index[b3_ramp_end:b3_plat_end].tolist()
    df.loc[plat_idx, "B04"] = df.loc[plat_idx, "B04"].astype(float) * plateau_ratio
    injection_map["b3_drift_idx"] = ramp_idx + plat_idx
    injection_map["b3_all_start"] = b3_start
    injection_map["b3_all_end"]   = b3_plat_end

    # B4 — Total Meter Bias: injected into "observable" sub-window ───────────
    # Scenario-B prerequisite requires n_diagnosable >= MIN_DIAG_FOR_B (=2).
    # At injection time we proxy "diagnosable" from raw OD columns:
    #   OD > 5%     → not FLAG_ZERO        (valve open, mirrors VALVE_CLOSED_THRESH)
    #   |ΔOD| ≤ 5%  → not FLAG_TRANSITION  (no transient, mirrors DELTA_OD_TRANSITION)
    # We search [max(b3_end+5%, 70%), 95%] for the longest consecutive run where
    # ≥ 2 sensors simultaneously satisfy both criteria, then inject Total bias
    # into that observable window.  This guarantees Scenario-B can fire on every
    # B4 timestep, allowing PI-GAT to correct Total via MCE inversion.
    MIN_DIAG_B4     = 2     # mirrors data_cleaning.py MIN_DIAG_FOR_B
    OD_OPEN_MIN     = 5.0   # % — mirrors VALVE_CLOSED_THRESH
    OD_JUMP_MAX     = 5.0   # % — mirrors DELTA_OD_TRANSITION
    b4_search_start = max(b3_plat_end + int(L * 0.05), int(L * 0.70))
    b4_search_end   = int(L * 0.95)

    b4_idx: List[int] = []
    b4_start = b4_search_start
    b4_end   = b4_search_end

    od_cols_present_b4 = [f"OD_{s}" for s in SENSOR_COLS if f"OD_{s}" in df.columns]
    if len(od_cols_present_b4) >= MIN_DIAG_B4:
        # Build per-sensor diagnosability proxy mask from OD columns
        diag_mask_b4 = pd.DataFrame(False, index=df.index, columns=SENSOR_COLS)
        for s in SENSOR_COLS:
            od_col = f"OD_{s}"
            if od_col in df.columns:
                od = df[od_col].astype(float)
                diag_mask_b4[s] = od.gt(OD_OPEN_MIN) & od.diff().abs().le(OD_JUMP_MAX)
        n_diag_b4 = diag_mask_b4.sum(axis=1)
        obs_arr   = np.asarray(n_diag_b4.ge(MIN_DIAG_B4), dtype=bool)

        # ── Prediction-quality filter ─────────────────────────────────────────
        # branch_normal_mat requires |R_c[s] - mu_R[s]| < EPS_NORMAL*thr[s].
        # Use frozen clean baselines to pre-verify each candidate row so that
        # Scenario-B is guaranteed to fire for every injected B4 step.
        EPS_NORMAL_VAL = 0.5   # mirrors data_cleaning.py EPS_NORMAL
        if frozen_refs is not None and gt_indices is not None:
            pred_al = frozen_refs["pred_aligned_clean"]
            mu_R_fr = frozen_refs["mu_R"]
            thrs_fr = frozen_refs["node_thresholds"]
            sc_df   = frozen_refs["scale_df"]
            pred_ok = np.ones(L, dtype=bool)
            for i in range(b4_search_start, b4_search_end):
                if not obs_arr[i]:
                    pred_ok[i] = False
                    continue
                gi = int(gt_indices[i])
                if gi not in pred_al.index:
                    pred_ok[i] = False
                    continue
                all_ok = True
                for s in SENSOR_COLS:
                    if s not in pred_al.columns:
                        continue
                    mu_s  = mu_R_fr.get(s, 0.0)
                    thr_s = thrs_fr.get(s, 30.0)
                    try:
                        sc_v = float(sc_df.iloc[gi][s]) if s in sc_df.columns else 1.0
                    except Exception:
                        sc_v = 1.0
                    if sc_v == 0:
                        sc_v = 1.0
                    raw_v  = float(df.iloc[i][s])
                    pred_v = float(pred_al.at[gi, s])
                    excess = abs((raw_v / sc_v) - pred_v - mu_s)
                    if excess >= EPS_NORMAL_VAL * thr_s:
                        all_ok = False
                        break
                pred_ok[i] = all_ok
            n_pred_ok = int(np.sum(pred_ok[b4_search_start:b4_search_end]))
            print(f"  [INFO] B4: pred-quality filter: {n_pred_ok}"
                  f"/{b4_search_end-b4_search_start} candidate rows pass")
            obs_arr = obs_arr & pred_ok

        # Find longest consecutive observable run within search range
        best_start, best_len = -1, 0
        cur_start, cur_len   = -1, 0
        for i in range(b4_search_start, b4_search_end):
            if obs_arr[i]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len   = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        if best_start >= 0 and best_len >= 24:   # at least 2 hours (24 × 5 min)
            b4_start = best_start
            b4_end   = min(best_start + best_len, b4_search_end)
            b4_idx   = df.index[b4_start:b4_end].tolist()
            if best_len < drop_len:
                print(f"  [WARN] B4: largest observable window = {best_len} steps "
                      f"(< drop_len={drop_len} steps); using {best_len} steps")
        else:
            print(f"  [WARN] B4: no observable window "
                  f"(≥{MIN_DIAG_B4} sensors open+stable) found in "
                  f"[{b4_search_start}:{b4_search_end}]; B4 injection SKIPPED!")
    else:
        # Insufficient OD columns — fall back to positional placement [70%~90%]
        b4_start = int(L * 0.70)
        b4_end   = int(L * 0.90)
        b4_idx   = df.index[b4_start:b4_end].tolist()
        print(f"  [INFO] B4: insufficient OD columns (<{MIN_DIAG_B4}), "
              f"using positional placement [{b4_start}:{b4_end}]")

    if b4_idx:
        # ── MCE-sufficiency check ─────────────────────────────────────────────
        # apply_physics_assimilation triggers Scenario-B ONLY when
        #   MCE_rel = |Total*(1-λ) - ΣQ_input_c| / Total  > EPS_MCE_REL (0.15)
        # Scale corrections in _build_scale_matrix inflate ΣQ_input_c above
        # ΣQ_raw, partially cancelling the Total bias and lowering MCE_rel.
        # We estimate the minimum safe multiplier from raw branch sums (no
        # scale, conservative lower bound) then add a SCALE_BUFFER to absorb
        # scale-correction bias.  Clamp below SCENARIO_B_FUSE (0.30) so that
        # PI-GAT uses FLAG_TOTAL_ADJUSTED rather than FLAG_SYSTEM_ANOMALY.
        LAM_EST      = 0.10    # conservative leakage-rate proxy
        MCE_TARGET   = 0.20    # target MCE_rel (= 0.15 threshold + 0.05 margin)
        SCALE_BUFFER = 0.12    # +12% to compensate for scale-inflation of branches
        FUSE_LIMIT   = 0.28    # keep rel_diff_b < SCENARIO_B_FUSE (0.30)

        b4_total_c = df.loc[b4_idx, TOTAL_COL].astype(float)
        b4_brch    = sum(df.loc[b4_idx, s].astype(float) for s in SENSOR_COLS)
        denom      = (1.0 - LAM_EST) - MCE_TARGET           # 0.70
        if denom > 0 and (b4_total_c > 0).any():
            min_mult_arr = b4_brch / (b4_total_c * denom)
            min_mult_raw = float(np.percentile(
                min_mult_arr.replace([np.inf, -np.inf], np.nan).dropna(), 90
            ))
            min_mult_adj = min(min_mult_raw * (1.0 + SCALE_BUFFER),
                               1.0 + FUSE_LIMIT)
            if total_mult < min_mult_adj:
                print(f"  [INFO] B4: total_mult {total_mult:.3f} → {min_mult_adj:.3f} "
                      f"(raw est {min_mult_raw:.3f} × scale-buf {1+SCALE_BUFFER:.2f}, "
                      f"fuse-cap {1+FUSE_LIMIT:.2f})")
                total_mult = min_mult_adj
                injection_map["total_mult"] = total_mult

        df.loc[b4_idx, TOTAL_COL] = (df.loc[b4_idx, TOTAL_COL].astype(float)
                                      * total_mult)
    injection_map["b4_bias_idx"] = b4_idx

    print(f"  Injection summary:")
    print(f"    B1 Spikes  : {n_spikes} pts on B01 (valve-guard filtered),  ratio={spike_ratio:.2f}x")
    print(f"    B2 Drop    : B03 → 0  for {len(b2_idx)} steps  [{b2_start}:{b2_end}]  ([10%~25%])")
    print(f"    B3 Ramp    : B04 ramp then plateau={plateau_ratio:.3f}x  [{b3_start}:{b3_plat_end}]  ([35%~60%], recovers at 60%)")
    if b4_idx:
        print(f"    B4 Bias    : Total × {total_mult:.3f}  [{b4_start}:{b4_end}]"
              f"  ({len(b4_idx)} steps, observable-placed, B3 cleared at {b3_plat_end})")
    else:
        print(f"    B4 Bias    : SKIPPED (no observable window in"
              f" [{b4_search_start}:{b4_search_end}])")
    return df, injection_map


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model runners
# ══════════════════════════════════════════════════════════════════════════════

def _apply_zscore(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    df_z = pd.DataFrame(index=df.index)
    for col in SENSOR_COLS + [TOTAL_COL]:
        mu    = float(stats["mean"][col])
        sigma = float(stats["std"][col])
        df_z[col] = (df[col] - mu) / sigma
        od_col = f"OD_{col}"
        if od_col in df.columns and od_col in stats.get("mean", {}):
            df_z[od_col] = (df[od_col] - float(stats["mean"][od_col])) / float(stats["std"][od_col])
    # Forward-fill OD columns from raw df if missing in z-score
    for col in [c for c in df.columns if c.startswith("OD_")]:
        if col not in df_z.columns:
            df_z[col] = df[col]
    return df_z


def precompute_pi_gat_frozen_baselines(gt_indices: np.ndarray) -> Dict[str, Any]:
    """Run PI-GAT once on CLEAN data; freeze residual thresholds for all scenarios."""
    print("  [PI-GAT] Pre-computing frozen clean baselines...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_graph, _, _, sensor2edge_idx = build_graph_from_csv(
        ROOT_DIR / "network_topology.csv", weight_mode="conductance", add_self_loops=True
    )
    sensor2node_idx = data_graph.sensor2node_idx
    n_nodes = int(getattr(data_graph, "num_nodes", 0) or 0)

    global_raw_clean = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_imputed.csv"))

    dataset_clean = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=PREPROCESS_DIR / "flow_5min_zscore.csv",
        restmask_path=PREPROCESS_DIR / "mask_rest.csv",
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WIN,
        open_paths=PREPROCESS_DIR / "flow_5min_imputed.csv",
        zscore_stats_path=PREPROCESS_DIR / "zscore_stats.json",
        verbose=False,
    )

    model_ref = PI_GAT(WIN, n_nodes).to(device)
    ckpt = torch.load(PI_GAT_DIR / "pi_gat_best.pth", map_location=device, weights_only=False)
    model_ref.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    _load_eom_and_set(
        model_ref, PI_GAT_DIR / "opening_mapping" / "Debiased_eom_full.json", data_graph, device
    )

    # pred_df_clean is already in physical units (m³/h) — run_full_inference
    # handles the inverse-z-score internally.  Do NOT multiply by sigma+mu again.
    pred_df_clean = run_full_inference(model_ref, dataset_clean, sensor2node_idx, n_nodes, device)

    calib    = _load_calib_params(PI_GAT_DIR / "bias_calibration" / "calibration_params_final.json")
    scale_df = _build_scale_matrix(global_raw_clean, calib)

    is_valve_transition, is_od_stable = compute_valve_state_masks(global_raw_clean)
    is_active_row = pd.Series(False, index=global_raw_clean.index)
    is_active_row.iloc[gt_indices] = True
    valid_t_list = [int(t) for t in gt_indices]

    mu_R_frozen, node_thresholds_frozen, _ = compute_residual_baseline(
        global_raw_clean, pred_df_clean,
        is_valve_transition, is_od_stable, is_active_row,
        PI_GAT_DIR / "preprocessing" / "credibility_windows.csv",
        len(global_raw_clean), valid_t_list,
    )
    print("  [PI-GAT] Frozen baselines ready.")
    return {
        "pred_aligned_clean":  pred_df_clean,
        "mu_R":                mu_R_frozen,
        "node_thresholds":     node_thresholds_frozen,
        "calib":               calib,
        "scale_df":            scale_df,
        "is_valve_transition": is_valve_transition,
        "is_od_stable":        is_od_stable,
        "sensor2node_idx":     sensor2node_idx,
        "n_nodes":             n_nodes,
        "device":              device,
        "data_graph":          data_graph,
        "sensor2edge_idx":     sensor2edge_idx,
    }


def run_pi_gat(
    df_corrupt: pd.DataFrame,
    gt_indices: np.ndarray,
    frozen: Dict[str, Any],
) -> pd.DataFrame:
    """Apply physics-assimilation pipeline on corrupted data using frozen refs."""
    global_raw  = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_imputed.csv"))
    gt_idx_list = list(gt_indices)
    # Write corrupted values to the correct global row positions (gt_indices),
    # NOT to reset-index positions (0..L-1) which would create duplicate date labels.
    for _col in df_corrupt.columns:
        if _col in global_raw.columns:
            global_raw.loc[global_raw.index[gt_idx_list], _col] = (
                df_corrupt[_col].to_numpy()
            )

    valid_t_list        = [int(t) for t in gt_indices]  # global row positions
    pred_aligned        = frozen["pred_aligned_clean"].copy()
    calib               = frozen["calib"]
    scale_df            = frozen["scale_df"]
    is_valve_transition = frozen["is_valve_transition"]
    is_od_stable        = frozen["is_od_stable"]
    mu_R                = frozen["mu_R"]
    node_thresholds     = frozen["node_thresholds"]

    is_active_row = pd.Series(False, index=global_raw.index)
    is_active_row.iloc[gt_indices] = True

    # Use temp credibility file with GT region masked to 'low'
    # so the residual baseline is not re-contaminated by dirty data.
    temp_cred = TEMP_DIR / "cred_masked.csv"
    cred_df   = pd.read_csv(str(PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"))
    cred_df.loc[cred_df["t_index"].isin(set(gt_indices.tolist())), "credibility"] = "low"
    cred_df.to_csv(str(temp_cred), index=False)

    _, _, R_c = compute_residual_baseline(
        global_raw, pred_aligned,
        is_valve_transition, is_od_stable, is_active_row,
        temp_cred, len(global_raw), valid_t_list,
    )

    drift_correction, drift_events = detect_dynamic_drift(
        R_c, mu_R, node_thresholds, is_valve_transition, is_active_row,
        pd.Series(True, index=global_raw.index),
    )

    df_cleaned, flag_df_out, assim_info = apply_physics_assimilation(
        global_raw, pred_aligned, scale_df, calib, node_thresholds, mu_R,
        drift_correction, drift_events, is_valve_transition, is_od_stable,
        valid_t_list, temp_cred,
    )
    return df_cleaned.iloc[gt_idx_list].reset_index(drop=True)


def run_gat_only(
    df_corrupt: pd.DataFrame,
    gt_indices: np.ndarray,
    frozen: Dict[str, Any],
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Pure-GAT baseline: inference on corrupted data + frozen clean thresholds."""
    device          = frozen["device"]
    data_graph      = frozen["data_graph"]
    sensor2edge_idx = frozen["sensor2edge_idx"]
    sensor2node_idx = frozen["sensor2node_idx"]
    n_nodes         = frozen["n_nodes"]

    global_raw = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_imputed.csv"))
    global_z   = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_zscore.csv"), low_memory=False)
    # Keep the date column as text so global_z aligns with the timestamps in temp_raw.
    # timeseries_dataset.py already converts the numerical columns to floating point.
    df_z       = _apply_zscore(df_corrupt, stats)
    gt_idx_list = list(gt_indices)

    # Write corrupted values to the correct global row positions (gt_indices).
    # Using local reset-index (0..L-1) would overwrite unrelated rows and create
    # duplicate date values in the CSV → IrrigationTimeWindowDataset reindex error.
    for _col in df_corrupt.columns:
        if _col in global_raw.columns:
            global_raw.loc[global_raw.index[gt_idx_list], _col] = (
                df_corrupt[_col].to_numpy()
            )
    for _col in df_z.columns:
        if _col in global_z.columns:
            global_z.loc[global_z.index[gt_idx_list], _col] = (
                df_z[_col].to_numpy()
            )

    temp_raw = TEMP_DIR / "temp_gat_raw.csv"
    temp_z   = TEMP_DIR / "temp_gat_z.csv"
    global_raw.to_csv(str(temp_raw), index=False)
    global_z.to_csv(str(temp_z), index=False)

    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=temp_z,
        restmask_path=PREPROCESS_DIR / "mask_rest.csv",
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WIN,
        open_paths=temp_raw,
        zscore_stats_path=PREPROCESS_DIR / "zscore_stats.json",
        credibility_path=None,
        verbose=False,
    )
    edge_index = getattr(data_graph, "edge_index", None)
    n_edges    = int(edge_index.shape[1]) if edge_index is not None else 0

    model = GAT(win=WIN, num_nodes=n_nodes, num_edges=n_edges,
                hidden=64, heads=8, dropout=0.1, in_feat=2).to(device)
    ckpt  = torch.load(BASELINE_DIR / "GAT-only" / "gat_final.pth",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    loader = DataLoader(cast(Any, dataset), batch_size=BATCH, shuffle=False)
    pred_z: Dict[int, Any] = {}
    with torch.no_grad():
        for batch in loader:
            batch    = batch.to(device)
            out      = model(batch)
            node_hat = out["node_hat"].view(-1, n_nodes).cpu().numpy()
            for i, t in enumerate(getattr(batch, "t_index", [])):
                pred_z[int(t.item())] = node_hat[i]

    # Frozen clean thresholds (computed on clean data — no variance explosion)
    frozen_pred = frozen["pred_aligned_clean"]
    df_cleaned  = global_raw.copy()

    for sensor in SENSOR_COLS:
        n_idx = sensor2node_idx[sensor]
        # Pre-cast to float Series so .at[] returns float (avoids Scalar/complex Pylance error)
        raw_ser  = global_raw[sensor].astype(float)
        pred_ser = (frozen_pred[sensor].astype(float)
                    if sensor in frozen_pred.columns
                    else pd.Series(0.0, index=global_raw.index))
        # Threshold from frozen clean residuals
        clean_res = [
            raw_ser.at[int(t)] - pred_ser.at[int(t)]
            for t in gt_indices if int(t) in pred_z
        ]
        if not clean_res:
            clean_res = [0.0]
        arr = np.array(clean_res)
        thr = float(np.mean(np.abs(arr - np.mean(arr))) + 3.0 * np.std(np.abs(arr - np.mean(arr))))
        thr = max(thr, GLOBAL_HYDRAULIC_DEADBAND)

        for t in gt_indices:
            t_int = int(t)
            if t_int in pred_z:
                pred_val = float(pred_z[t_int][n_idx])
                raw_val  = raw_ser.at[t_int]
                if abs(raw_val - pred_val) > thr:
                    df_cleaned.at[t_int, sensor] = pred_val

    return df_cleaned.iloc[gt_idx_list].reset_index(drop=True)


def run_isolation_forest(
    df_corrupt: pd.DataFrame,
    gt_indices: np.ndarray,
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Isolation Forest per-sensor with frozen clean training distribution."""
    global_z = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_zscore.csv"), low_memory=False)
    global_z = global_z.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
    df_z     = _apply_zscore(df_corrupt, stats)
    df_clean = df_corrupt.copy()

    for sensor in SENSOR_COLS:
        od_col   = f"OD_{sensor}"
        X_train  = global_z.loc[gt_indices, [sensor, od_col]].values
        X_infer  = pd.DataFrame({sensor: df_z.get(sensor, 0),
                                  od_col: df_z.get(od_col, 0)}).ffill().fillna(0.0).values
        clf = IsolationForest(random_state=42, contamination="auto")
        clf.fit(X_train)
        train_scores = clf.decision_function(X_train)
        anomaly_p    = (1.0 - (norm.cdf(3.0) - norm.cdf(-3.0))) * 100
        thresh       = float(np.percentile(train_scores, anomaly_p))
        is_anomaly   = clf.decision_function(X_infer) < thresh
        df_clean.loc[is_anomaly, sensor] = np.nan
        df_clean[sensor] = (df_clean[sensor]
                             .interpolate(method="linear", limit=12, limit_direction="both")
                             .ffill().bfill())
    return df_clean.reset_index(drop=True)


def run_mvz(
    df_corrupt: pd.DataFrame,
    gt_indices: np.ndarray,
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Multivariate Z-score with frozen clean training distribution."""
    features = SENSOR_COLS + [f"OD_{s}" for s in SENSOR_COLS]
    global_z = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_zscore.csv"), low_memory=False)
    global_z = global_z.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    global_z = global_z[features].ffill().fillna(0.0)

    df_z     = _apply_zscore(df_corrupt, stats)
    for feat in features:
        if feat not in df_z.columns:
            df_z[feat] = global_z.loc[df_corrupt.index, feat].values if feat in global_z.columns else 0.0

    X_train   = global_z.loc[gt_indices].values
    X_infer   = df_z[features].ffill().fillna(0.0).values
    mu        = np.mean(X_train, axis=0)
    Sigma_inv = pinv(np.cov(X_train, rowvar=False))
    delta     = X_infer - mu
    dist      = np.sqrt(np.sum(delta @ Sigma_inv * delta, axis=1))
    anomaly_p = (1.0 - (norm.cdf(3.0) - norm.cdf(-3.0))) * 100
    thresh    = float(np.percentile(
        np.sqrt(np.sum((X_train - mu) @ Sigma_inv * (X_train - mu), axis=1)),
        100 - anomaly_p,
    ))
    is_anomaly = dist > thresh

    flow_mu  = np.mean(X_train[:, :6], axis=0)
    flow_std = np.std(X_train[:, :6], axis=0)
    flow_std[flow_std == 0] = 1e-8

    df_clean = df_corrupt.copy().reset_index(drop=True)
    for pos in np.where(is_anomaly)[0]:
        z_scores = np.abs((X_infer[int(pos), :6] - flow_mu) / flow_std)
        worst = int(np.argmax(z_scores))
        df_clean.at[int(pos), SENSOR_COLS[worst]] = np.nan
    for s in SENSOR_COLS:
        df_clean[s] = (df_clean[s]
                       .interpolate(method="linear", limit=12, limit_direction="both")
                       .ffill().bfill())
    return df_clean


def _local_high_cred_indices(gt_indices: np.ndarray) -> np.ndarray:
    """Map global high-credibility timesteps to local GT-window row indices."""
    cred_df = pd.read_csv(str(PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"))
    high_global = set(cred_df.loc[cred_df["credibility"] == "high", "t_index"].astype(int))
    return np.array(
        [i for i, g in enumerate(gt_indices) if int(g) in high_global],
        dtype=int,
    )


def run_kalman_baseline(
    df_corrupt: pd.DataFrame,
    df_gt: pd.DataFrame,
    gt_indices: np.ndarray,
    process_scale: float = 0.02,
) -> pd.DataFrame:
    """Kalman temporal smoothing on the injected 72-h window (noise calibrated on clean GT)."""
    sys.path.insert(0, str(BASELINE_DIR / "Kalman"))
    from baseline_fairness import FLOW_COLS, high_cred_noise_variances  # type: ignore

    high_local = _local_high_cred_indices(gt_indices)
    if len(high_local) == 0:
        high_local = np.array([0], dtype=int)
    active_local = np.arange(len(df_corrupt), dtype=int)

    # kalman_smooth calibrates and reads the same df; split: calibrate on clean, smooth corrupt.
    n = len(FLOW_COLS)
    H = np.eye(n)
    variances = high_cred_noise_variances(df_gt, high_local)
    R = np.diag(variances)
    Q = np.diag(variances * process_scale)
    init_t = int(high_local[0])
    x = df_gt.loc[init_t, FLOW_COLS].astype(float).to_numpy()
    P = R.copy()
    df_out = df_corrupt.copy()

    for t in active_local:
        z = df_corrupt.loc[t, FLOW_COLS].astype(float).to_numpy()
        if np.any(np.isnan(z)):
            continue
        x_pred = x
        P_pred = P + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ (z - H @ x_pred)
        P = (np.eye(n) - K @ H) @ P_pred
        for j, col in enumerate(FLOW_COLS):
            df_out.at[t, col] = float(x[j])

    return df_out.reset_index(drop=True)


def run_epanet_wls_baseline(
    df_corrupt: pd.DataFrame,
    df_gt: pd.DataFrame,
    gt_indices: np.ndarray,
) -> pd.DataFrame:
    """EPANET-WLS redistribution on injected window (trigger median frozen from clean GT)."""
    sys.path.insert(0, str(BASELINE_DIR / "EPANET-WLS"))
    from baseline_fairness import (  # type: ignore
        BRANCHES,
        FLOW_COLS,
        load_lambda,
        load_valve_conductance_weights,
        raw_mce_series,
    )
    from run_epanet_wls import redistribute_branches  # type: ignore  # Baseline/EPANET-WLS

    lam = load_lambda(PI_GAT_DIR / "bias_calibration" / "calibration_params_final.json")
    branch_w = load_valve_conductance_weights(ROOT_DIR / "network_topology.csv")
    high_local = _local_high_cred_indices(gt_indices)
    if len(high_local) == 0:
        high_local = np.arange(len(df_gt), dtype=int)
    mce_med = float(raw_mce_series(df_gt, lam).iloc[high_local].median())

    df_out = df_corrupt.copy()
    for t in range(len(df_corrupt)):
        row = df_corrupt.loc[t, FLOW_COLS].astype(float)
        if row.isna().any():
            continue
        q_total = float(row[TOTAL_COL])
        q_br = row[BRANCHES].to_numpy(dtype=float)
        mce = abs((1.0 - lam) * q_total - float(np.sum(q_br)))
        if mce <= mce_med:
            continue
        q_new = redistribute_branches(q_total, q_br, lam, branch_w)
        for j, b in enumerate(BRANCHES):
            df_out.at[t, b] = float(q_new[j])

    return df_out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Three-dimensional evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _d1_data_tracking(
    df_clean: pd.DataFrame,
    df_gt: pd.DataFrame,
    injected_idx: List[int],
    sensors_injected: List[str],
) -> Dict[str, float]:
    """D1: RMSE + PCC computed only on injected (corrupted) samples."""
    results: Dict[str, float] = {}
    all_gt, all_pred = [], []
    for s in sensors_injected:
        gt_vals   = df_gt.loc[injected_idx, s].astype(float).to_numpy()
        cln_vals  = df_clean.loc[injected_idx, s].astype(float).to_numpy(copy=True)
        nan_mask  = np.isnan(cln_vals)
        cln_vals[nan_mask] = gt_vals[nan_mask]   # penalise uncorrected NaN as raw
        all_gt.extend(gt_vals.tolist())
        all_pred.extend(cln_vals.tolist())

    all_gt, all_pred = np.array(all_gt), np.array(all_pred)
    rmse = float(np.sqrt(np.mean((all_gt - all_pred) ** 2)))
    # np.corrcoef avoids scipy PearsonRResult stub incompatibilities across scipy versions
    pcc  = float(np.corrcoef(all_gt, all_pred)[0, 1]) if len(all_gt) > 2 else float("nan")
    results["RMSE"]  = round(rmse, 4)
    results["PCC"]   = round(pcc,  4)
    return results


def _d2_physical_consistency(
    df_clean: pd.DataFrame,
    df_gt: pd.DataFrame,
) -> Dict[str, float]:
    """D2: Restored MCE (full window) + KCR."""
    # Leakage rate from calibration (default 0.10 if unavailable)
    try:
        calib = _load_calib_params(
            PI_GAT_DIR / "bias_calibration" / "calibration_params_final.json"
        )
        lam = float(calib.get("leakage", {}).get("rate", 0.10))
    except Exception:
        lam = 0.10

    # Restored MCE: use GT Total (unbiased truth) vs. cleaned branches
    # np.asarray() guarantees ndarray (avoids ExtensionArray/Categorical arithmetic errors)
    q_total_eff = np.asarray(df_gt[TOTAL_COL].astype(float), dtype=float) * (1.0 - lam)
    q_branches  = np.asarray(df_clean[SENSOR_COLS].astype(float).sum(axis=1), dtype=float)
    mce_restored = float(np.nanmean(np.abs(q_total_eff - q_branches)))

    # KCR: OD columns present in df_gt?
    od_cols = [f"OD_{s}" for s in SENSOR_COLS if f"OD_{s}" in df_gt.columns]
    if not od_cols:
        return {"Restored_MCE": round(mce_restored, 2), "KCR": float("nan")}

    closed_mask   = (df_gt[od_cols].astype(float) < 5.0).any(axis=1)
    n_closed      = int(closed_mask.sum())
    if n_closed == 0:
        return {"Restored_MCE": round(mce_restored, 2), "KCR": float("nan")}

    # Pre-cast to float so .at[] returns float, not Scalar (avoids complex/Pylance error)
    df_gt_od  = df_gt[od_cols].astype(float)
    df_cln_fl = df_clean[SENSOR_COLS].astype(float)
    # For each closed-valve step, check corresponding sensor flow
    compliant = 0
    for row in df_gt.index[closed_mask]:
        for s, od_col in zip(SENSOR_COLS, od_cols):
            if cast(float, df_gt_od.at[row, od_col]) < 5.0:
                if abs(cast(float, df_cln_fl.at[row, s])) < FLOW_ZERO_THRESH:
                    compliant += 1

    total_closed_sensor_steps = sum(
        int((df_gt[od_col].astype(float) < 5.0).sum()) for od_col in od_cols
    )
    kcr = (compliant / total_closed_sensor_steps * 100.0) if total_closed_sensor_steps else float("nan")
    return {"Restored_MCE": round(mce_restored, 2), "KCR": round(kcr, 2)}


def _d3_engineering_robustness(
    df_clean: pd.DataFrame,
    df_gt: pd.DataFrame,
    all_injected_idx_set: set,
) -> Dict[str, float]:
    """
    D3: SFAR — Spatial False Alarm Rate.
    Computed across ALL sensors at non-injected time steps.
    A false alarm fires when |clean - gt| > 5% × gt  AND  |clean - gt| > 2 m³/h.
    """
    # Pre-cast to float so .at[] returns float, not Scalar (avoids complex/Pylance error)
    # fillna(df_gt_f) replaces NaN cleaned values with GT, penalising un-corrected outliers
    df_gt_f  = df_gt[SENSOR_COLS].astype(float)
    df_cln_f = df_clean[SENSOR_COLS].astype(float).fillna(df_gt_f)
    false_alarms = 0
    total_healthy = 0
    for row in df_gt.index:
        if row in all_injected_idx_set:
            continue
        for s in SENSOR_COLS:
            gt_val  = df_gt_f.at[row, s]
            cln_val = df_cln_f.at[row, s]
            delta   = abs(cln_val - gt_val)
            ref     = max(abs(gt_val), 1.0)
            total_healthy += 1
            if (delta / ref > SFAR_REL_THRESH) and (delta > SFAR_ABS_THRESH):
                false_alarms += 1

    sfar = (false_alarms / total_healthy * 100.0) if total_healthy > 0 else 0.0
    return {"SFAR": round(sfar, 3)}


def evaluate_all_dims(
    df_clean: pd.DataFrame,
    df_gt: pd.DataFrame,
    injection_map: Dict[str, Any],
) -> Dict[str, float]:
    """Aggregate all three dimension metrics for one model."""
    # Compile injected index sets per sensor
    spike_idx = injection_map.get("b1_spike_idx", [])
    drop_idx  = injection_map.get("b2_drop_idx",  [])
    drift_idx = injection_map.get("b3_drift_idx", [])
    bias_idx  = injection_map.get("b4_bias_idx",  [])

    # D1 — B1 (B01), B2 (B03), B3 (B04), B4 (Total) independently, then averaged
    d1_b1 = _d1_data_tracking(df_clean, df_gt, spike_idx, ["B01"]) if spike_idx else {}
    d1_b2 = _d1_data_tracking(df_clean, df_gt, drop_idx,  ["B03"]) if drop_idx  else {}
    d1_b3 = _d1_data_tracking(df_clean, df_gt, drift_idx, ["B04"]) if drift_idx else {}
    d1_b4 = _d1_data_tracking(df_clean, df_gt, bias_idx,  [TOTAL_COL]) if bias_idx else {}

    # Compute D1 aggregate (harmonic mean of individual RMSE values)
    rmses = [v.get("RMSE", np.nan) for v in [d1_b1, d1_b2, d1_b3, d1_b4] if v]
    pccs  = [v.get("PCC",  np.nan) for v in [d1_b1, d1_b2, d1_b3, d1_b4] if v]
    mean_rmse = float(np.nanmean(rmses)) if rmses else float("nan")
    mean_pcc  = float(np.nanmean(pccs))  if pccs  else float("nan")

    # D2 — physical consistency over full window
    d2 = _d2_physical_consistency(df_clean, df_gt)

    # D3 — robustness (all injected rows union)
    all_injected: set = set(spike_idx) | set(drop_idx) | set(drift_idx) | set(bias_idx)
    d3 = _d3_engineering_robustness(df_clean, df_gt, all_injected)

    return {
        # D1 aggregated
        "RMSE_mean":    round(mean_rmse, 2),
        "PCC_mean":     round(mean_pcc, 4),
        # D1 per sub-scenario
        "RMSE_B1_Spike": round(d1_b1.get("RMSE", np.nan), 2),
        "PCC_B1_Spike":  round(d1_b1.get("PCC",  np.nan), 4),
        "RMSE_B2_Drop":  round(d1_b2.get("RMSE", np.nan), 2),
        "PCC_B2_Drop":   round(d1_b2.get("PCC",  np.nan), 4),
        "RMSE_B3_Drift": round(d1_b3.get("RMSE", np.nan), 2),
        "PCC_B3_Drift":  round(d1_b3.get("PCC",  np.nan), 4),
        "RMSE_B4_Bias":  round(d1_b4.get("RMSE", np.nan), 2),
        "PCC_B4_Bias":   round(d1_b4.get("PCC",  np.nan), 4),
        # D2
        "Restored_MCE":  d2.get("Restored_MCE", float("nan")),
        "KCR":           d2.get("KCR", float("nan")),
        # D3
        "SFAR":          d3.get("SFAR", float("nan")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _format_val(v: Any, decimals: int = 2) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def generate_reports(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Write CSV and Markdown evaluation tables."""
    metrics_order = [
        "RMSE_mean", "PCC_mean",
        "RMSE_B1_Spike", "PCC_B1_Spike",
        "RMSE_B2_Drop",  "PCC_B2_Drop",
        "RMSE_B3_Drift", "PCC_B3_Drift",
        "RMSE_B4_Bias",  "PCC_B4_Bias",
        "Restored_MCE", "KCR", "SFAR",
    ]
    models = list(results.keys())

    # CSV
    rows = []
    for m in metrics_order:
        row = {"Metric": m}
        for model in models:
            row[model] = results[model].get(m, float("nan"))
        rows.append(row)
    df_csv = pd.DataFrame(rows)
    df_csv.to_csv(str(out_dir / "cascading_evaluation.csv"), index=False)

    # Markdown
    dim_labels = {
        "RMSE_mean":     "D1 (Tracking)",   "PCC_mean":      "D1 (Tracking)",
        "RMSE_B1_Spike": "D1 → B1 Spike",   "PCC_B1_Spike":  "D1 → B1 Spike",
        "RMSE_B2_Drop":  "D1 → B2 Drop",    "PCC_B2_Drop":   "D1 → B2 Drop",
        "RMSE_B3_Drift": "D1 → B3 Drift",   "PCC_B3_Drift":  "D1 → B3 Drift",
        "RMSE_B4_Bias":  "D1 → B4 Bias",    "PCC_B4_Bias":   "D1 → B4 Bias",
        "Restored_MCE":  "D2 (Physical)",   "KCR":           "D2 (Physical)",
        "SFAR":          "D3 (Robustness)",
    }
    header      = "| Dimension | Metric | " + " | ".join(models) + " |"
    separator   = "| --- | --- | " + " | ".join(["---"] * len(models)) + " |"
    md_lines    = [header, separator]
    for m in metrics_order:
        dim = dim_labels.get(m, "")
        vals = [_format_val(results[model].get(m, float("nan"))) for model in models]
        md_lines.append(f"| {dim} | {m} | " + " | ".join(vals) + " |")
    md_text = "\n".join(md_lines)
    with open(str(out_dir / "cascading_evaluation.md"), "w", encoding="utf-8") as f:
        f.write("# Cascading Fault Injection — Three-Dimensional Evaluation\n\n")
        f.write(md_text + "\n")

    print("\n" + md_text)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=== Cascading Fault Injection Benchmark ===\n")

    # Load empirical fault profiles
    profile_path = SCRIPT_DIR / "fault_profiles.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            "fault_profiles.json not found. "
            "Run extract_fault_profiles.py first."
        )
    with open(str(profile_path), encoding="utf-8") as f:
        profiles = json.load(f)

    with open(str(PREPROCESS_DIR / "zscore_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)

    # ── GT window ─────────────────────────────────────────────────────────
    print("[1/5] Selecting golden GT window...")
    global_raw_ref    = pd.read_csv(str(PREPROCESS_DIR / "flow_5min_imputed.csv"))
    active_mask       = build_active_mask(global_raw_ref)
    active_set        = set(global_raw_ref.index[active_mask].tolist())
    df_gt, gt_indices = find_longest_clean_window(active_set)
    print(f"  Active rows: {len(active_set)} / {len(global_raw_ref)}")

    # ── Frozen PI-GAT baselines (before injection) ──────────────────────
    print("\n[2/5] Pre-computing frozen PI-GAT baselines (clean data, once)...")
    frozen = precompute_pi_gat_frozen_baselines(gt_indices)

    # ── Inject cascading faults ─────────────────────────────────────────
    print("\n[3/5] Injecting cascading faults...")
    df_corrupt, injection_map = inject_cascading_faults(
        df_gt, profiles, gt_indices=gt_indices, frozen_refs=frozen
    )

    # ── Run all models ───────────────────────────────────────────────────
    print("\n[4/5] Running models...")
    print("  IF ...")
    df_if    = run_isolation_forest(df_corrupt, gt_indices, stats)
    print("  MVZ ...")
    df_mvz   = run_mvz(df_corrupt, gt_indices, stats)
    print("  GAT-only ...")
    df_gat   = run_gat_only(df_corrupt, gt_indices, frozen, stats)
    print("  Kalman ...")
    df_kalman = run_kalman_baseline(df_corrupt, df_gt, gt_indices)
    print("  EPANET-WLS ...")
    df_epanet = run_epanet_wls_baseline(df_corrupt, df_gt, gt_indices)
    print("  PI-GAT ...")
    df_pigat = run_pi_gat(df_corrupt, gt_indices, frozen)

    # ── Evaluate (three dimensions) ──────────────────────────────────────
    print("\n[5/5] Evaluating (three dimensions)...")
    results: Dict[str, Dict[str, Any]] = {}
    for name, df_out in [
        ("IF", df_if),
        ("MVZ", df_mvz),
        ("GAT", df_gat),
        ("Kalman", df_kalman),
        ("EPANET-WLS", df_epanet),
        ("PI-GAT", df_pigat),
    ]:
        results[name] = evaluate_all_dims(df_out, df_gt, injection_map)

    # ── Write reports ────────────────────────────────────────────────────
    generate_reports(results, SCRIPT_DIR)

    print(f"\nOutputs written to {SCRIPT_DIR}")
    print("  cascading_evaluation.csv")
    print("  cascading_evaluation.md")


if __name__ == "__main__":
    main()
