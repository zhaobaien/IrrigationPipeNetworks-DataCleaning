"""
Shared fairness settings for Baseline methods (IF / MVZ / Kalman / EPANET-WLS).

Aligns with PI-GAT pipeline conventions:
  - Input: flow_5min_imputed.csv
  - High-credibility windows: training / noise calibration (credibility == 'high')
  - Active mask: build_active_mask (26,472 expected)
  - lambda: Stage-I leakage from calibration_params_final.json (frozen)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASELINE_DIR.parent

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
from effective_value_filtering import build_active_mask, REQUIRED_COLUMNS  # noqa: E402

BRANCHES = ["B01", "B02", "B03", "B04", "B05", "B06"]
TOTAL = "Total"
FLOW_COLS = [TOTAL] + BRANCHES
DEFAULT_K_SIGMA = 3.0


def project_paths() -> dict[str, Path]:
    preprocess = PROJECT_ROOT / "Preprocess_Timeseries" / "outputs" / "preprocessed"
    return {
        "project_root": PROJECT_ROOT,
        "imputed": preprocess / "flow_5min_imputed.csv",
        "pipeline_f": PROJECT_ROOT / "PipelineDataset-F.csv",
        "topology": PROJECT_ROOT / "network_topology.csv",
        "credibility": PROJECT_ROOT / "PI-GAT" / "preprocessing" / "credibility_windows.csv",
        "calib_json": PROJECT_ROOT / "PI-GAT" / "bias_calibration" / "calibration_params_final.json",
    }


def load_lambda(calib_json: Path | None = None) -> float:
    path = calib_json or project_paths()["calib_json"]
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            rate = json.load(f).get("leakage", {}).get("rate")
        if rate is not None:
            return float(rate)
    return 0.10


def load_credibility_indices(cred_csv: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = cred_csv or project_paths()["credibility"]
    df_cred = pd.read_csv(path)
    high = df_cred[df_cred["credibility"] == "high"]["t_index"].to_numpy(dtype=int)
    active = df_cred["t_index"].to_numpy(dtype=int)
    return high, active


def load_imputed(imputed_path: Path | None = None) -> pd.DataFrame:
    path = imputed_path or project_paths()["imputed"]
    return pd.read_csv(path)


def high_cred_noise_variances(
    df: pd.DataFrame,
    high_indices: np.ndarray,
    *,
    min_var: float = 1.0,
) -> np.ndarray:
    """Per-sensor flow variance on high-credibility timesteps (7 sensors)."""
    vars_ = []
    for col in FLOW_COLS:
        v = df.loc[high_indices, col].astype(float).var()
        vars_.append(max(float(v), min_var))
    return np.array(vars_, dtype=float)


def mass_balance_matrix(lam: float) -> np.ndarray:
    """Row vector A with A @ x = 0  <=>  (1-lam)*Q_T - sum(Q_branch) = 0."""
    return np.array([-(1.0 - lam)] + [-1.0] * len(BRANCHES), dtype=float)


def project_onto_mass_balance(
    x: np.ndarray,
    lam: float,
    branch_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Conductance-weighted correction: keep Q_T, adjust branches toward mass closure.
    (1-lambda)*Q_T = sum(Q_branch).
    """
    x = np.asarray(x, dtype=float).copy()
    target_sum = (1.0 - lam) * x[0]
    branch_sum = float(np.sum(x[1:]))
    gap = target_sum - branch_sum
    if abs(gap) < 1e-9:
        return x
    w = np.ones(len(BRANCHES), dtype=float) if branch_weights is None else np.asarray(branch_weights, dtype=float)
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    x[1:] += gap * w
    return x


def raw_mce_series(df: pd.DataFrame, lam: float) -> pd.Series:
    sum_b = df[BRANCHES].sum(axis=1)
    return (df[TOTAL] * (1.0 - lam) - sum_b).abs()


def calibration_balance_variance(df: pd.DataFrame, high_indices: np.ndarray, lam: float) -> float:
    """Pseudo-measurement variance for soft mass-balance (high-cred raw MCE)."""
    mce = raw_mce_series(df, lam).iloc[high_indices].astype(float)
    v = float(mce.var())
    return max(v, 100.0)


def soft_wls_flows(
    z: np.ndarray,
    variances: np.ndarray,
    lam: float,
    balance_var: float,
) -> np.ndarray:
    """
    Augmented WLS: 7 flow meters + 1 soft mass-balance pseudo-observation (target 0).
    balance_var large => weak forcing (stays near raw); small => strong closure.
    """
    z = np.asarray(z, dtype=float)
    A = mass_balance_matrix(lam)
    H = np.vstack([np.eye(len(FLOW_COLS)), A.reshape(1, -1)])
    z_aug = np.concatenate([z, [0.0]])
    r_diag = np.concatenate([variances, [balance_var]])
    W = np.diag(1.0 / np.maximum(r_diag, 1.0))
    HTW = H.T @ W
    try:
        x = np.linalg.solve(HTW @ H, HTW @ z_aug)
    except np.linalg.LinAlgError:
        x = z.copy()
    return x


def constrained_wls_flows(
    z: np.ndarray,
    variances: np.ndarray,
    lam: float,
    balance_var: float | None = None,
) -> np.ndarray:
    """Snapshot WLS with soft mass-balance pseudo-measurement (EPANET-WLS)."""
    if balance_var is None:
        balance_var = 10000.0
    return soft_wls_flows(z, variances, lam, balance_var)


def load_valve_conductance_weights(topology_csv: Path | None = None) -> np.ndarray:
    """
    EPANET-style conductance proxy from network_topology.csv (C / L for each valve pipe).
    Returns length-6 weights aligned with BRANCHES order.
    """
    path = topology_csv or project_paths()["topology"]
    topo = pd.read_csv(path)
    valve_rows = topo[topo["Valve"].notna() & (topo["Valve"].astype(str).str.len() > 0)]
    weights = []
    for b in BRANCHES:
        row = valve_rows[valve_rows["Valve"] == b]
        if len(row) == 0:
            weights.append(1.0)
            continue
        r = row.iloc[0]
        c = float(r.get("C", 130) or 130)
        length = float(r.get("Length", 100) or 100)
        diam = float(r.get("Diameter", 0.4) or 0.4)
        cond = c / max(length, 1.0) * (diam ** 2.63)
        weights.append(max(cond, 1e-6))
    w = np.array(weights, dtype=float)
    return w / w.sum()


def active_row_indices(df: pd.DataFrame) -> np.ndarray:
    mask = build_active_mask(df, REQUIRED_COLUMNS)
    return np.where(mask.to_numpy())[0]
