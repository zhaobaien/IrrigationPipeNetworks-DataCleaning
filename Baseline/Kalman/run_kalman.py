#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman filter baseline — joint temporal smoothing of 7 flow meters.

Fairness:
  - Same input / active mask / high-cred noise calibration as IF & MVZ
  - No hard mass-balance projection (physics closure tested via Table 1 MCE)
  - lambda used only by M4 evaluator, not forced inside filter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from baseline_fairness import (  # noqa: E402
    FLOW_COLS,
    active_row_indices,
    high_cred_noise_variances,
    load_credibility_indices,
    load_imputed,
    load_lambda,
    project_paths,
)


def kalman_smooth(
    df_raw: pd.DataFrame,
    active_idx: np.ndarray,
    high_idx: np.ndarray,
    process_scale: float,
) -> pd.DataFrame:
    n = len(FLOW_COLS)
    H = np.eye(n)
    variances = high_cred_noise_variances(df_raw, high_idx)
    R = np.diag(variances)
    Q = np.diag(variances * process_scale)

    init_t = int(high_idx[0])
    x = df_raw.loc[init_t, FLOW_COLS].astype(float).to_numpy()
    P = R.copy()
    df_out = df_raw.copy()

    for t in sorted(int(i) for i in active_idx):
        z = df_raw.loc[t, FLOW_COLS].astype(float).to_numpy()
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

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalman temporal smoothing baseline")
    parser.add_argument("--process-scale", type=float, default=0.02)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    paths = project_paths()
    print("=== Kalman Baseline (temporal smoothing) ===")
    lam = load_lambda()
    print(f"  lambda (eval only) = {lam:.4f}")

    df_raw = load_imputed()
    high_idx, _ = load_credibility_indices()
    active_idx = active_row_indices(df_raw)
    print(f"  high-cred: {len(high_idx)}, active: {len(active_idx)}")

    df_clean = kalman_smooth(df_raw, active_idx, high_idx, args.process_scale)

    out_dir = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / "cleaning_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "flow_5min_cleaned_kalman.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"[Done] Saved: {out_csv}")


if __name__ == "__main__":
    main()
