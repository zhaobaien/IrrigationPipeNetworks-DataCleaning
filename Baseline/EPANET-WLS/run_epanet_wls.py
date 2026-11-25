#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPANET-style WLS baseline — topology-weighted mass-balance redistribution.

When raw MCE exceeds high-cred median on an active timestep:
  - Hold Total meter fixed (same as PI-GAT Scenario B spirit)
  - Minimally adjust branch flows with conductance weights from network_topology.csv
  - Same district / pipes as PipelineDataset-F (via shared topology CSV)

Fairness: high-cred statistics for triggering; lambda frozen from Stage-I JSON.
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
    BRANCHES,
    FLOW_COLS,
    TOTAL,
    active_row_indices,
    load_credibility_indices,
    load_imputed,
    load_lambda,
    load_valve_conductance_weights,
    project_paths,
    raw_mce_series,
)


def redistribute_branches(
    q_total: float,
    q_branches: np.ndarray,
    lam: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Minimal L2 adjustment on branches toward (1-lambda)*Q_T closure."""
    target = (1.0 - lam) * q_total
    gap = target - float(np.sum(q_branches))
    if abs(gap) < 1e-6:
        return q_branches.copy()
    w = np.maximum(weights, 1e-6)
    w = w / w.sum()
    return q_branches + gap * w


def epanet_wls_clean(
    df_raw: pd.DataFrame,
    active_idx: np.ndarray,
    high_idx: np.ndarray,
    lam: float,
    branch_weights: np.ndarray,
) -> pd.DataFrame:
    mce_med = float(raw_mce_series(df_raw, lam).iloc[high_idx].median())
    df_out = df_raw.copy()

    for t in sorted(int(i) for i in active_idx):
        row = df_raw.loc[t, FLOW_COLS].astype(float)
        if row.isna().any():
            continue
        q_total = float(row[TOTAL])
        q_br = row[BRANCHES].to_numpy(dtype=float)
        mce = abs((1.0 - lam) * q_total - float(np.sum(q_br)))
        if mce <= mce_med:
            continue
        q_new = redistribute_branches(q_total, q_br, lam, branch_weights)
        for j, b in enumerate(BRANCHES):
            df_out.at[t, b] = float(q_new[j])

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="EPANET-WLS conductance-weighted redistribution")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    paths = project_paths()
    print("=== EPANET-WLS Baseline ===")
    print(f"  topology: {paths['topology']}")
    lam = load_lambda()
    print(f"  lambda = {lam:.4f}")

    df_raw = load_imputed()
    high_idx, _ = load_credibility_indices()
    active_idx = active_row_indices(df_raw)
    branch_w = load_valve_conductance_weights()
    print(f"  high-cred: {len(high_idx)}, active: {len(active_idx)}")

    df_clean = epanet_wls_clean(df_raw, active_idx, high_idx, lam, branch_w)

    out_dir = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / "cleaning_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "flow_5min_cleaned_epanet_wls.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"[Done] Saved: {out_csv}")


if __name__ == "__main__":
    main()
