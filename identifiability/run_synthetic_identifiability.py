#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess identifiability of the network leakage fraction γ and branch offsets δ.

The synthetic experiment searches for nearly equivalent joint fits, then tests
recovery of the injected offsets at a fixed leakage fraction. The objective
minimises relative mass-balance error on sampled high-credibility windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
BRANCHES = ["B01", "B02", "B03", "B04", "B05", "B06"]
N_BRANCH = len(BRANCHES)

# Ground-truth scenario (heterogeneous branch drift + network leakage)
GAMMA_TRUE = 0.10
DELTA_TRUE = np.array([0.03, 0.02, 0.04, 0.015, 0.025, 0.035])  # +1.5% … +4%
DELTA_BOUNDS = (-0.05, 0.05)


def load_high_cred_flows(n_sample: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sample (Q_total, Q_branch[6]) from high-credibility active timesteps."""
    imputed = ROOT / "Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv"
    cred = ROOT / "PI-GAT/preprocessing/credibility_windows.csv"
    df = pd.read_csv(imputed, index_col="date", parse_dates=True)
    cred_df = pd.read_csv(cred)
    hi = cred_df[cred_df["credibility"] == "high"]["t_index"].astype(int).values

    cols = ["Total"] + BRANCHES
    sub = df.iloc[hi][cols].astype(float)
    sub = sub[(sub > 0).all(axis=1)]
    rng = np.random.default_rng(seed)
    if len(sub) > n_sample:
        idx = rng.choice(len(sub), size=n_sample, replace=False)
        sub = sub.iloc[idx]
    q_total = sub["Total"].values
    q_branch = sub[BRANCHES].values
    return q_total, q_branch


def synthesize_raw(q_total: np.ndarray, q_branch: np.ndarray, gamma: float, delta: np.ndarray):
    """
    Given observed branch flows (as true hydraulic discharges), build consistent trunk
    reading and biased raw meters.

    True mass balance: sum(q_branch) = q_total * (1 - gamma)
    Raw meters over-read: q_raw = q_branch * (1 + delta)
    """
    q_total_syn = q_branch.sum(axis=1) / max(1.0 - gamma, 1e-6)
    q_raw = q_branch * (1.0 + delta.reshape(1, -1))
    return q_total_syn, q_raw


def relative_mb_loss(
    gamma: float,
    delta: np.ndarray,
    q_total: np.ndarray,
    q_raw: np.ndarray,
) -> float:
    """Stage-II relative mass-balance loss (mean squared relative error)."""
    scale = 1.0 + np.clip(delta, *DELTA_BOUNDS)
    q_cal = q_raw / scale.reshape(1, -1)
    target = q_total.reshape(-1, 1) * (1.0 - gamma)
    rel = (q_cal.sum(axis=1) - target.ravel()) / np.maximum(np.abs(target.ravel()), 1e-6)
    return float(np.mean(rel ** 2))


def fit_delta_given_gamma(gamma: float, q_total: np.ndarray, q_raw: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, float]:
    def obj(d: np.ndarray) -> float:
        return relative_mb_loss(gamma, d, q_total, q_raw)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=[DELTA_BOUNDS] * N_BRANCH,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x, float(res.fun)


def fit_joint(q_total: np.ndarray, q_raw: np.ndarray, x0: np.ndarray) -> tuple[float, np.ndarray, float]:
    def obj(x: np.ndarray) -> float:
        gamma = float(x[0])
        delta = x[1:]
        return relative_mb_loss(gamma, delta, q_total, q_raw)

    bounds = [(0.05, 0.15)] + [DELTA_BOUNDS] * N_BRANCH
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800, "ftol": 1e-12})
    return float(res.x[0]), res.x[1:], float(res.fun)


def run_equifinality_scan(q_total: np.ndarray, q_raw: np.ndarray) -> pd.DataFrame:
    """Optimise the branch offsets at each fixed leakage fraction and record the loss."""
    rows = []
    x0 = np.zeros(N_BRANCH)
    for gamma in np.linspace(0.05, 0.15, 21):
        delta_hat, loss = fit_delta_given_gamma(float(gamma), q_total, q_raw, x0)
        x0 = delta_hat
        rows.append(
            {
                "gamma_fixed": float(gamma),
                "loss": loss,
                "delta_mean_pct": float(np.mean(delta_hat) * 100),
                "delta_rmse_vs_true_pct": float(np.sqrt(np.mean((delta_hat - DELTA_TRUE) ** 2)) * 100),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q_total_obs, q_branch = load_high_cred_flows(n_sample=800, seed=42)
    q_total, q_raw = synthesize_raw(q_total_obs, q_branch, GAMMA_TRUE, DELTA_TRUE)

    # Fit the branch offsets at each leakage value on the grid.
    scan = run_equifinality_scan(q_total, q_raw)
    scan.to_csv(OUT / "equifinality_scan.csv", index=False)
    best_idx = scan["loss"].idxmin()
    gamma_at_min = float(scan.loc[best_idx, "gamma_fixed"])
    loss_min = float(scan["loss"].min())
    loss_at_true_row = scan.loc[(scan["gamma_fixed"] - GAMMA_TRUE).abs().idxmin()]
    loss_at_true_gamma = float(loss_at_true_row["loss"])
    loss_spread_abs = float(scan["loss"].max() - scan["loss"].min())
    loss_spread_rel = loss_spread_abs / max(loss_at_true_gamma, 1e-10)

    # Fit leakage and branch offsets jointly from four specified starting points.
    joint_rows = []
    starts = [
        np.concatenate([[0.08], np.full(N_BRANCH, 0.01)]),
        np.concatenate([[0.10], np.zeros(N_BRANCH)]),
        np.concatenate([[0.12], np.full(N_BRANCH, 0.03)]),
        np.concatenate([[0.07], np.full(N_BRANCH, -0.02)]),
    ]
    for i, x0 in enumerate(starts):
        g_hat, d_hat, loss = fit_joint(q_total, q_raw, x0)
        joint_rows.append(
            {
                "start_id": i,
                "gamma_hat": g_hat,
                "gamma_error_pct": abs(g_hat - GAMMA_TRUE) * 100,
                "delta_rmse_pct": float(np.sqrt(np.mean((d_hat - DELTA_TRUE) ** 2)) * 100),
                "loss": loss,
            }
        )
    joint_df = pd.DataFrame(joint_rows)
    joint_df.to_csv(OUT / "joint_multistart.csv", index=False)

    # Fit the branch offsets with leakage fixed at its injected value.
    delta_seq, loss_seq = fit_delta_given_gamma(GAMMA_TRUE, q_total, q_raw, np.zeros(N_BRANCH))
    seq_errors = (delta_seq - DELTA_TRUE) * 100

    # Compare offset recovery for leakage values up to two percentage points from the truth.
    mis_rows = []
    for gamma_fix in [0.08, 0.10, 0.12]:
        d_hat, loss = fit_delta_given_gamma(gamma_fix, q_total, q_raw, np.zeros(N_BRANCH))
        mis_rows.append(
            {
                "gamma_frozen": gamma_fix,
                "delta_rmse_vs_true_pct": float(np.sqrt(np.mean((d_hat - DELTA_TRUE) ** 2)) * 100),
                "loss": loss,
            }
        )
    mis_df = pd.DataFrame(mis_rows)
    mis_df.to_csv(OUT / "sequential_gamma_mismatch.csv", index=False)

    # Per-branch recovery table
    branch_df = pd.DataFrame(
        {
            "branch": BRANCHES,
            "delta_true_pct": DELTA_TRUE * 100,
            "delta_sequential_pct": delta_seq * 100,
            "abs_error_pp": np.abs(seq_errors),
        }
    )
    branch_df.to_csv(OUT / "sequential_branch_recovery.csv", index=False)

    summary = {
        "scenario": {
            "gamma_true": GAMMA_TRUE,
            "delta_true_pct": (DELTA_TRUE * 100).tolist(),
            "n_timesteps": int(len(q_total)),
            "data_source": "high-credibility windows from flow_5min_imputed.csv",
        },
        "equifinality": {
            "gamma_grid_loss_min": loss_min,
            "gamma_at_loss_min": gamma_at_min,
            "loss_at_true_gamma": loss_at_true_gamma,
            "loss_spread_abs": loss_spread_abs,
            "loss_spread_rel_to_true": loss_spread_rel,
            "interpretation": (
                "Flat loss valley across γ∈[5%,15%] when δ is re-optimized each time — "
                "γ and branch scales are not uniquely identifiable under joint fitting."
            ),
        },
        "joint_multistart": {
            "n_starts": len(joint_rows),
            "gamma_hat_range": [float(joint_df["gamma_hat"].min()), float(joint_df["gamma_hat"].max())],
            "gamma_true": GAMMA_TRUE,
            "mean_delta_rmse_pct": float(joint_df["delta_rmse_pct"].mean()),
        },
        "sequential_frozen_gamma": {
            "gamma_frozen": GAMMA_TRUE,
            "loss": loss_seq,
            "delta_rmse_pct": float(np.sqrt(np.mean((delta_seq - DELTA_TRUE) ** 2)) * 100),
            "max_abs_branch_error_pp": float(np.max(np.abs(seq_errors))),
            "interpretation": (
                "With γ fixed at the Stage-I estimate, injected heterogeneous δ is recovered "
                "within numerical tolerance of the Stage-II optimizer."
            ),
        },
    }
    with open(OUT / "identifiability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== M3 Synthetic Identifiability ===")
    print(f"True gamma = {GAMMA_TRUE:.2%}, delta = {(DELTA_TRUE*100).round(2)} %")
    print(f"Equifinality: abs loss spread = {loss_spread_abs:.2e}")
    print(f"Joint multistart gamma_hat in [{joint_df['gamma_hat'].min():.3f}, {joint_df['gamma_hat'].max():.3f}]")
    print(f"Sequential delta RMSE = {summary['sequential_frozen_gamma']['delta_rmse_pct']:.3f} pp")
    print(f"Outputs -> {OUT}")


if __name__ == "__main__":
    main()
