#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv

def mahalanobis_distance(x, mu, Sigma_inv):
    """Compute the Mahalanobis distance for each observation."""
    delta = x - mu
    return np.sqrt(np.sum(delta @ Sigma_inv * delta, axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-sigma", type=float, default=3.0, help="k-sigma threshold")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print(f"=== Multi-Variate Z-score Baseline (Fair & Strong) [k={args.k_sigma}] ===")

    # 1. Set the paths.
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    preprocess_dir = project_root / "Preprocess_Timeseries" / "outputs" / "preprocessed"

    csv_raw = preprocess_dir / "flow_5min_imputed.csv"
    csv_zscore = preprocess_dir / "flow_5min_zscore.csv"
    csv_cred = project_root / "PI-GAT" / "preprocessing" / "credibility_windows.csv"

    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "cleaning_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "flow_5min_cleaned_mvz.csv"
    
    df_raw = pd.read_csv(csv_raw)
    # Use low_memory=False to avoid mixed-type warnings and coerce input columns to numeric values.
    df_zscore = pd.read_csv(csv_zscore, low_memory=False)
    df_zscore = df_zscore.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_cred = pd.read_csv(csv_cred)
    
    # Forward-fill missing features such as valve openings, then fill any remaining gaps with zero.
    feature_cols = ["B01", "B02", "B03", "B04", "B05", "B06"] + [f"OD_{s}" for s in ["B01", "B02", "B03", "B04", "B05", "B06"]]
    df_zscore[feature_cols] = df_zscore[feature_cols].ffill().fillna(0.0)
    
    # 3. Read the active and high-credibility window indices.
    high_cred_indices = df_cred[df_cred['credibility'] == 'high']['t_index'].to_numpy(dtype=int)
    active_indices = df_cred['t_index'].to_numpy(dtype=int)
    
    print(f"Total rows: {len(df_raw)}, Active windows: {len(active_indices)}, High credibility (Train) windows: {len(high_cred_indices)}")
    
    # 4. Build the 12-feature network matrix.
    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]
    
    X_all = df_zscore[feature_cols].to_numpy(dtype=float)
    X_train = X_all[high_cred_indices]
    X_infer = X_all[active_indices]
    
    # Estimate the joint distribution using only high-credibility observations.
    mu = np.mean(X_train, axis=0)
    Sigma = np.cov(X_train, rowvar=False)
    # Use a pseudoinverse to accommodate a singular covariance matrix.
    Sigma_inv = pinv(Sigma)
    
# Use the k-sigma quantile of the training Mahalanobis distances as the cutoff.
    dist_train = mahalanobis_distance(X_train, mu, Sigma_inv)
    anomaly_p = (1 - (norm.cdf(args.k_sigma) - norm.cdf(-args.k_sigma))) * 100
    dist_thresh = np.percentile(dist_train, 100 - anomaly_p)
    print(f"Computed MV-Zscore {100 - anomaly_p:.4f}th percentile threshold: {dist_thresh:.4f}")
    
    # Compute univariate Gaussian statistics for attributing anomalies to individual readings.
    # Flow features occupy columns 0 to 5.
    flow_train = X_train[:, :6]
    flow_mu_1d = np.mean(flow_train, axis=0)
    flow_std_1d = np.std(flow_train, axis=0)
    # Guard against division by a zero standard deviation.
    flow_std_1d[flow_std_1d == 0] = 1e-8 
    
    # 5. Run inference.
    dist_infer = mahalanobis_distance(X_infer, mu, Sigma_inv)
    is_anomaly = dist_infer > dist_thresh
    num_network_anomalies = np.sum(is_anomaly)
    print(f"Total network-level anomalies detected based on correlation: {num_network_anomalies}")
    
    df_cleaned = df_raw.copy()
    
    # 6. Attribute anomalies to a flow sensor.
    # Assume a single affected sensor at each flagged timestep and select the largest univariate deviation.
    anomaly_positions = np.where(is_anomaly)[0]
    punish_counts = {s: 0 for s in sensors}
    
    for pos in anomaly_positions:
        t_index_global = active_indices[pos]
        # Read the six flow features at this timestep.
        x_flows = X_infer[pos, :6]
        # Calculate the univariate standardised deviations for all six branches.
        z_scores_1d = np.abs((x_flows - flow_mu_1d) / flow_std_1d)
        worst_sensor_idx = np.argmax(z_scores_1d)
        culprit_sensor = sensors[worst_sensor_idx]
        
        # Mark the selected flow reading as NaN.
        df_cleaned.loc[t_index_global, culprit_sensor] = np.nan
        punish_counts[culprit_sensor] += 1
        
    print("Culprit assignment breakdown:")
    for k, v in punish_counts.items():
        print(f"  - {k}: penalized {v} times")
        
    # 7. Impute the flagged values.
    # Limit local linear interpolation to 12 consecutive masked timesteps (one hour).
    for sensor in sensors:
        df_cleaned[sensor] = df_cleaned[sensor].interpolate(method='linear', limit=12, limit_direction='both')
        
    # Save the results.
    df_cleaned.to_csv(out_csv, index=False)
    print(f"\\n - Cleaned MV-Zscore data saved to: {out_csv}")
    print("✓ MV-Zscore script execution complete!")

if __name__ == "__main__":
    main()
