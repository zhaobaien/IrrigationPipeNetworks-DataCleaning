#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
from pathlib import Path
from sklearn.ensemble import IsolationForest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-sigma", type=float, default=3.0, help="k-sigma threshold")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print(f"=== Isolation Forest Baseline (Fair & Strong) [k={args.k_sigma}] ===")

    # 1. Set the paths.
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    preprocess_dir = project_root / "Preprocess_Timeseries" / "outputs" / "preprocessed"

    csv_raw = preprocess_dir / "flow_5min_imputed.csv"
    csv_zscore = preprocess_dir / "flow_5min_zscore.csv"
    csv_cred = project_root / "PI-GAT" / "preprocessing" / "credibility_windows.csv"

    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "cleaning_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "flow_5min_cleaned_if.csv"
    
    df_raw = pd.read_csv(csv_raw)
    # Use low_memory=False to avoid mixed-type warnings and coerce input columns to numeric values.
    # Coercion prevents malformed tokens such as '?' from breaking to_numpy(dtype=float).
    df_zscore = pd.read_csv(csv_zscore, low_memory=False)
    df_zscore = df_zscore.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_cred = pd.read_csv(csv_cred)
    
    # 3. Select high-credibility windows for training.
    high_cred_indices = df_cred[df_cred['credibility'] == 'high']['t_index'].to_numpy(dtype=int)
    active_indices = df_cred['t_index'].to_numpy(dtype=int)
    
    print(f"Total rows: {len(df_raw)}, Active windows: {len(active_indices)}, High credibility (Train) windows: {len(high_cred_indices)}")
    
    # 4. Train an independent detector for each sensor.
    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]
    df_cleaned = df_raw.copy()
    
    total_anomalies_detected = 0
    
    for sensor in sensors:
        od_col = f"OD_{sensor}"
        flow_col = sensor
        
        # Use standardised flow and valve opening as the two features.
        X_all = df_zscore[[flow_col, od_col]].to_numpy(dtype=float)
        X_train = X_all[high_cred_indices]
        X_infer = X_all[active_indices] # Run inference on all active windows.
        
        # Fit the Isolation Forest only on high-credibility records to limit contamination.
        clf = IsolationForest(random_state=42, contamination='auto')
        clf.fit(X_train)
        
        # Smaller decision scores indicate more anomalous observations.
        # Use the lower training-score quantile corresponding to the normal k-sigma tail.
        train_scores = clf.decision_function(X_train)
        score_thresh = np.percentile(train_scores, (1 - (norm.cdf(args.k_sigma) - norm.cdf(-args.k_sigma))) * 100)
        
        # Score and flag anomalous readings in the active data.
        infer_scores = clf.decision_function(X_infer)
        is_anomaly = infer_scores < score_thresh
        num_anomalies = np.sum(is_anomaly)
        total_anomalies_detected += num_anomalies
        print(f"Sensor {sensor}: Train threshold score = {score_thresh:.4f}, anomalies detected = {num_anomalies}")
        
        # A. Replace anomalous readings with NaN.
        anomaly_t_indices = active_indices[is_anomaly]
        df_cleaned.loc[anomaly_t_indices, flow_col] = np.nan
        
        # B. Linearly interpolate flagged readings with a 12-timestep (one-hour) limit in either direction.
        df_cleaned[flow_col] = df_cleaned[flow_col].interpolate(method='linear', limit=12, limit_direction='both')
    
    # 5. Write the cleaned data and summarise the results.
    df_cleaned.to_csv(out_csv, index=False)
    print(f"\\n - Total anomalies handled across all sensors: {total_anomalies_detected}")
    print(f" - Cleaned IF data saved to: {out_csv}")
    print("√ IF script execution complete!")

if __name__ == "__main__":
    main()
