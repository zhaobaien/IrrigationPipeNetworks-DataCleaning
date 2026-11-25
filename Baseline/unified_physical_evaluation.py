import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
import os
from scipy.stats import pearsonr
from effective_value_filtering import build_active_mask

def load_lambda_from_ckpt(ckpt_path):
    """Read the leakage fraction from the checkpoint state dictionary."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # The archived loader looks for a checkpoint leakage parameter.
    # Some fixed-leakage models store the range only as model configuration.
    # The calibration JSON can provide the final value through leakage.rate.
    # The checkpoint path is retained for compatibility with earlier models.
    if 'leakage_logit' in state_dict:
        logit = state_dict['leakage_logit']
        lam = 0.05 + 0.10 * torch.sigmoid(logit).item()
        print(f"[Info] Found 'leakage_logit' in {ckpt_path}. Extracted raw \u03bb = {lam:.4f}")
        return lam
    else:
        print("[Warning] 'leakage_logit' not found in checkpoint. Using default \u03bb = 0.10")
        return 0.10

def load_calibration_json_leakage(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('leakage', {}).get('rate', None)

def load_bias_params(bias_file, branches):
    """Read the archived instrument offsets from calibration_params_final.json.

    Use the mean offset across opening bins as the constant correction for each
    branch. This evaluator retains the archived simplification.
    """
    biases = {b: 0.0 for b in branches}
    if not os.path.exists(bias_file):
        return biases
    with open(bias_file, 'r', encoding='utf-8') as f:
        calib = json.load(f)
        
    for b in branches:
        if b in calib:
            # This archived evaluator uses a constant offset. An opening-dependent offset would have to be evaluated at each timestep.
            biases[b] = np.mean(calib[b].get("biases", [0.0]))  
    return biases

def get_eom_effective_opening(od_series, valve_name, eom_data):
    """Convert the recorded mechanical opening to effective opening in percent.

    The resulting values enter the Pearson correlation with the cleaned flow
    series. The mapping does not scale the opening by a maximum flow.
    """
    if valve_name not in eom_data:
        return np.zeros(len(od_series))
    
    cfg = eom_data[valve_name]
    x_knots = np.array(cfg["x_knots_pct"])
    y_knots = np.array(cfg.get("y_knots_pct", x_knots))
    
    # Map mechanical to effective opening with piecewise linear interpolation.
    od_eff = np.interp(od_series.fillna(0), x_knots, y_knots)
    return od_eff  

def main():
    root_dir = Path(__file__).parent.parent
    
    # 1. Set the input and output paths.
    imputed_data_path = root_dir / 'Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv'
    cred_file_path = root_dir / 'PI-GAT/preprocessing/credibility_windows.csv'
    bias_file_path = root_dir / 'PI-GAT/bias_calibration/calibration_params_final.json'
    eom_file_path  = root_dir / 'PI-GAT/opening_mapping/Debiased_eom_full.json'
    ckpt_path      = root_dir / 'PI-GAT/pi_gat_best.pth'
    
    # Discover the supplementary sensitivity runs dynamically.
    sensitivity_dir = root_dir / 'sensitivity_analysis'
    sensitivity_models = {}
    if sensitivity_dir.exists():
        for d in sensitivity_dir.iterdir():
            if d.is_dir() and d.name.startswith('sensitivity_analysis_'):
                val_str = d.name.split('_')[-1]
                if "-Sigma" in val_str:
                    # Combine the available model outputs.
                    if (d / "flow_5min_cleaned.csv").exists():
                        sensitivity_models[f"PI-GAT_{val_str}"] = d / "flow_5min_cleaned.csv"
                    if (d / "flow_5min_cleaned_gat.csv").exists():
                        sensitivity_models[f"GAT_{val_str}"] = d / "flow_5min_cleaned_gat.csv"
                    if (d / "flow_5min_cleaned_if.csv").exists():
                        sensitivity_models[f"IF_{val_str}"] = d / "flow_5min_cleaned_if.csv"
                    if (d / "flow_5min_cleaned_mvz.csv").exists():
                        sensitivity_models[f"MVZ_{val_str}"] = d / "flow_5min_cleaned_mvz.csv"
                else:
                    # For example, sensitivity_analysis_0.05 becomes Leakage_0.05.
                    model_name = f"Leakage_{val_str}"
                    csv_path = d / "flow_5min_cleaned.csv"
                    if csv_path.exists():
                        sensitivity_models[model_name] = csv_path

    models = {
        "Raw(Imputed)": imputed_data_path,
        "IsolationForest": root_dir / 'BaseLine/IsolationForest/cleaning_results/flow_5min_cleaned_if.csv',
        "Multi-variate_Z-score": root_dir / 'BaseLine/Multi-variate_Z-score/cleaning_results/flow_5min_cleaned_mvz.csv',
        "GAT-only": root_dir / 'BaseLine/GAT-only/cleaning_results/flow_5min_cleaned_gat.csv',
        "PI-GAT_without_bias": root_dir / 'ablation_study/flow_5min_cleaned_without_bias.csv',
        "PI-GAT": root_dir / 'data_clean/outputs/flow_5min_cleaned.csv',
    }
    # Add sensitivity runs to the comparison.
    models.update(sensitivity_models)

    # Name the trunk and branch flow columns.
    branches = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06']
    total_name = 'Total'
    
    print("[1] Loading system parameters...")
    # Read the leakage fraction.
    lam = load_lambda_from_ckpt(ckpt_path) if ckpt_path.exists() else 0.10
    
    # Read the PI-GAT offsets.
    pi_gat_biases = load_bias_params(bias_file_path, branches)
    
    # Read the effective-opening mapping.
    with open(eom_file_path, 'r', encoding='utf-8') as f:
        eom_data = json.load(f)

    # Load the full imputed series to preserve the original row positions referenced by t_index.
    df_imputed_full = pd.read_csv(imputed_data_path, index_col='date', parse_dates=True)

    # Exclude all-zero and incomplete rows from the active mask.
    print("[*] Building Active Mask (excluding all-zero and missing-value windows)...")
    active_mask = build_active_mask(df_imputed_full)
    active_count = int(active_mask.sum())
    print(f"    --> Active windows: {active_count} / {len(df_imputed_full)} ({active_count / len(df_imputed_full):.1%})")

    # Restrict the evaluation to active windows.
    df_imputed = df_imputed_full[active_mask]
    total_points = len(df_imputed)
    df_mean = df_imputed[branches].mean()  # Compute the denominators for the modification rates.

    # Identify high-credibility windows.
    df_cred = pd.read_csv(cred_file_path)
    # Use t_index as a positional index into the full imputed series.
    high_cred_indices = df_cred[df_cred['credibility'] == 'high']['t_index'].values

    # Use the high-credibility mass-balance errors to calibrate the MCE threshold.
    print("[2] Computing Adaptive MCE Threshold...")
    # Compute the baseline MCE with zero offset on the complete timeline before selecting high-credibility rows.
    sum_branches_imputed_full = df_imputed_full[branches].sum(axis=1)
    imputed_mce_full = (df_imputed_full[total_name] * (1 - lam) - sum_branches_imputed_full).abs()

    # Set the MCE threshold to three standard deviations of the high-credibility errors.
    # Ensure t_index values are within bounds
    valid_indices = np.array([idx for idx in high_cred_indices if idx < len(imputed_mce_full)], dtype=int)
    high_cred_mce = imputed_mce_full.iloc[valid_indices]
    threshold_mce = 3 * high_cred_mce.std()
    print(f"    --> Threshold MCE (3-sigma): {threshold_mce:.4f}")

    results = []

    print("[3] Evaluating Models...")
    for model_name, path in models.items():
        if not path.exists():
            print(f"Skipping {model_name}, file not found: {path}")
            continue
            
        df = pd.read_csv(path, index_col='date', parse_dates=True)
        # Align each model to the full date index before applying the active mask.
        df = df.reindex(df_imputed_full.index)[active_mask]
        
        # 1. Data recovery rate based on the proportion of non-missing readings.
        recovery_rate = df[branches + [total_name]].notna().mean().mean()
        
        # 2. Modification rate for readings that differ from the baseline by more than 2% of the mean.
        mod_counts = 0
        valid_comp_points = 0
        for b in branches:
            diff = (df[b] - df_imputed[b]).abs()
            thres = 0.02 * df_mean[b]
            mod_counts += (diff > thres).sum()
            valid_comp_points += diff.notna().sum()
        mod_ratio = mod_counts / max(valid_comp_points, 1)

        # 3. Calculate MCE on complete flow rows.
        # Exclude a row from MCE if any required flow reading is missing for this method.
        df_clean_mce = df[[total_name] + branches].dropna()
        if len(df_clean_mce) == 0:
            avg_mce = np.nan
            violation_rate = np.nan
        else:
            q_tot = df_clean_mce[total_name]
            
            # Use the leakage fraction and offsets for this model.
            current_lam = lam
            bias_dict = {b: 0.0 for b in branches}
            
            if model_name == "PI-GAT" or model_name.startswith("PI-GAT_"):
                bias_dict = pi_gat_biases
            elif model_name.startswith("Leakage_"):
                # Read parameters for the current sensitivity analysis when available.
                sens_dir = path.parent
                sens_bias = sens_dir / "calibration_params_final.json"
                if sens_bias.exists():
                    bias_dict = load_bias_params(sens_bias, branches)
                    # Prefer the JSON for fixed-leakage analyses because it contains the exported fraction.
                    with open(sens_bias, 'r', encoding='utf-8') as f:
                        cal_data = json.load(f)
                        if "leakage" in cal_data and "rate" in cal_data["leakage"]:
                            current_lam = cal_data["leakage"]["rate"]
            
            sum_b = pd.Series(0, index=df_clean_mce.index)
            for b in branches:
                sum_b += (df_clean_mce[b] - bias_dict.get(b, 0.0))
                
            model_mce = np.abs(q_tot * (1 - current_lam) - sum_b)
            avg_mce = model_mce.mean()
            violation_rate = (model_mce > threshold_mce).mean()
        
        # Count rows that exceed the mass-balance error threshold.
        physically_valid = (avg_mce < threshold_mce) if pd.notna(avg_mce) else False

        # 4. Measure effective-opening consistency with Pearson correlation.
        # Compare the cleaned flow series with the fitted effective-opening series.
        corr_list = []
        for b in branches:
            if f"OD_{b}" in df.columns:
                od_col = df[f"OD_{b}"]
            elif f"OD_{b}" in df_imputed.columns:
                od_col = df_imputed[f"OD_{b}"]
            else:
                continue
            
            od_eff = get_eom_effective_opening(od_col, b, eom_data)
            
            # Use complete observation pairs for correlation.
            q_clean = np.asarray(df[b].values, dtype=float)
            od_eff_arr = np.asarray(od_eff, dtype=float)
            valid_mask = ~np.isnan(q_clean) & ~np.isnan(od_eff_arr)
            
            if valid_mask.sum() > 1:
                valid_q = q_clean[valid_mask]
                valid_od = od_eff_arr[valid_mask]
                # Correlation is undefined for a constant series.
                if np.std(valid_q) > 1e-6 and np.std(valid_od) > 1e-6:
                    corr, _ = pearsonr(valid_q, valid_od)
                    corr_list.append(corr)
            
        eom_corr_mean = np.nanmean(corr_list) if corr_list else np.nan

        results.append({
            "Model": model_name,
            "Recovery Rate (%)": recovery_rate * 100,
            "Mod_Ratio (%)": mod_ratio * 100,
            "Avg-MCE": avg_mce,
            "MCE Violation Rate (%)": violation_rate * 100,
            "Is Physically Valid": "Yes" if physically_valid else "No",
            "EOM-Flow Correlation (Pearson)": eom_corr_mean
        })

    # Print the summary table.
    print("\n[4] Physical Consistency Evaluation Summary:")
    res_df = pd.DataFrame(results)
    print(res_df.to_markdown(index=False))
    
    # Save the evaluation report.
    out_file = root_dir / 'BaseLine/evaluation_metrics.csv'
    res_df.to_csv(out_file, index=False)
    print(f"\nReport saved to: {out_file}")

if __name__ == "__main__":
    main()
