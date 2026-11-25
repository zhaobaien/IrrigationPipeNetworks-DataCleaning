import os
import sys
import subprocess
from pathlib import Path

def main():
    # Resolve the existing Stage I, II and III entry points.
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    pi_gat_dir = root_dir / "PI-GAT"
    bias_script = pi_gat_dir / "bias_calibration" / "instrument_bias.py"
    clean_script = root_dir / "data_clean" / "data_cleaning.py"
    train_script = pi_gat_dir / "train_pi_gat.py"
    
    # Fixed leakage fractions evaluated in the sensitivity sweep.
    leakage_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    # Use the interpreter from the current environment.
    python_exe = sys.executable
    
    for val in leakage_values:
        print("="*60)
        print(f"Starting Sensitivity Analysis for Leakage = {val}")
        print("="*60)
        
        # Prepare one output directory per leakage fraction.
        # For example, sensitivity_analysis/sensitivity_analysis_0.05.
        out_folder = script_dir / f"sensitivity_analysis_{val}"
        out_folder.mkdir(parents=True, exist_ok=True)
        
        # Record the artefact paths used by each stage.
        stage1_ckpt = out_folder / "pi_gat_best.pth"
        bias_json = out_folder / "calibration_params_final.json"
        stage2_ckpt = out_folder / "instrument_bias_best.pth"
        cleaned_csv = out_folder / "flow_5min_cleaned.csv"
        
        # Train Stage I with the leakage fraction held fixed.
        if not stage1_ckpt.exists():
            print(f"\n[Stage I: Training Model] val = {val}")
            cmd_train = [
                python_exe,
                str(train_script),
                "--fixed-leakage", str(val),
                "--out-dir", str(out_folder)
            ]
            subprocess.run(cmd_train, check=True)
        else:
            print(f"\n[Stage I] Skip training, checkpoint exists: {stage1_ckpt}")
            
        # Fit Stage-II scales using the checkpoint for this leakage setting.
        if not bias_json.exists() or not stage2_ckpt.exists():
            print(f"\n[Stage II: Bias Calibration] val = {val}")
            cmd_bias = [
                python_exe,
                str(bias_script),
                "--ckpt", str(stage1_ckpt),
                "--fixed-leakage", str(val),
                "--out-json", str(bias_json),
                "--out-ckpt", str(stage2_ckpt)
            ]
            subprocess.run(cmd_bias, check=True)
        else:
            print(f"\n[Stage II] Skip bias calibration, outputs exist.")
            
        # Apply Stage III using the matching checkpoint and scale calibration.
        if not cleaned_csv.exists():
            print(f"\n[Stage III: Data Cleaning] val = {val}")
            cmd_clean = [
                python_exe,
                str(clean_script),
                "--ckpt", str(stage1_ckpt),     # Stage-I checkpoint for this run.
                "--calib-json", str(bias_json), # Stage-II scales for this leakage fraction.
                "--out-dir", str(out_folder)
            ]
            subprocess.run(cmd_clean, check=True)
        else:
            print(f"\n[Stage III] Skip data cleaning, output exists {cleaned_csv}")
            
        print(f"✓ Completed Leakage={val}\n\n")

if __name__ == "__main__":
    main()
