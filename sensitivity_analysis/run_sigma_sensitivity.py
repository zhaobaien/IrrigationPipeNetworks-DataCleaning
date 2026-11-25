import os
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent.resolve()
    sensitivity_dir = root_dir / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    
    # Residual multipliers used in the sweep.
    sigma_gradients = [2.0, 3.0, 4.0]
    
    scripts = [
        {
            "name": "PI-GAT",
            "path": root_dir / "data_clean" / "data_cleaning.py",
            "args": lambda out_dir, k: ["--out-dir", str(out_dir), "--k-spike", str(k)]
        },
        {
            "name": "GAT-only",
            "path": root_dir / "Baseline" / "GAT-only" / "data_cleaning_gat.py",
            "args": lambda out_dir, k: ["--out_dir", str(out_dir), "--k_spike", str(k)]
        },
        {
            "name": "IsolationForest",
            "path": root_dir / "Baseline" / "IsolationForest" / "run_isolation_forest.py",
            "args": lambda out_dir, k: ["--out-dir", str(out_dir), "--k-sigma", str(k)]
        },
        {
            "name": "MV-Zscore",
            "path": root_dir / "Baseline" / "Multi-variate_Z-score" / "run_multivariate_zscore.py",
            "args": lambda out_dir, k: ["--out-dir", str(out_dir), "--k-sigma", str(k)]
        }
    ]
    
    import sys
    python_exe = sys.executable
    
    for k in sigma_gradients:
        out_folder = sensitivity_dir / f"sensitivity_analysis_{int(k)}-Sigma"
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=======================================================")
        print(f" Starting Sigma Sensitivity Analysis for {k}-Sigma ")
        print(f" Output Directory: {out_folder}")
        print(f"=======================================================\n")
        
        for script in scripts:
            print(f"--- Running {script['name']} ---")
            script_path = script['path']
            if not script_path.exists():
                print(f"[Error] Script not found: {script_path}")
                continue
                
            cmd = [python_exe, str(script_path)] + script['args'](out_folder, k)
            print(f"Executing: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Execution failed for {script['name']} with code {e.returncode}")
                # Continue with the remaining methods after reporting the failure.
                
    print("\n[+] All Sigma sensitivity tasks completed.")

if __name__ == "__main__":
    main()
