import pandas as pd
import numpy as np
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"
BASELINE_DIR = ROOT_DIR / "Baseline"
PI_GAT_DIR = ROOT_DIR / "PI-GAT"
DATA_CLEAN_DIR = ROOT_DIR / "data_clean"

def main():
    with open(SCRIPT_DIR / "kcr_results.json", "r") as f:
        kcr_data = json.load(f)
    
    start_idx = kcr_data["window"]["start"]
    end_idx = kcr_data["window"]["end"]
    lam = 0.10 # Default from find_real_anomaly.py fallback if needed, but we don't strictly need it for branches
    zoom_size = 200 # as requested: 200-500
    
    raw_csv   = PREPROCESS_DIR / "flow_5min_imputed.csv"
    pigat_csv = DATA_CLEAN_DIR / "outputs" / "flow_5min_cleaned.csv"
    if_csv    = BASELINE_DIR / "IsolationForest" / "cleaning_results" / "flow_5min_cleaned_if.csv"
    mvz_csv   = BASELINE_DIR / "Multi-variate_Z-score" / "cleaning_results" / "flow_5min_cleaned_mvz.csv"

    df_raw   = pd.read_csv(str(raw_csv))
    df_pigat = pd.read_csv(str(pigat_csv))
    df_if    = pd.read_csv(str(if_csv))
    df_mvz   = pd.read_csv(str(mvz_csv))
    
    sensors = ["B01", "B02", "B03", "B04", "B05", "B06"]
    
    # Let's find highly volatile sub-window for Case B/C
    pigat_sum_win = df_pigat[sensors].iloc[start_idx:end_idx].sum(axis=1)
    s_reset = pigat_sum_win.reset_index(drop=True)
    roll_std = s_reset.rolling(zoom_size).std()
    best_end_pos = int(roll_std.idxmax())
    best_start_pos = max(0, best_end_pos - zoom_size + 1)
    
    zoom_start = start_idx + best_start_pos
    zoom_end = zoom_start + zoom_size
    time_axis = np.arange(zoom_start, zoom_end)
    
    # ==========================
    # Case B: Active branch
    # Let's pick B03 or B04, basically the one with max OD > 60%
    # ==========================
    active_sensor = "B04" # Default
    for s in sensors:
        if s != "B05" and df_raw[f"OD_{s}"].iloc[zoom_start:zoom_end].max() > 50:
            active_sensor = s
            break
            
    df_case_b = pd.DataFrame({
        "Time": time_axis,
        f"OD_{active_sensor}": df_raw[f"OD_{active_sensor}"].iloc[zoom_start:zoom_end].values,
        "Raw": df_raw[active_sensor].iloc[zoom_start:zoom_end].values,
        "IF": df_if[active_sensor].iloc[zoom_start:zoom_end].values,
        "MVZ": df_mvz[active_sensor].iloc[zoom_start:zoom_end].values,
        "PI_GAT": df_pigat[active_sensor].iloc[zoom_start:zoom_end].values
    })
    df_case_b.to_csv(SCRIPT_DIR / "case_b_active_branch.csv", index=False)
    
    # ==========================
    # Case C: B05 (The Crime Scene)
    # ==========================
    # sum of active ODs excluding B05
    active_od_cols = [f"OD_{s}" for s in sensors if s != "B05"]
    od_active_sum = df_raw[active_od_cols].iloc[zoom_start:zoom_end].sum(axis=1).values
    
    df_case_c = pd.DataFrame({
        "Time": time_axis,
        "Total_Flow": df_raw["Total"].iloc[zoom_start:zoom_end].values,
        "OD_B05": df_raw["OD_B05"].iloc[zoom_start:zoom_end].values,
        "OD_Active": od_active_sum,
        "Raw_B05": df_raw["B05"].iloc[zoom_start:zoom_end].values,
        "IF_B05": df_if["B05"].iloc[zoom_start:zoom_end].values,
        "MVZ_B05": df_mvz["B05"].iloc[zoom_start:zoom_end].values,
        "PI_GAT_B05": df_pigat["B05"].iloc[zoom_start:zoom_end].values
    })
    df_case_c.to_csv(SCRIPT_DIR / "case_c_B05_crime_scene.csv", index=False)
    print(f"Generated case_b_active_branch.csv (using {active_sensor}) and case_c_B05_crime_scene.csv")

if __name__ == "__main__":
    main()
