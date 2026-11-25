import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------   
# Project-relative paths.
# ---------------------------------------------------------------------------   
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
PI_GAT_DIR = ROOT_DIR / "PI-GAT"
PREPROCESS_DIR = ROOT_DIR / "Preprocess_Timeseries" / "outputs" / "preprocessed"

sys.path.insert(0, str(PI_GAT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from graph_utils import build_graph_from_csv          # type: ignore
from model_gat import GAT                             # type: ignore
from timeseries_dataset import IrrigationTimeWindowDataset  # type: ignore      

# ---------------------------------------------------------------------------
# Shared constants aligned with PI-GAT data_cleaning.py.
# ---------------------------------------------------------------------------
# Global hydraulic deadband, approximately 4.3% of the typical trunk mean of 1862 m³/h.
# This floor represents trunk pressure fluctuations that can influence even low-flow branches.
GLOBAL_HYDRAULIC_DEADBAND = 80.0  # m³/h

# Sensor column names.
SENSOR_COLS = ["B01", "B02", "B03", "B04", "B05", "B06"]
TOTAL_COL = "Total"


def main():
    parser = argparse.ArgumentParser(description="Pure GAT 基线模型的数据清洗脚手架")
    parser.add_argument("--ckpt", type=str, default=str(SCRIPT_DIR / "gat_final.pth"), help="GAT 模型权重路径")
    parser.add_argument("--min_abs_thresh", type=float, default=GLOBAL_HYDRAULIC_DEADBAND, help="异常判定残差绝对下限 (m3/h)，默认为项目统一全局水力本底死区")
    parser.add_argument("--k_spike", type=float, default=3.0, help="异常判定的 sigma 倍数")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WINDOW = 12
    BATCH_SIZE = 64
    
    # Input and output paths.
    TOPO_CSV = ROOT_DIR / "network_topology.csv"
    ZSCORE_CSV = PREPROCESS_DIR / "flow_5min_zscore.csv"
    RESTMASK_CSV = PREPROCESS_DIR / "mask_rest.csv"
    ZSCORE_STATS = PREPROCESS_DIR / "zscore_stats.json"
    OPEN_CSV = PREPROCESS_DIR / "flow_5min_imputed.csv"  # Contains all imputed flow and recorded valve openings.
    CREDIBILITY_CSV = PI_GAT_DIR / "preprocessing" / "credibility_windows.csv"

    OUTPUT_DIR = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / "cleaning_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CLEANED_CSV = OUTPUT_DIR / "flow_5min_cleaned_gat.csv"
    OUTPUT_METRICS_JSON = OUTPUT_DIR / "cleaning_metrics.json"

    # 1. Load the data and build the graph and model.
    print("=== [1/5] 构建基础环境与模型 ===")
    data_graph, node2idx, pipe2idx, sensor2edge_idx = build_graph_from_csv(
        TOPO_CSV, weight_mode="conductance", add_self_loops=True
    )
    sensor2node_idx = data_graph.sensor2node_idx
    
    # Read the flow statistics to reverse standardisation.
    with open(ZSCORE_STATS, "r", encoding="utf-8") as f:
        stats = json.load(f)
        
    df_raw = pd.read_csv(OPEN_CSV)
    n_total = len(df_raw)

    # Select valid and active timesteps consistently with PI-GAT.
    is_active_row = pd.Series(False, index=df_raw.index)
    if CREDIBILITY_CSV.exists():
        _cred_tmp = pd.read_csv(CREDIBILITY_CSV, usecols=["t_index"])
        _active_arr = _cred_tmp["t_index"].astype(int).to_numpy()
        _in_range = (_active_arr >= 0) & (_active_arr < n_total)
        is_active_row.iloc[_active_arr[_in_range]] = True
    
    # Build the dataset without additional masking.
    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=ZSCORE_CSV,
        restmask_path=RESTMASK_CSV,
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WINDOW,
        random_mask_range=None,  # Apply no extra mask at inference.
        open_paths=OPEN_CSV,
        zscore_stats_path=ZSCORE_STATS,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None)
    )
    
    loader = DataLoader(
        cast(Any, dataset), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    N_nodes = int(data_graph.num_nodes) if data_graph.num_nodes is not None else 0
    N_edges = data_graph.edge_index.shape[1] if data_graph.edge_index is not None else 0
    
    model = GAT(
        win=WINDOW,
        num_nodes=N_nodes,
        num_edges=N_edges,
        hidden=64,
        heads=8,
        dropout=0.1,
        in_feat=2
    ).to(DEVICE)
    
    print(f"=== [2/5] 加载模型权重: {args.ckpt} ===")
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    # Accept a state dictionary or a checkpoint wrapping it under "model_state_dict".
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    # Run inference and collect predictions.
    print("=== [3/5] 执行纯数据驱动推理... ===")
    
    # Index the prediction buffer by valid timestep.
    pred_z_dict = {}  # {time_idx: {node_idx: pred_val}}
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            
            # node_hat: [B*N]
            node_hat = out["node_hat"].view(-1, N_nodes)
            t_idxs = batch.t_index
            
            for i, t_idx in enumerate(t_idxs):
                t = int(t_idx.item())
                pred_z_dict[t] = node_hat[i].cpu().numpy()

    print("=== [4/5] 基于自适应残差的异常判定与替换 ===")

    # A. Compute residuals on valid predictions and estimate the adaptive thresholds.
    node_thresholds = {}
    
    for sensor in SENSOR_COLS:
        if sensor not in df_raw.columns:
            continue
        n_idx = sensor2node_idx.get(sensor, None)
        if n_idx is None:
            continue
            
        mu = stats["mean"][sensor]
        sigma = stats["std"][sensor]
        
        valid_residuals = []
        for t in range(n_total):
            if not is_active_row.iloc[t]:
                continue
            if t in pred_z_dict:
                pred_z = pred_z_dict[t][n_idx]
                pred_val = pred_z * sigma + mu
                
                raw_val_item = df_raw.at[t, sensor]
                raw_val = float(raw_val_item) if not pd.isna(raw_val_item) else float('nan') # type: ignore
                
                if not pd.isna(raw_val) and not pd.isna(pred_val):
                    valid_residuals.append(raw_val - pred_val)
                    
        # Adaptive threshold: Thr_i = max(mean(|R - mu_R|) + 3*sigma_R, GLOBAL_HYDRAULIC_DEADBAND).
        if len(valid_residuals) > 0:
            arr_R = np.array(valid_residuals)
            mu_R_val = np.mean(arr_R)
            abs_excess = np.abs(arr_R - mu_R_val)
            thr = np.mean(abs_excess) + args.k_spike * np.std(abs_excess)
            node_thresholds[sensor] = max(thr, args.min_abs_thresh)
        else:
            node_thresholds[sensor] = args.min_abs_thresh
            
    print(f" - 计算得到的各节点自适应阈值: \n" + json.dumps({k: round(v, 2) for k, v in node_thresholds.items()}, indent=2))
    
    df_cleaned = df_raw.copy()
    
    total_anomalies_fixed = 0
    total_valid_points = 0
    
    # Initialise the arrays for the evaluation summaries.
    sum_abs_err_before = 0.0
    sum_abs_err_after = 0.0
    
    # B. Apply the cleaning rule.
    for sensor in SENSOR_COLS:
        if sensor not in df_raw.columns:
            continue
            
        n_idx = sensor2node_idx.get(sensor, None)
        if n_idx is None:
            continue
            
        mu = stats["mean"][sensor]
        sigma = stats["std"][sensor]
        thr_i = node_thresholds.get(sensor, args.min_abs_thresh)
        
        cleans_for_sensor = []
        
        # Scan all data rows.
        for t in range(n_total):
            raw_val_item = df_raw.at[t, sensor]
            raw_val = float(raw_val_item) if not pd.isna(raw_val_item) else float('nan') # type: ignore
            
            clean_val = raw_val
            
            # Preserve the observed value on inactive rows, including missing and shut-down periods.
            if not is_active_row.iloc[t]:
                cleans_for_sensor.append(clean_val)
                continue
                
            # Fetch the prediction.
            # A prediction is unavailable before WINDOW-1 or when the timestep is absent from valid_indices.
            if t in pred_z_dict:
                pred_z = pred_z_dict[t][n_idx]
                pred_val = pred_z * sigma + mu
            else:
                # Retain the observed reading where the prediction is missing.
                pred_val = raw_val
            
            if not pd.isna(raw_val) and not pd.isna(pred_val):
                residual = raw_val - pred_val
                abs_err = abs(residual)
                
                total_valid_points += 1
                sum_abs_err_before += abs_err
                
                # Replace the observation with the GAT prediction if its absolute residual exceeds the adaptive threshold.
                if abs_err > thr_i:
                    clean_val = pred_val
                    total_anomalies_fixed += 1
                    
                sum_abs_err_after += abs(clean_val - pred_val)
                
            cleans_for_sensor.append(clean_val)
            
        df_cleaned[sensor] = cleans_for_sensor
        
    print(f"=== [5/5] 收尾统计与保存 ===")
    
    mae_before = sum_abs_err_before / max(1, total_valid_points)
    mae_after = sum_abs_err_after / max(1, total_valid_points)
    
    print(f" - 总计检测到有效数据点: {total_valid_points}")
    print(f" - 发现并修正异常值点数: {total_anomalies_fixed}")
    print(f" - 清洗前全局 MCE (这里等价于 MAE vs Pred): {mae_before:.4f}")
    print(f" - 清洗后全局 MCE (这里等价于 MAE vs Pred): {mae_after:.4f}")
    
    df_cleaned.to_csv(OUTPUT_CLEANED_CSV, index=False)
    
    # Save the run summary alongside the cleaned series.
    metrics_dict = {
        "total_valid_points": total_valid_points,
        "total_anomalies_fixed": total_anomalies_fixed,
        "mae_before": mae_before,
        "mae_after": mae_after,
        "min_abs_thresh": args.min_abs_thresh,
        "k_spike": args.k_spike,
        "node_thresholds": node_thresholds
    }
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f" - 清洗后数据已保存至: {OUTPUT_CLEANED_CSV}")
    print(f" - 运行指标已保存至: {OUTPUT_METRICS_JSON}")
    print("纯 GAT 基线清洗流程执行完毕！\n"
          "注：此版本移除了所有的物理守恒控制（质量/能量）、EOM 映射、仪表偏差校准处理。")

if __name__ == "__main__":
    main()