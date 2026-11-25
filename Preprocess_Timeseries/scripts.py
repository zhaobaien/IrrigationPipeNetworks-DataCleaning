import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from data_loading import (
    read_flow_csv,
    FlowratePreprocessor,
    TimePreprocessConfig,
    save_preprocess_outputs,
    quick_report,
)

DEFAULT_THRESHOLDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "OD_B01":    (0.0, 100.0),
    "OD_B02":  (0.0, 100.0),
    "OD_B03":    (0.0, 100.0),
    "OD_B04":    (0.0, 100.0),
    "OD_B05":    (0.0, 100.0),
    "OD_B06":    (0.0, 100.0),
    "OD_Total": (0.0, 100.0),
    "B01":    (0.0, 1600.0),
    "B02":  (0.0, 1600.0),
    "B03":    (0.0, 1600.0),
    "B04":    (0.0, 1600.0),
    "B05":    (0.0, 1600.0),
    "B06":    (0.0, 1600.0),
    "Total": (0.0, 4000.0),
    "P_B01":    (0.0, 1.0),
    "P_B02":  (0.0, 1.0),
    "P_B03":    (0.0, 1.0),
    "P_B04":    (0.0, 1.0),
    "P_B05":    (0.0, 1.0),
    "P_B06":    (0.0, 1.0),
    "P_Total": (0.0, 1.0),
}


def main():
    """Preprocess the authorised telemetry restored at the repository root."""
    # The input can include pressure channels.
    csv_path = PROJECT_ROOT / "PipelineDataset-F.csv"
    out_dir = SCRIPT_DIR / "outputs" / "preprocessed"

    print(f"Reading from: {csv_path}")

    try:
        df = read_flow_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return

    cfg = TimePreprocessConfig(
        target_freq="5min",
        short_gap_limit_minutes=10,
        long_gap_as_rest_minutes=24*60,  # Identify prolonged missing runs as rest periods.
        zero_is_valid=True,
        clip_negative_to_zero=True,      # Clip negative readings to zero.
        apply_zscore=True,
        keep_stats_for_inverse=True      # Retain statistics for inverse standardisation.
    )

    pre = FlowratePreprocessor(cfg)

    cols_order = [
        "OD_B01", "B01", "P_B01",
        "OD_B02", "B02", "P_B02",
        "OD_B03", "B03", "P_B03",
        "OD_B04", "B04", "P_B04",
        "OD_B05", "B05", "P_B05",
        "OD_B06", "B06", "P_B06",
        "OD_Total", "Total", "P_Total",
    ]

    out = pre.transform(
        df,
        thresholds=DEFAULT_THRESHOLDS,
        columns_order=cols_order,
    )

    print(quick_report(out))
    save_preprocess_outputs(out_dir, out)
    print(f"Saved results to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
