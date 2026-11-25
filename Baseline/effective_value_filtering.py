"""Build the active mask for the five-minute imputed flow series.

The two exclusion rules match PI-GAT/preprocessing/preprocess_imputed.py.
Exclude a row if any required reading is missing or if all required readings
are zero. The reference series has 26,472 active timesteps.

Import build_active_mask and REQUIRED_COLUMNS for use in another script.
Run this module directly for a summary and to write active_mask.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Use the same required columns as preprocess_imputed.py.
REQUIRED_COLUMNS = ["Total", "B01", "B02", "B03", "B04", "B05", "B06"]

# Expected active timestep count for the reference records.
EXPECTED_ACTIVE_COUNT = 26472


def build_active_mask(df: pd.DataFrame, required_columns: Optional[list] = None) -> pd.Series:
    """Mark complete, nonzero network observations as active.

    The input is normally the imputed flow dataframe indexed by date.
    Pass required_columns to override REQUIRED_COLUMNS. The returned Boolean
    series aligns with the input index. True marks an eligible timestep.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 中缺少以下必需列: {missing_cols}")

    # Rule 1. Mark rows with any missing required reading as inactive.
    has_nan = df[required_columns].isna().any(axis=1)

    # Rule 2. Mark rows with all required readings equal to zero as inactive.
    all_zero = (df[required_columns] == 0.0).all(axis=1)

    active = ~has_nan & ~all_zero
    active.name = "active"
    return active


def main() -> pd.Series:
    root_dir = Path(__file__).parent.parent
    imputed_path = (
        root_dir
        / "Preprocess_Timeseries"
        / "outputs"
        / "preprocessed"
        / "flow_5min_imputed.csv"
    )
    out_mask_csv = Path(__file__).parent / "active_mask.csv"

    if not imputed_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {imputed_path}")

    print(f"正在读取: {imputed_path}")
    df = pd.read_csv(imputed_path, index_col="date", parse_dates=True)
    print(f"总时间步: {len(df)}")

    active_mask = build_active_mask(df)

    # Count the exclusions by rule.
    total = len(df)
    nan_flag  = df[REQUIRED_COLUMNS].isna().any(axis=1)
    zero_flag = (df[REQUIRED_COLUMNS] == 0.0).all(axis=1)

    only_nan  = nan_flag  & ~zero_flag
    only_zero = zero_flag & ~nan_flag
    both_flag = nan_flag  &  zero_flag
    inactive  = ~active_mask
    active_count = int(active_mask.sum())

    print(f"\n过滤统计:")
    print(f"  含缺失值窗口 (仅):          {int(only_nan.sum())}")
    print(f"  全零值窗口   (仅):          {int(only_zero.sum())}")
    print(f"  同时满足两条规则:           {int(both_flag.sum())}")
    print(f"  总排除窗口:                 {int(inactive.sum())}")
    print(f"  活跃窗口数量 (Active):      {active_count}")
    print(f"  活跃率:                     {active_count / total:.2%}")

    # Check the active timestep count against the reference count.
    if active_count != EXPECTED_ACTIVE_COUNT:
        print(
            f"\n[警告] 活跃窗口数量 {active_count} 与预期 {EXPECTED_ACTIVE_COUNT} 不符，"
            f"请检查输入数据或过滤规则！"
        )
    else:
        print(f"\n[校验通过] 活跃窗口数量 = {active_count}，符合预期。")

    # Save the mask.
    active_mask.to_frame().to_csv(out_mask_csv, encoding="utf-8")
    print(f"\n活跃掩码已保存至: {out_mask_csv}")

    return active_mask


if __name__ == "__main__":
    main()
