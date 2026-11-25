from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
import json
from pathlib import Path


@dataclass
class TimePreprocessConfig:
    """Configuration for the raw-flow preprocessing pipeline."""
    target_freq: str = "5min"
    short_gap_limit_minutes: int = 0
    long_gap_as_rest_minutes: int = 24 * 60 # Identify prolonged missing runs as rest periods.
    zero_is_valid: bool = True
    clip_negative_to_zero: bool = True
    apply_zscore: bool = True
    keep_stats_for_inverse: bool = True # Retain statistics for inverse standardisation.


COLUMNS_RENAME = {
    "open_degree_B01": "OD_B01",
    "open_degree_B02": "OD_B02",
    "open_degree_B03": "OD_B03",
    "open_degree_B04": "OD_B04",
    "open_degree_B05": "OD_B05",
    "open_degree_B06": "OD_B06",
    "open_degree_Total": "OD_Total",
    "flow(m^3/h)_B01": "B01",
    "flow(m^3/h)_B02": "B02",
    "flow(m^3/h)_B03": "B03",
    "flow(m^3/h)_B04": "B04",
    "flow(m^3/h)_B05": "B05",
    "flow(m^3/h)_B06": "B06",
    "flow(m^3/h)_Total": "Total",
    "pressure(MPa)_B01": "P_B01",
    "pressure(MPa)_B02": "P_B02",
    "pressure(MPa)_B03": "P_B03",
    "pressure(MPa)_B04": "P_B04",
    "pressure(MPa)_B05": "P_B05",
    "pressure(MPa)_B06": "P_B06",
    "pressure(MPa)_Total": "P_Total",
}


def read_flow_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read the flow CSV and map source channels to the standard column names."""
    path = Path(path)
    df = pd.read_csv(path)

    if "date" not in df.columns and "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    if "date" not in df.columns:
        raise ValueError("The CSV must contain a 'date' column.")

    rename_map = {c: COLUMNS_RENAME.get(c, c) for c in df.columns}
    df = df.rename(columns=rename_map)

    wanted_cols = [
        "date",
        "OD_B01", "OD_B02", "OD_B03", "OD_B04", "OD_B05", "OD_B06", "OD_Total",
        "B01", "B02", "B03", "B04", "B05", "B06", "Total",
        "P_B01", "P_B02", "P_B03", "P_B04", "P_B05", "P_B06", "P_Total",
    ]
    existing_cols = [c for c in wanted_cols if c in df.columns]

    core_cols = ["B01", "B02", "B03", "B04", "B05", "B06", "Total"]
    missing_core = [c for c in core_cols if c not in existing_cols]
    if missing_core:
        print(f"Warning: Some flow columns may be missing: {missing_core}")

    df = df[existing_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    df = (
        df.set_index("date")
          .groupby("date", as_index=True)
          .mean(numeric_only=True)
          .sort_index()
    )
    return df


def _apply_min_max_validation(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
    zero_is_valid: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply channel bounds and return a mask for invalid readings."""
    df = df.copy()
    invalid_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if thresholds is None:
        return df, invalid_mask

    for col in df.columns:
        cmin, cmax = thresholds.get(col, (None, None))
        is_zero = df[col].eq(0.0)

        if cmin is not None:
            invalid_mask[col] |= df[col] < cmin
        if cmax is not None:
            invalid_mask[col] |= df[col] > cmax

        df.loc[invalid_mask[col], col] = np.nan

        if zero_is_valid:
            recover_mask = is_zero & invalid_mask[col]
            df.loc[recover_mask, col] = 0.0
            invalid_mask.loc[recover_mask, col] = False

    return df, invalid_mask

def _resample_to_5min_mean(data: pd.DataFrame, target_freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average records in bins of the target frequency and identify empty bins."""
    df_5min = data.resample(target_freq).mean()
    resample_gap_mask = pd.DataFrame(False, index=df_5min.index, columns=df_5min.columns)
    return df_5min, resample_gap_mask

def _run_length_encode(arr: np.ndarray):
    """Return values and lengths for consecutive runs in an array."""
    n = len(arr)
    if n == 0: return []

    y = np.array(arr[1:] != arr[:-1])
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(z) - z
    return zip(arr[p], z)

# Identify prolonged missing runs as rest periods.
def _flag_long_gaps_as_rest(nan_mask: pd.DataFrame,
                            target_freq: str,
                            long_gap_minutes: int) -> pd.DataFrame:
    """Flag each missing run that meets the channel-level rest-period threshold."""
    freq_min = pd.to_timedelta(target_freq).total_seconds() / 60.0
    steps = max(1, int(long_gap_minutes / freq_min))

    rest_mask = pd.DataFrame(False, index=nan_mask.index, columns=nan_mask.columns)

    for col in nan_mask.columns:
        col_values = np.asarray(nan_mask[col].values, dtype=bool)
        flags = np.zeros(len(nan_mask), dtype=bool)
        idx = 0
        for val, length in _run_length_encode(col_values):
            if val and length >= steps:
                flags[idx:idx+length] = True
            idx += length
        rest_mask[col] = flags

    return rest_mask

def _interpolate_short_gaps(
    df: pd.DataFrame,
    target_freq: str,
    short_gap_limit_minutes: int,
    resample_gap_mask: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Interpolate the first eligible cells of internal gaps and retain quality masks."""
    if short_gap_limit_minutes <= 0:
        imputed_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        original_missing_mask = df.isna()
        return df.copy(), imputed_mask, original_missing_mask

    max_gap = int(short_gap_limit_minutes // (pd.to_timedelta(target_freq).total_seconds() / 60.0))

    df_imputed = df.copy()
    df_imputed = df_imputed.interpolate(method="linear", limit=max_gap, limit_area="inside")

    imputed_mask = (df.isna()) & (~df_imputed.isna())
    original_missing_mask = df.isna() & (~imputed_mask)

    return df_imputed, imputed_mask, original_missing_mask

def _zscore(df: pd.DataFrame):
    """Standardise each channel and retain its mean and standard deviation."""
    mean = df.mean(skipna=True)
    std = df.std(skipna=True).replace(0, np.nan)
    std = std.fillna(1.0)
    z = (df - mean) / std
    return z, mean, std


class FlowratePreprocessor:
    """Preprocess irregular flow, valve-opening and pressure telemetry."""
    def __init__(self, cfg: TimePreprocessConfig):
        """Store the preprocessing configuration."""
        self.cfg = cfg

    def transform(
        self,
        df: pd.DataFrame,
        thresholds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        columns_order: Optional[List[str]] = None,
    ):
        """Validate, align, interpolate and standardise the input telemetry."""
        cfg = self.cfg
        data = df.copy().sort_index()

        if cfg.clip_negative_to_zero:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].clip(lower=0.0)

        data_validated, invalid_mask = _apply_min_max_validation(
            data, thresholds=thresholds, zero_is_valid=cfg.zero_is_valid
        )

        df_5min, resample_gap_mask = _resample_to_5min_mean(data_validated, cfg.target_freq)

        df_imputed, imputed_mask, original_missing_mask = _interpolate_short_gaps(
            df_5min, cfg.target_freq, cfg.short_gap_limit_minutes, resample_gap_mask
        )

        # Identify prolonged missing runs as rest periods.
        rest_mask = _flag_long_gaps_as_rest(
            nan_mask=df_imputed.isna(),
            target_freq=cfg.target_freq,
            long_gap_minutes=cfg.long_gap_as_rest_minutes
        )

        if columns_order is not None:
            for c in columns_order:
                if c not in df_imputed.columns:
                    df_imputed[c] = np.nan
                    df_5min[c] = np.nan
                    invalid_mask[c] = False
                    imputed_mask[c] = False
                    rest_mask[c] = False
                    original_missing_mask[c] = False

            df_imputed = df_imputed[columns_order]
            df_5min = df_5min[columns_order]
            invalid_mask = invalid_mask[columns_order]
            imputed_mask = imputed_mask[columns_order]
            rest_mask = rest_mask[columns_order]
            original_missing_mask = original_missing_mask[columns_order]

        if cfg.apply_zscore:
            z, mean, std = _zscore(df_imputed)
            if cfg.keep_stats_for_inverse:
                self.mean_ = mean
                self.std_ = std
            out_z = z
        else:
            out_z = df_imputed
            mean = None
            std = None

        _base = pd.DataFrame(False, index=df_imputed.index, columns=df_imputed.columns)
        _inv_mask = _base.copy(); _inv_mask.update(invalid_mask)
        _rest_mask = _base.copy(); _rest_mask.update(rest_mask)
        _orig_mask = _base.copy(); _orig_mask.update(original_missing_mask)

        return {
            "df_5min": df_5min,
            "df_imputed": df_imputed,
            "imputed_mask": imputed_mask,
            "invalid_mask": _inv_mask,
            "rest_mask": _rest_mask,
            "z": out_z,
            "mean": mean,
            "std": std,
            "original_missing_mask": _orig_mask
        }


def save_preprocess_outputs(out_dir: Union[str, Path], out: dict):
    """Save preprocessed series, masks and normalisation statistics to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out["df_5min"].to_csv(out_dir / "flow_5min_raw.csv", index=True, encoding="utf-8")
    out["df_imputed"].to_csv(out_dir / "flow_5min_imputed.csv", index=True, encoding="utf-8")
    out["z"].to_csv(out_dir / "flow_5min_zscore.csv", index=True, encoding="utf-8")
    out["imputed_mask"].astype(int).to_csv(out_dir / "mask_imputed.csv", index=True, encoding="utf-8")
    out["invalid_mask"].astype(int).to_csv(out_dir / "mask_invalid.csv", index=True, encoding="utf-8")
    out["rest_mask"].astype(int).to_csv(out_dir / "mask_rest.csv", index=True, encoding="utf-8")

    df = out["df_imputed"]
    pressure_cols = [col for col in df.columns if col.startswith("P_")]
    if pressure_cols:
        pressure_coeff = 101.9716
        pressure_head_df = pd.DataFrame(index=df.index)
        for col in pressure_cols:
            ph_col = col.replace("P_", "PH_")
            pressure_head_df[ph_col] = df[col] * pressure_coeff
        pressure_head_df.to_csv(out_dir / "pressure_5min_head.csv", index=True, encoding="utf-8")
        pressure_missing_mask = df[pressure_cols].isna().astype(int)
        pressure_missing_mask.columns = [col.replace("P_", "PH_") for col in pressure_cols]
        pressure_missing_mask.to_csv(out_dir / "mask_pressure_missing.csv", index=True, encoding="utf-8")

    class NaNEncoder(json.JSONEncoder):
        def default(self, obj):
            if np.isnan(obj):
                return "NaN"
            return super().default(obj)

    if out.get("mean") is not None:
        stats = {
            "mean": out["mean"].to_dict(),
            "std": {k: (None if np.isnan(v) else float(v)) for k, v in out["std"].to_dict().items()},
        }
        with open(out_dir / "zscore_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, cls=NaNEncoder)

def quick_report(out: dict) -> str:
    """Summarise the preprocessing outputs."""
    n_all = out["df_imputed"].size
    n_nan_before = out["df_5min"].isna().sum().sum()
    n_imputed = out["imputed_mask"].sum().sum()
    n_rest = out["rest_mask"].sum().sum()
    n_nan_final = out["df_imputed"].isna().sum().sum()

    return (
        f"[Preprocess Summary]\n"
        f"- total cells (5-min grid): {n_all}\n"
        f"- NaN (raw resampling):     {n_nan_before} ({n_nan_before/n_all:.2%})\n"
        f"- Imputed cells:            {n_imputed} ({n_imputed/n_all:.2%})\n"
        f"- Rest (Long gap) cells:    {n_rest} ({n_rest/n_all:.2%})\n"
        f"- Final Remaining NaN:      {n_nan_final} ({n_nan_final/n_all:.2%}) (Input for ML Model)\n"
    )
