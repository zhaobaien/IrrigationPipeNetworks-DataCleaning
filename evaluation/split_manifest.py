"""
Export the historical Stage-I training, calibration and test split.

Reproduce the 70/15/15 random allocation of high-credibility window endpoints
used by PI-GAT/train_pi_gat.py. The manifest records row indices and timestamps.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SEED = 42
DEFAULT_SHUFFLE_SEED_OFFSET = 11
DEFAULT_WINDOW = 12
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def _level_from_label(lbl: str) -> int:
    s = str(lbl).strip().lower()
    if s == "high":
        return 2
    if s == "mid":
        return 1
    return 0


def _split_70_15_15(xs: list[int]) -> tuple[list[int], list[int], list[int]]:
    n = len(xs)
    if n <= 0:
        return [], [], []
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
    train = xs[:n_train]
    val = xs[n_train : n_train + n_val]
    test = xs[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def build_high_cred_split(
    credibility_csv: Path,
    imputed_csv: Path,
    *,
    window: int = DEFAULT_WINDOW,
    seed: int = DEFAULT_SEED,
    shuffle_seed_offset: int = DEFAULT_SHUFFLE_SEED_OFFSET,
) -> dict[str, Any]:
    cred_df = pd.read_csv(credibility_csv)
    for col in ("t_index", "credibility"):
        if col not in cred_df.columns:
            raise ValueError(f"Missing column {col!r} in {credibility_csv}")

    cred_level_by_t: dict[int, int] = {}
    for row in cred_df.itertuples(index=False):
        t_idx = int(row.t_index)
        cred_level_by_t[t_idx] = _level_from_label(row.credibility)

    df_imputed = pd.read_csv(imputed_csv, index_col="date", parse_dates=True)
    n_rows = len(df_imputed)

    # Construct valid endpoints in the same way as IrrigationTimeWindowDataset.
    valid_t = [t for t in range(window - 1, n_rows) if t in cred_level_by_t]

    idx_high = [di for di, t in enumerate(valid_t) if cred_level_by_t[int(t)] >= 2]

    idx_high_s = idx_high.copy()
    rng = random.Random(seed + shuffle_seed_offset)
    rng.shuffle(idx_high_s)

    train_di, val_di, test_di = _split_70_15_15(idx_high_s)

    def _map_indices(dataset_indices: list[int]) -> tuple[list[int], list[str]]:
        t_indices = [int(valid_t[di]) for di in dataset_indices]
        dates = [str(df_imputed.index[t]) for t in t_indices]
        return t_indices, dates

    train_t, train_dates = _map_indices(train_di)
    cal_t, cal_dates = _map_indices(val_di)
    test_t, test_dates = _map_indices(test_di)

    return {
        "protocol": "M4_evaluation_v1",
        "description": "Stage-I high-credibility 70/15/15 split aligned with train_pi_gat.py",
        "parameters": {
            "seed": seed,
            "shuffle_seed": seed + shuffle_seed_offset,
            "window": window,
            "train_frac": TRAIN_FRAC,
            "calibration_frac": VAL_FRAC,
            "test_frac": "remainder",
        },
        "counts": {
            "valid_t_samples": len(valid_t),
            "high_cred_dataset_indices": len(idx_high),
            "train_samples": len(train_di),
            "calibration_samples": len(val_di),
            "test_samples": len(test_di),
        },
        "train_t_index": train_t,
        "calibration_t_index": cal_t,
        "test_t_index": test_t,
        "train_dates": train_dates,
        "calibration_dates": cal_dates,
        "test_dates": test_dates,
        "paths": {
            "credibility_csv": str(credibility_csv.resolve()),
            "imputed_csv": str(imputed_csv.resolve()),
        },
    }


def export_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[split_manifest] Saved: {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    default_out = Path(__file__).resolve().parent / "outputs" / "evaluation_split_manifest.json"

    parser = argparse.ArgumentParser(description="Export M4 evaluation train/calibration/test split manifest.")
    parser.add_argument(
        "--credibility-csv",
        type=Path,
        default=root / "PI-GAT" / "preprocessing" / "credibility_windows.csv",
    )
    parser.add_argument(
        "--imputed-csv",
        type=Path,
        default=root
        / "Preprocess_Timeseries"
        / "outputs"
        / "preprocessed"
        / "flow_5min_imputed.csv",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    manifest = build_high_cred_split(
        args.credibility_csv,
        args.imputed_csv,
        window=args.window,
        seed=args.seed,
    )
    export_manifest(manifest, args.out)

    c = manifest["counts"]
    print(
        f"[split_manifest] high={c['high_cred_dataset_indices']} "
        f"train/cal/test={c['train_samples']}/{c['calibration_samples']}/{c['test_samples']}"
    )


if __name__ == "__main__":
    main()
