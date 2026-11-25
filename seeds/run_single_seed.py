#!/usr/bin/env python3
"""Run Stages I, II and III for one random seed."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from progress import append_batch_log, update_progress

ROOT = Path(__file__).resolve().parent.parent
M8 = Path(__file__).resolve().parent
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def resolve_python() -> str:
    import os

    if os.environ.get("M8_PYTHON"):
        return os.environ["M8_PYTHON"]
    candidates = [
        Path.home() / "miniconda3/envs/PyTorch/python.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


PY = resolve_python()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(cmd: list[str], cwd: Path | None = None) -> None:
    if cmd[0] == sys.executable:
        cmd[0] = PY
    if "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True, env=_subprocess_env())


def seed_dirs(seed: int) -> dict[str, Path]:
    base = M8 / "runs" / f"seed_{seed}"
    return {
        "base": base,
        "stage1": base / "stage1",
        "stage2": base / "stage2",
        "stage3": base / "stage3",
    }


def import_seed42_existing(d: dict[str, Path]) -> None:
    """Reuse the existing seed-42 artefacts."""
    for sub, src in [
        (d["stage1"] / "pi_gat_best.pth", ROOT / "PI-GAT" / "pi_gat_best.pth"),
        (d["stage2"] / "calibration_params_final.json", ROOT / "PI-GAT" / "bias_calibration" / "calibration_params_final.json"),
        (d["stage2"] / "Debiased_eom_full.json", ROOT / "PI-GAT" / "opening_mapping" / "Debiased_eom_full.json"),
        (d["stage3"] / "flow_5min_cleaned.csv", ROOT / "data_clean" / "outputs" / "flow_5min_cleaned.csv"),
    ]:
        sub.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, sub)
    print(f"[seed 42] Imported existing artefacts into {d['base']}")


def run_pipeline(seed: int, *, smoke: bool = False, batch_seeds: list[int] | None = None) -> None:
    py = PY
    d = seed_dirs(seed)
    d["stage1"].mkdir(parents=True, exist_ok=True)
    d["stage2"].mkdir(parents=True, exist_ok=True)
    d["stage3"].mkdir(parents=True, exist_ok=True)

    if batch_seeds:
        update_progress(batch_seeds=batch_seeds, current_seed=seed, stage="stage1", status="running")

    default_eom = ROOT / "PI-GAT" / "opening_mapping" / "Debiased_eom_full.json"
    train_cmd = [
        py, str(ROOT / "PI-GAT" / "train_pi_gat.py"),
        "--seed", str(seed),
        "--out-dir", str(d["stage1"]),
        "--effective_opening_mapping_json", str(default_eom),
    ]
    if smoke:
        train_cmd.append("--smoke")
    append_batch_log(f"seed {seed} | Stage-I train START")
    run(train_cmd)
    append_batch_log(f"seed {seed} | Stage-I train DONE")

    update_progress(current_seed=seed, stage="stage2", status="running")
    bias_cmd = [
        py, str(ROOT / "PI-GAT" / "bias_calibration" / "instrument_bias.py"),
        "--seed", str(seed),
        "--ckpt", str(d["stage1"] / "pi_gat_best.pth"),
        "--out-json", str(d["stage2"] / "calibration_params_final.json"),
        "--out-ckpt", str(d["stage2"] / "instrument_bias_best.pth"),
    ]
    append_batch_log(f"seed {seed} | Stage-II bias START")
    run(bias_cmd, cwd=ROOT / "PI-GAT" / "bias_calibration")
    append_batch_log(f"seed {seed} | Stage-II bias DONE")

    update_progress(current_seed=seed, stage="eom", status="running")
    eom_cmd = [
        py, str(ROOT / "PI-GAT" / "opening_mapping" / "Debiased_build_eom.py"),
        "--calib_json", str(d["stage2"] / "calibration_params_final.json"),
        "--out", str(d["stage2"] / "Debiased_eom_full.json"),
    ]
    append_batch_log(f"seed {seed} | Debiased EOM START")
    run(eom_cmd, cwd=ROOT / "PI-GAT" / "opening_mapping")
    append_batch_log(f"seed {seed} | Debiased EOM DONE")

    update_progress(current_seed=seed, stage="stage3", status="running")
    clean_cmd = [
        py, str(ROOT / "data_clean" / "data_cleaning.py"),
        "--ckpt", str(d["stage1"] / "pi_gat_best.pth"),
        "--calib-json", str(d["stage2"] / "calibration_params_final.json"),
        "--eom-json", str(d["stage2"] / "Debiased_eom_full.json"),
        "--out-dir", str(d["stage3"]),
    ]
    append_batch_log(f"seed {seed} | Stage-III clean START")
    run(clean_cmd)
    append_batch_log(f"seed {seed} | Stage-III clean DONE | seed pipeline complete")
    update_progress(current_seed=seed, stage="stage3", status="seed_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 single-seed PI-GAT pipeline")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--import-existing-42", action="store_true", help="Copy default seed=42 outputs")
    parser.add_argument("--smoke", action="store_true", help="Fast sanity train (2 epochs)")
    parser.add_argument(
        "--batch-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Full batch seed list (for progress.json); set by run_batch_seeds.py",
    )
    args = parser.parse_args()

    if args.import_existing_42 and args.seed == 42:
        import_seed42_existing(seed_dirs(42))
        return

    run_pipeline(args.seed, smoke=args.smoke, batch_seeds=args.batch_seeds)


if __name__ == "__main__":
    main()
