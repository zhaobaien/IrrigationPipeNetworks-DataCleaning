"""Record batch progress as JSON and optionally append events to the training log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

M8 = Path(__file__).resolve().parent
PROGRESS_JSON = M8 / "outputs" / "progress.json"
BATCH_LOG = M8 / "outputs" / "batch_train.log"

STAGES = ("stage1", "stage2", "eom", "stage3", "aggregate")
STAGE_LABELS = {
    "stage1": "Stage-I  PI-GAT train",
    "stage2": "Stage-II bias calibration",
    "eom": "Debiased EOM build",
    "stage3": "Stage-III data cleaning",
    "aggregate": "Multiseed aggregate",
    "done": "All complete",
    "idle": "Idle",
}
# Each stage has a share of one seed run. The shares sum to one.
STAGE_WEIGHT = {"stage1": 0.70, "stage2": 0.15, "eom": 0.05, "stage3": 0.10}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load() -> dict:
    if not PROGRESS_JSON.exists():
        return {}
    try:
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(state: dict) -> None:
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_batch_log(msg: str) -> None:
    BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {msg}\n"
    with open(BATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def update_progress(
    *,
    batch_seeds: list[int] | None = None,
    current_seed: int | None = None,
    stage: str | None = None,
    status: str = "running",
    max_epochs: int = 200,
    append_log: bool = True,
) -> None:
    state = _load()
    if batch_seeds is not None:
        state["batch_seeds"] = batch_seeds
        state["batch_active"] = True
    if current_seed is not None:
        state["current_seed"] = current_seed
        if batch_seeds and current_seed in batch_seeds:
            state["seed_index"] = batch_seeds.index(current_seed) + 1
            state["seed_total"] = len(batch_seeds)
    if stage is not None:
        state["stage"] = stage
        state["stage_label"] = STAGE_LABELS.get(stage, stage)
    state["status"] = status
    state["max_epochs"] = max_epochs
    if status == "batch_complete":
        state["batch_active"] = False
        state["stage"] = "done"
        state["stage_label"] = STAGE_LABELS["done"]
    _save(state)
    if append_log and stage and current_seed is not None:
        append_batch_log(f"seed {current_seed} | {STAGE_LABELS.get(stage, stage)} | {status}")


def read_training_epoch(seed: int) -> int | None:
    log_csv = M8 / "runs" / f"seed_{seed}" / "stage1" / "training_log.csv"
    if not log_csv.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(log_csv)
        if "Epoch" not in df.columns or len(df) == 0:
            return None
        return int(df["Epoch"].iloc[-1])
    except Exception:
        return None


def seed_is_complete(seed: int) -> bool:
    return (M8 / "runs" / f"seed_{seed}" / "stage3" / "flow_5min_cleaned.csv").exists()


def infer_stage(seed: int) -> str | None:
    base = M8 / "runs" / f"seed_{seed}"
    if seed_is_complete(seed):
        return None
    calib = base / "stage2" / "calibration_params_final.json"
    eom = base / "stage2" / "Debiased_eom_full.json"
    train_log = base / "stage1" / "training_log.csv"
    if not calib.exists():
        # The checkpoint appears during Stage I. Use the calibration JSON to mark Stage I as complete.
        if train_log.exists() or (base / "stage1" / "pi_gat_best.pth").exists():
            return "stage1"
        return "stage1"
    if not eom.exists():
        return "stage2"
    return "stage3"


def compute_overall_percent(
    all_seeds: list[int],
    current_seed: int | None,
    stage: str | None,
    epoch: int | None,
    max_epochs: int = 200,
    aggregate_done: bool = False,
) -> float:
    if aggregate_done:
        return 100.0
    n = len(all_seeds)
    if n == 0:
        return 0.0
    completed = sum(1 for s in all_seeds if seed_is_complete(s))
    base = completed / n
    if current_seed is None or stage is None or stage == "done":
        return min(100.0, base * 100.0)
    if stage == "aggregate":
        return min(99.0, (completed / n) * 100.0 + (100.0 / n) * 0.95)
    within = 0.0
    if stage == "stage1" and epoch is not None:
        within = STAGE_WEIGHT["stage1"] * min(1.0, epoch / max_epochs)
    else:
        done_w = 0.0
        for st in STAGES:
            if st == "aggregate":
                break
            if st == stage:
                within = done_w + STAGE_WEIGHT.get(st, 0) * 0.5
                break
            done_w += STAGE_WEIGHT.get(st, 0)
    seed_frac = (completed + within) / n
    return min(99.5, seed_frac * 100.0)


def infer_current_seed(seeds: list[int]) -> int | None:
    for s in seeds:
        if not seed_is_complete(s):
            return s
    return None


def snapshot(all_seeds: list[int] | None = None) -> dict:
    state = _load()
    seeds = all_seeds or state.get("batch_seeds") or [42, 123, 456, 789, 2024]
    cur = state.get("current_seed") or infer_current_seed(seeds)
    stage = state.get("stage")
    if cur and not stage:
        stage = infer_stage(cur)
    epoch = read_training_epoch(cur) if cur else None
    max_ep = int(state.get("max_epochs", 200))
    agg_done = (M8 / "outputs" / "multiseed_summary.csv").exists() and state.get("status") == "batch_complete"
    pct = compute_overall_percent(seeds, cur, stage, epoch, max_ep, agg_done)
    completed = [s for s in seeds if seed_is_complete(s)]
    return {
        **state,
        "current_seed": cur,
        "all_seeds": seeds,
        "completed_seeds": completed,
        "n_completed": len(completed),
        "n_total": len(seeds),
        "inferred_stage": stage,
        "epoch": epoch,
        "overall_percent": round(pct, 1),
        "batch_complete": state.get("status") == "batch_complete",
    }
