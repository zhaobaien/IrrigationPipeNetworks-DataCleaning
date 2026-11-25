#!/usr/bin/env python3
"""Display seed-batch progress while training runs in another terminal."""

from __future__ import annotations

import argparse
import os
import sys
import time

from progress import STAGE_LABELS, snapshot


def _bar(pct: float, width: int = 40) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct:5.1f}%"


def _render(s: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("M8 multiseed training progress")
    lines.append("=" * 60)
    lines.append(_bar(float(s.get("overall_percent", 0))))
    lines.append(
        f"Seeds done: {s.get('n_completed', 0)}/{s.get('n_total', '?')}  "
        f"completed: {s.get('completed_seeds', [])}"
    )
    if s.get("batch_complete"):
        lines.append("Status: === BATCH COMPLETE === (all seeds + aggregate)")
        return "\n".join(lines)

    cur = s.get("current_seed")
    if cur:
        batch_seeds = s.get("batch_seeds") or s.get("all_seeds") or []
        try:
            idx = batch_seeds.index(cur) + 1 if cur in batch_seeds else "?"
            tot = len(batch_seeds) if batch_seeds else s.get("n_total", "?")
        except Exception:
            idx, tot = "?", s.get("n_total", "?")
        stage = s.get("inferred_stage") or s.get("stage") or "?"
        label = s.get("stage_label") or STAGE_LABELS.get(str(stage), stage)
        lines.append(f"Current seed: {cur}  (batch {idx}/{tot})")
        lines.append(f"Step: {label}")
        ep = s.get("epoch")
        if ep is not None and stage == "stage1":
            mx = s.get("max_epochs", 200)
            lines.append(f"Stage-I epoch: {ep}/{mx}  (early-stop patience ~30)")
    else:
        lines.append("Current: waiting or between seeds")

    lines.append(f"Updated: {s.get('updated_at', 'n/a')}")
    lines.append("Log: seeds/outputs/batch_train.log")
    lines.append("JSON: seeds/outputs/progress.json")
    lines.append("(Ctrl+C to exit watcher; training continues)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch M8 training progress")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    try:
        while True:
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")
            print(_render(snapshot()))
            sys.stdout.flush()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()
