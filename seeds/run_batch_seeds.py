#!/usr/bin/env python3
"""Run the seed experiments in sequence and aggregate their evaluation metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M8 = Path(__file__).resolve().parent
sys.path.insert(0, str(M8))
import run_single_seed as _rs  # noqa: E402
from progress import append_batch_log, update_progress  # noqa: E402

PY = _rs.PY
LIVE_LOG = M8 / "outputs" / "batch_live.log"


def run_logged(cmd: list[str], cwd: Path, meta_fh) -> None:
    """Run a child process and append its standard output and errors to the live log."""
    env = _rs._subprocess_env()
    if "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"EXEC {' '.join(cmd)}", meta_fh)
    with open(LIVE_LOG, "a", encoding="utf-8") as live:
        live.write(f"\n{'='*72}\n[{datetime.now().isoformat(timespec='seconds')}] EXEC\n")
        live.write(" ".join(cmd) + "\n")
        live.flush()
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            env=env,
            stdout=live,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789, 2024])
    parser.add_argument("--include-42", action="store_true", help="Also import seed 42 before aggregate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log_path = M8 / "outputs" / "batch_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as fh:
        log(f"=== batch start seeds={args.seeds} smoke={args.smoke} python={PY} ===", fh)
        append_batch_log(f"=== batch RESTART seeds={args.seeds} ===")
        update_progress(batch_seeds=args.seeds, status="running", append_log=False)

        if args.include_42:
            cmd = [PY, str(M8 / "run_single_seed.py"), "--seed", "42", "--import-existing-42"]
            log(f"RUN {' '.join(cmd)}", fh)
            subprocess.run(cmd, cwd=str(ROOT), check=True, env=_rs._subprocess_env())

        for seed in args.seeds:
            cmd = [
                PY, "-u", str(M8 / "run_single_seed.py"),
                "--seed", str(seed),
                "--batch-seeds", *[str(s) for s in args.seeds],
            ]
            if args.smoke:
                cmd.append("--smoke")
            log(f"RUN {' '.join(cmd)}", fh)
            update_progress(batch_seeds=args.seeds, current_seed=seed, stage="stage1", status="running", append_log=False)
            try:
                run_logged(cmd, ROOT, fh)
                log(f"OK seed {seed} (full pipeline)", fh)
                append_batch_log(f"OK seed {seed} full pipeline")
            except subprocess.CalledProcessError as e:
                log(f"FAIL seed {seed} exit={e.returncode}", fh)
                append_batch_log(f"FAIL seed {seed} exit={e.returncode}")
                update_progress(status="failed", append_log=False)
                raise

        all_seeds = ([42] if args.include_42 else []) + list(args.seeds)
        if 42 not in all_seeds:
            all_seeds = [42] + all_seeds

        update_progress(current_seed=None, stage="aggregate", status="running", append_log=False)
        append_batch_log("aggregate multiseed metrics START")
        agg = [PY, "-u", str(M8 / "aggregate_multiseed.py"), "--seeds", *[str(s) for s in all_seeds]]
        log(f"RUN {' '.join(agg)}", fh)
        run_logged(agg, ROOT, fh)
        append_batch_log("aggregate multiseed metrics DONE")
        log("=== batch complete ===", fh)
        update_progress(stage="done", status="batch_complete", append_log=False)

    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
