# PI-GAT stability across training seeds

The historical training runs use seeds `42`, `123`, `456`, `789` and `2024`. The batch runner trains the four later seeds by default and then aggregates all five. The seed-42 results must already exist or be imported with `--include-42`.

## Requirements

Use an environment with the dependencies listed in the project README. Set the optional `M8_PYTHON` environment variable to select a different interpreter. Without that override, the runner checks the standard per-user Miniconda environment named `PyTorch` and then falls back to the current interpreter.

These commands require authorised field data and matching preprocessed inputs, calibration, opening maps and reconstruction outputs. The public repository does not include the restricted time series, evaluation split or run outputs. Generate the split using [the historical evaluation instructions](../evaluation/README.md) before running the aggregation.

## Run from the project root

```bash
# Import available seed-42 artefacts, including its Stage-III flow series.
python seeds/run_single_seed.py --seed 42 --import-existing-42

# Train a single seed, or train the four later seeds and aggregate all five.
python seeds/run_single_seed.py --seed 123
python seeds/run_batch_seeds.py --seeds 123 456 789 2024
```

The import requires the existing seed-42 checkpoint, calibration, effective opening map and `data_clean/outputs/flow_5min_cleaned.csv`. If you have the authorised inputs but need to regenerate the seed-42 series, run `python seeds/run_single_seed.py --seed 42`. In a second terminal, `python seeds/watch_progress.py` displays progress. Add `--once` to show a single snapshot.

The batch is complete when the log reports `=== batch complete ===`. See `outputs/batch_train.log` for stage events and `outputs/batch_live.log` for child process output.

## Generated outputs

| File | Content |
| --- | --- |
| `runs/seed_*/stage3/flow_5min_cleaned.csv` | Cleaned flow series for each seed |
| `outputs/multiseed_metrics.csv` | Per-seed metrics using the historical all-active pool and calibration-derived tolerance |
| `outputs/multiseed_summary.csv` | Mean and sample standard deviation across seeds |
| `outputs/wilcoxon_paired.csv` | Historical seed-42 comparisons if `paired_wilcoxon.py` is run separately |

The seed aggregation and paired Wilcoxon script implement the historical protocol. The revised manuscript uses a common-timestamp cohort for the seed comparison and a paired calendar-block bootstrap for the uncertainty intervals.
