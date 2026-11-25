# Historical physical evaluation

The scripts in this directory reproduce the historical split and evaluation protocol. The default reports metrics on all active timesteps with a mass-closure tolerance derived from the calibration endpoints. The revised manuscript's common-timestamp evaluation and paired calendar-block bootstrap are separate analyses.

## Inputs and commands

Use authorised `Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv`, `PI-GAT/preprocessing/credibility_windows.csv`, the model calibration and effective opening map, and the relevant saved flow-series CSVs. The public repository does not include the restricted time series, split manifest or evaluation outputs.

Run from the project root:

```bash
python evaluation/split_manifest.py
python evaluation/unified_physical_evaluation.py
python evaluation/unified_physical_evaluation.py --report-set test --out evaluation/outputs/evaluation_metrics_test.csv
python evaluation/unified_physical_evaluation.py --legacy-all-active --out evaluation/outputs/evaluation_metrics_legacy.csv
```

## Split and report pools

Eligible high-credibility endpoints are shuffled with seed `42 + 11` after allowing a complete 12-step input window. The split assigns 70% of eligible endpoints to training, 15% to calibration and the remainder to the test set. The allocation is random at window endpoints. Windows can overlap across sets.

| Option | Report pool or tolerance |
| --- | --- |
| `--report-set all_active` (default) | All active field timesteps |
| `--report-set test` | Active timesteps in the nominal test endpoint list |
| `--threshold-set calibration` (default) | Tolerance equal to three sample standard deviations of raw-imputed absolute mass-closure error at the active calibration endpoints |
| `--legacy-all-active` | All-active report with tolerance fitted from all active high-credibility endpoints, including training endpoints |
| `--all-models` | Also evaluate available sensitivity-run flow series |

The evaluator reports output availability, the branch modification ratio, mean absolute mass-closure error, the violation rate and mean branch-wise Pearson correlation against effective opening. It reads the network leakage fraction from the Stage-II calibration record.

## Generated files

`split_manifest.py` writes `evaluation/outputs/evaluation_split_manifest.json` by default. It accepts `--out`, `--credibility-csv`, `--imputed-csv`, `--window` and `--seed`.

`unified_physical_evaluation.py` writes `evaluation/outputs/evaluation_metrics.csv` and `evaluation_run_meta.json` by default. Use `--split-manifest` to select a different manifest, `--out` to name the metrics CSV and `--write-meta` to name the metadata file. The manifest contains field timestamps and should remain with the authorised analysis materials.
