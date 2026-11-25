# Synthetic identifiability experiment

The experiment examines whether the network leakage fraction and branch-meter offsets can be identified separately using the Stage-II relative mass-balance objective. The code uses `gamma` for the leakage fraction, denoted by λ in the manuscript, and `delta` for the branch offsets.

The experiment samples up to 800 positive, high-credibility field timesteps to set the scale of the branch flows. It generates trunk flows that satisfy the balance at an assumed leakage fraction of 10% and applies heterogeneous branch offsets from +1.5% to +4.0%. The code compares joint fits from four specified starting points with fixed-leakage fits. Branch offsets are bounded to ±5% and fitted with L-BFGS-B.

## Run

```bash
python identifiability/run_synthetic_identifiability.py
```

Run from the project root with authorised `Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv` and `PI-GAT/preprocessing/credibility_windows.csv`. The synthetic experiment uses field flow magnitudes and the credibility mask. These inputs and generated output files are held separately from the public repository.

## Generated outputs

| File under `identifiability/outputs/` | Content |
| --- | --- |
| `equifinality_scan.csv` | Loss and offset recovery error at each fixed leakage fraction |
| `joint_multistart.csv` | Joint leakage and offset fits from four starting points |
| `sequential_branch_recovery.csv` | Per-branch offset recovery at the injected leakage fraction |
| `sequential_gamma_mismatch.csv` | Recovery error when the fixed leakage fraction is varied |
| `identifiability_summary.json` | Numeric summary of the experiment |

`plot_identifiability.py` is an optional wrapper for the historical plotting script `illustrations/Figure 11/plot_figure11_identifiability.py`, which is outside this public repository. The numerical entry point writes the CSV and JSON results on its own.
