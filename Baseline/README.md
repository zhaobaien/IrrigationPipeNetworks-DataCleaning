# Baseline methods and historical evaluation

The five baselines operate on the same five-minute series. Use the release root
as the working directory for the commands below. The public release supplies the
anonymised topology and available calibration parameters and checkpoints. Obtain
the telemetry and timestamp-level records through the authorised access process
before running the cleaning or evaluation steps.

| Task | Command from the release root | Default cleaned series or checkpoint |
| --- | --- | --- |
| Isolation Forest | `python Baseline/IsolationForest/run_isolation_forest.py` | `Baseline/IsolationForest/cleaning_results/flow_5min_cleaned_if.csv` |
| Multivariate Z-score | `python Baseline/Multi-variate_Z-score/run_multivariate_zscore.py` | `Baseline/Multi-variate_Z-score/cleaning_results/flow_5min_cleaned_mvz.csv` |
| Kalman | `python Baseline/Kalman/run_kalman.py` | `Baseline/Kalman/cleaning_results/flow_5min_cleaned_kalman.csv` |
| Historical Conductance WLS | `python Baseline/EPANET-WLS/run_epanet_wls.py` | `Baseline/EPANET-WLS/cleaning_results/flow_5min_cleaned_epanet_wls.csv` |
| GAT-only training | `python Baseline/GAT-only/train_gat.py` | `Baseline/GAT-only/gat_final.pth` |
| GAT-only cleaning | `python Baseline/GAT-only/data_cleaning_gat.py` | `Baseline/GAT-only/cleaning_results/flow_5min_cleaned_gat.csv` |

## Methods

- Isolation Forest fits one detector per branch on standardised flow and valve
  opening in the high-credibility records. The normal-equivalent tail probability
  from `--k-sigma` (default 3.0) sets the lower cutoff of the empirical training
  scores. Flagged values are interpolated.
- Multivariate Z-score fits the joint mean and covariance of the six branch
  flows and six openings using high-credibility records. The normal-equivalent
  tail probability sets an upper cutoff of the empirical Mahalanobis distances.
  For each flagged timestep the flow with the greatest univariate standardised
  deviation is marked missing and interpolated. `--k-sigma` defaults to 3.0.
- GAT-only fits the data-only graph model on standardised records with high
  credibility. Its cleaning rule uses a branch-specific adaptive residual
  threshold and a shared hydraulic deadband. `--ckpt` selects the checkpoint,
  `--k_spike` selects the residual multiplier (default 3.0), and
  `--min_abs_thresh` sets the residual floor. Use the released checkpoint for
  cleaning with the archived model.
- Kalman applies the shared active mask and high-credibility variance
  calibration to temporal estimation of the seven flow series. See
  [the Kalman notes](Kalman/README.md).
- The historical Conductance WLS script retains the original pipe-row mapping.
  See [the historical WLS notes](EPANET-WLS/README.md).

Use `--out-dir` to override the output directory for IF, MVZ, Kalman and
historical WLS. GAT-only cleaning uses `--out_dir`. Keep the generated
per-timestamp CSVs in the authorised workspace.

## Required inputs and run order

1. Prepare the authorised telemetry with `Preprocess_Timeseries/scripts.py`
   and use `PI-GAT/preprocessing/preprocess_imputed.py` to classify active
   records. Supply the aligned `flow_5min_imputed.csv`,
   `flow_5min_zscore.csv`, `mask_rest.csv` and `zscore_stats.json` under
   `Preprocess_Timeseries/outputs/preprocessed/`, and
   `PI-GAT/preprocessing/credibility_windows.csv`. The credibility table's
   `t_index` refers to the unfiltered imputed series.
2. Run the baseline entries above. Use the GAT-only checkpoint at
   `Baseline/GAT-only/gat_final.pth`, or train the model first. The WLS and
   Kalman scripts use `PI-GAT/bias_calibration/calibration_params_final.json`
   for the leakage reference where needed.
3. For the historical comparison, supply the aligned PI-GAT and ablation
   results at `data_clean/outputs/flow_5min_cleaned.csv` and
   `ablation_study/flow_5min_cleaned_without_bias.csv`. Generate them using
   their respective cleaning scripts if authorised to do so.

The shared active mask excludes a row when any of the seven flow readings is
missing or all seven are zero. Import `build_active_mask` and
`REQUIRED_COLUMNS` from `Baseline/effective_value_filtering.py`, or run
`python Baseline/effective_value_filtering.py` to generate the restricted
`Baseline/active_mask.csv`.

## Historical unified evaluation

With the required authorised inputs present, run
`python Baseline/unified_physical_evaluation.py`. It evaluates the imputed
series, IF, MVZ, GAT-only, PI-GAT and the PI-GAT bias ablation on active
timesteps and discovers available sensitivity runs. It also reads the PI-GAT
calibration, effective-opening mapping and checkpoint.

The report covers data recovery, modifications greater than 2% of the sensor
mean, mean absolute mass-closure error on complete rows, threshold exceedances
and Pearson correlation with effective opening. The mass-closure residual
compares trunk flow adjusted for leakage with the sum of the branch flows.
The evaluation writes `Baseline/evaluation_metrics.csv` locally.

The baseline scripts use NumPy, pandas, SciPy and scikit-learn. The graph
models use PyTorch and PyTorch Geometric.
