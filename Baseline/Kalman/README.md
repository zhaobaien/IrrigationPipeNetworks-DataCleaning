# Kalman baseline

The Kalman baseline estimates trunk and branch flows over time using
high-credibility records to calibrate noise. It processes active timesteps.
Mass-closure error can be assessed externally because the filter does not
impose hard mass-balance projection.

Place authorised `flow_5min_imputed.csv` under
`Preprocess_Timeseries/outputs/preprocessed/` and
`credibility_windows.csv` under `PI-GAT/preprocessing/`. Run from the release root:

```sh
python Baseline/Kalman/run_kalman.py
```

The default local output is `Baseline/Kalman/cleaning_results/flow_5min_cleaned_kalman.csv`.
Pass `--out-dir` to write to a separate authorised location. The
timestamp-level output is held separately from the public release.
