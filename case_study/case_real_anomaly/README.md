# Historical field-window analysis

This workflow selects a field window by mass-closure error, flow smoothness
and the extent to which the Isolation Forest and Multivariate Z-score
outputs differ from the imputed record. It compares those methods with
PI-GAT on the same interval. The window is selected from the observations
and does not give independent flow truth.

Supply the authorised five-minute imputed series at
`Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv`
and aligned cleaned series at
`data_clean/outputs/flow_5min_cleaned.csv`,
`Baseline/IsolationForest/cleaning_results/flow_5min_cleaned_if.csv` and
`Baseline/Multi-variate_Z-score/cleaning_results/flow_5min_cleaned_mvz.csv`.
The script reads the leakage fraction from
`PI-GAT/bias_calibration/calibration_params_final.json` where available,
then falls back to the PI-GAT checkpoint and lastly to 0.10.

Run the scripts in order from the release root in an authorised workspace:

```sh
python case_study/case_real_anomaly/find_real_anomaly.py
python case_study/case_real_anomaly/export_case_bc.py
```

`find_real_anomaly.py` scans 1,152-step windows at 288-step intervals and uses
the active mask from `Baseline/effective_value_filtering.py`. It writes the
window and the historical kinematic compliance rate (KCR) to
`kcr_results.json` and produces `real_anomaly_comparison.png`
and `real_anomaly_zoom.png`. In this historical implementation KCR counts
readings below 5 m³/h where valve opening is below 5% and assigns 100% to
branches with no qualifying closed-valve readings.

`export_case_bc.py` reads the generated window bounds and exports selected
branch and trunk traces from a 200-step sub-window. It writes
`case_b_active_branch.csv` and `case_c_B05_crime_scene.csv` locally.
The figures, traces, indices and KCR diagnostics contain details derived
from authorised time series and are held separately from the public release.
