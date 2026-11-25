# Historical Conductance WLS

The `EPANET-WLS` directory name is retained for compatibility. The script
applies a conductance-weighted algebraic adjustment to the branch flows and
holds the trunk flow fixed. It uses the archived pipe-row weight calculation;
it does not call EPANET.

The trigger is absolute mass-closure error above the median on high-credibility
records. The adjustment uses the frozen leakage fraction from
`PI-GAT/bias_calibration/calibration_params_final.json` and the pipe metadata in
`network_topology.csv`.

With authorised imputed telemetry and corresponding
`PI-GAT/preprocessing/credibility_windows.csv` in place, run from the release root:

```sh
python Baseline/EPANET-WLS/run_epanet_wls.py
```

The default local output is `Baseline/EPANET-WLS/cleaning_results/flow_5min_cleaned_epanet_wls.csv`.
Pass `--out-dir` to write elsewhere in the authorised workspace. The
timestamp-level output is held separately from the public release.
