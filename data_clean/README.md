# Stage III · state-estimation-based flow cleaning

`data_cleaning.py` combines the Stage-I network estimates, Stage-II scale factors and valve states. It reverses the z-score standardisation to obtain predicted flow in m³/h and divides observed branch flows by the appropriate OD-bin scales. The residual is the scale-adjusted observation minus the prediction.

Stable, high-credibility valve intervals supply the branch residual baseline. The branch's threshold is the largest of three quantities: the mean absolute centred residual plus three sample standard deviations of that absolute residual, 10% of the branch's mean adjusted flow in the baseline, and 80 m³/h. The drift scan uses sequences of at least 12 active observations. The network closure check compares leakage-adjusted inlet flow with the sum of the branch flows. The default absolute and relative thresholds for this check are 50 m³/h and 15%.

## Run

Restore the authorised field files and generate the Stage-0 and credibility outputs before running the following from the repository root:

```bash
python data_clean/data_cleaning.py --device cpu --out-dir data_clean/outputs
```

The default replay also needs `network_topology.csv`, `PI-GAT/pi_gat_best.pth`, `PI-GAT/bias_calibration/calibration_params_final.json` and `PI-GAT/opening_mapping/Debiased_eom_full.json`. The model reads `flow_5min_imputed.csv`, `flow_5min_zscore.csv`, `mask_rest.csv` and `zscore_stats.json` from `Preprocess_Timeseries/outputs/preprocessed/` and the window labels from `PI-GAT/preprocessing/credibility_windows.csv`. Use `--ckpt`, `--calib-json` and `--eom-json` to select alternative model or calibration inputs. The default window is 12 steps and the default inference batch is 64.

| Output | Contents |
| --- | --- |
| `flow_5min_cleaned.csv` | Cleaned flow channels |
| `cleaning_flags.csv` | Per-cell cleaning decisions |
| `sensor_profile.json` | Diagnostic summary for each sensor |
| `cleaning_summary.json` | Run thresholds, flag counts and closure summaries |

### Cleaning flags

| Value | Code | Meaning |
| --- | --- | --- |
| −2 | `IGNORE` | Inactive record |
| −1 | `SPINUP` | Initial record before the first prediction window |
| 0 | `NORMAL` | No correction applied |
| 1 | `DRIFT` | Stage-II multiplicative bias correction |
| 2 | `SPIKE_FIXED` | Suspect branch value replaced by its prediction |
| 3 | `TOTAL_ADJUSTED` | Inlet flow adjusted, within the 30% limit |
| 4 | `DYNAMIC_DRIFT` | Sustained residual corrected with a moving mean |
| 5 | `ZERO` | Closed-valve flow set to zero |
| 6 | `VALVE_TRANSITION` | Observation protected around an opening change |
| 7 | `SYSTEM_ANOMALY` | Closure-based inlet correction exceeds the 30% limit |
| 8 | `FULL_RECONSTRUCT` | Fallback reconstruction weighted by network predictions |

## Diagnostic plots

After writing Stage III to its default output directory, run `python data_clean/data_analysis.py`. The script saves the sensor health radar plot, the inlet correction time series, and the time-series and density plots of mass closure error under `data_clean/analysis_results/`.

Inference uses PyTorch, PyTorch Geometric, NumPy and pandas. The diagnostic plots also use Matplotlib and SciPy. The timestamp-level inputs and outputs are held separately from the public package.
