# Stage-III ablation · instrument bias correction omitted

`data_cleaning_without_bias.py` sets every branch's instrument scale to 1.0, so the assimilation starts from the unadjusted observations. It retains valve-state masking, the residual baseline, dynamic drift correction and the closure-guided branch and inlet corrections. The leakage rate is read from the Stage-I checkpoint.

The archived ablation reuses the checkpoint trained with the high-credibility effective-opening mapping. The default checkpoint is `PI-GAT/stage1_results_high_eom/pi_gat_best.pth` and the default map is `PI-GAT/opening_mapping/eom_high.json`. The full pipeline uses the separate debiased mapping and its companion checkpoint. These settings describe the archived implementations and matter when interpreting their comparison.

## Run

After restoring authorised field telemetry and generating the Stage-0 and credibility files, run from the repository root:

```bash
python ablation_study/data_cleaning_without_bias.py --device cpu
```

The script accepts `--ckpt`, `--eom-json`, `--window` and `--batch`. A Stage-II calibration file is unnecessary for this run.

| Generated file in `ablation_study/` | Contents |
| --- | --- |
| `flow_5min_cleaned_without_bias.csv` | Five-minute ablation output |
| `cleaning_flags_without_bias.csv` | Per-cell flags as defined in the [Stage-III README](../data_clean/README.md#cleaning-flags) |
| `sensor_profile_without_bias.json` | Per-sensor diagnostic summary |
| `cleaning_summary_without_bias.json` | Run thresholds, closure summaries and flag counts |

The comparison consumes the regenerated flow CSV from this directory. Timestamp-level derivatives are held separately from the public package.
