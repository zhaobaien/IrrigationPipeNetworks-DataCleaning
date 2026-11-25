# Stage 0 · time-series preprocessing

`scripts.py` runs `data_loading.py` on the authorised `PipelineDataset-F.csv` at the repository root. The input contains a `date` column and channels for flow, valve opening degree (OD) and pressure. The field telemetry and its five-minute derivatives are held separately from the public package.

The pipeline clips negative numeric readings to zero, applies channel bounds and takes the mean within each five-minute bin. It interpolates short interior gaps using a 10-minute limit and flags missing runs of at least 24 hours as rest periods. It retains valid zeros and saves the moments used for z-score standardisation.

## Run

From the repository root, after restoring the authorised input:

```bash
python Preprocess_Timeseries/scripts.py
```

The script uses pandas and NumPy. It writes to `Preprocess_Timeseries/outputs/preprocessed/`.

| Output | Contents |
| --- | --- |
| `flow_5min_raw.csv` | Resampled data before interpolation |
| `flow_5min_imputed.csv` | Resampled flow, OD and pressure channels after interpolation |
| `flow_5min_zscore.csv` | Standardised values used by the model |
| `zscore_stats.json` | Channel means and standard deviations |
| `mask_imputed.csv` | Cells filled by interpolation |
| `mask_invalid.csv` | Cells flagged by the bounds check |
| `mask_rest.csv` | Cells in long missing runs |
| `pressure_5min_head.csv` | Pressure head in metres for available channels |
| `mask_pressure_missing.csv` | Missing pressure-head cells |

Next, run the [credibility preprocessing](../PI-GAT/preprocessing/README.md).
