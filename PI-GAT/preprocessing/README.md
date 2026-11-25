# Model preprocessing · credibility of five-minute windows

`preprocess_imputed.py` assigns credibility labels using the closure residual, defined as the inlet flow `Total` minus the sum of the branch flows. It reads the authorised `Preprocess_Timeseries/outputs/preprocessed/flow_5min_imputed.csv`, skips rows with a missing or non-finite flow and skips rows in which all seven flows are zero. A zero at one valve is a valid observation.

A two-component Gaussian mixture is fitted to the positive residuals when at least 30 are available. The larger-weight component provides the centre μ and spread σ. If the fit cannot be used, the script estimates the centre from the median and the spread from the median absolute deviation.

| Rule | Label | Weight |
| --- | --- | --- |
| Positive residual within 1.5σ of μ | `high` | 1.0 |
| Positive residual more than 1.5σ and up to 3σ from μ | `mid` | 0.5 |
| Positive residual more than 3σ from μ, or a non-positive residual | `low` | 0.1 |

The CSV includes `t_index` (zero-based input row), `residual`, `credibility` and `weight`. The diagnostic ratio `k_sum_over_total` is the branch sum divided by `Total`. The auxiliary fields `bias_usable`, `bias_candidate`, `bias_weight` and `bias_candidate_level` describe potential imbalance for inspection. The current Stage-II fitting script selects high-credibility rows.

## Run

After restoring the authorised input and running Stage 0:

```bash
python PI-GAT/preprocessing/preprocess_imputed.py
```

| Output in `PI-GAT/preprocessing/` | Contents |
| --- | --- |
| `credibility_windows.csv` | Window labels, weights and diagnostic fields |
| `credibility_summary.json` | Filtering counts, residual summaries, fitted component and available pressure coverage |
| `distribution_histogram.png` | Residual distribution and credibility thresholds, when Matplotlib is available |
| `press_mask.csv` | Indicator of valid pressure-head readings, when pandas and `pressure_5min_head.csv` are available |

NumPy is required. The field inputs and timestamp-level outputs are held separately from the public package.
