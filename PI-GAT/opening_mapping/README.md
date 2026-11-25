# Effective-opening mapping

The EOM maps reported valve opening degree (OD) to an effective opening used in the energy residual. The fitting code in `eom.py` uses segmented flow quantiles and a pool-adjacent-violators isotonic fit to learn monotone per-valve mappings.

| Entry point | Fit and outputs |
| --- | --- |
| `build_eom.py` | Fits `eom_full.json` from all valid rows and `eom_high.json` from the high-credibility subset. Writes `eom_high_vs_full.json` when both are available. |
| `Debiased_build_eom.py` | Divides each branch flow by its OD-bin scale before fitting `Debiased_eom_full.json`. Writes `new_eom_high_vs_full.json` when both earlier fits are available. |

The first script reads the authorised five-minute imputed flow and OD data and the generated credibility CSV. The second also reads `PI-GAT/bias_calibration/calibration_params_final.json`. Its opening-bin boundaries follow the model.

## Regenerate the fits

From the repository root, after Stage 0 and credibility preprocessing, run the following to save fresh files separately from the supplied mappings:

```bash
python PI-GAT/opening_mapping/build_eom.py --out reproduction/eom/eom_full.json
python PI-GAT/opening_mapping/Debiased_build_eom.py --calib_json PI-GAT/bias_calibration/calibration_params_final.json --eom_full_json reproduction/eom/eom_full.json --eom_high_json reproduction/eom/eom_high.json --out reproduction/eom/Debiased_eom_full.json
```

Both scripts accept `--flow` to override the imputed input and `--out_comparison` to control the comparison JSON path. Pass a mapping to the training script using `--effective_opening_mapping_json` and to Stage III using `--eom-json`.

The fitting scripts use NumPy and pandas.
