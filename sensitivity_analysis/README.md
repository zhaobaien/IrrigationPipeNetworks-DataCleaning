# Sensitivity analysis

The two runners examine how flow cleaning responds to an assumed network leakage fraction and a residual threshold multiplier. They use the historical procedure and require authorised field data. The restricted preprocessed records, run outputs and per-timestep flow series are held separately from the public code.

## Fixed network leakage

```bash
python sensitivity_analysis/run_leakage_sensitivity.py
```

Run from the project root after preparing the preprocessed field series, credibility windows and effective opening inputs. For each leakage fraction in `0.0`, `0.05`, `0.1`, `0.15` and `0.2`, the runner trains Stage I with leakage held fixed, fits Stage-II branch scales at the same value and applies Stage III using the resulting checkpoint and calibration. Each setting requires its own training run.

Within each `sensitivity_analysis_<fraction>/` directory, the runner checks for existing checkpoints, calibrations and cleaned flow series before launching the corresponding stage. Use a separate working copy when comparing independent runs.

## Residual multiplier

```bash
python sensitivity_analysis/run_sigma_sensitivity.py
```

For multipliers `2`, `3` and `4`, the runner reuses the method inputs, including saved model weights where required. It runs PI-GAT, GAT-only, Isolation Forest and the multivariate Z-score method in sequence and writes each setting to `sensitivity_analysis_<k>-Sigma/`. The multiplier sweep does not retrain the models. Review the messages for individual method failures because the runner continues to the next method after an error.

## Evaluation

Run the historical evaluator with `--all-models` to discover saved sweep outputs:

```bash
python evaluation/unified_physical_evaluation.py --all-models
```

The evaluator labels the fixed-leakage runs `Leakage_*` and the multiplier runs `PI-GAT_*-Sigma`, `GAT_*-Sigma`, `IF_*-Sigma` and `MVZ_*-Sigma`. See the [evaluation README](../evaluation/README.md) for report pools and threshold choices. The revised manuscript presents the PI-GAT and GAT-only multiplier series and discusses the hydraulic deadband separately.
