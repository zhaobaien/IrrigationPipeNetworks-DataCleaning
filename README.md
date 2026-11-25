# IrrigationPipeNetworks

Physics-informed graph attention for cleaning irrigation-network telemetry. This release includes the PI-GAT and baseline implementations, trained model states, fitted calibration and effective-opening parameters, and a de-identified hydraulic topology. See [Field data access](DATA_ACCESS.md) for the telemetry and field-derived inputs used by the scripts.

## Project layout

| Directory | Purpose |
| --- | --- |
| [Preprocess_Timeseries](Preprocess_Timeseries/README.md) | Five-minute resampling, validation, missingness masks and standardisation |
| [PI-GAT](PI-GAT/README.md) | Hydraulic graph, temporal windows, model, training and calibration |
| [data_clean](data_clean/README.md) | Stage III inference, fault diagnosis and analysis |
| [ablation_study](ablation_study/README.md) | PI-GAT cleaning with Stage II omitted |
| [Baseline](Baseline/README.md) | Isolation Forest, multivariate Z-score, GAT-only, Kalman and conductance-based redistribution |
| [evaluation](evaluation/README.md) | Split generation and the historical unified evaluation protocol |
| [seeds](seeds/README.md) | Multi-seed training, progress tracking and aggregation |
| [identifiability](identifiability/README.md) | Controlled synthetic identifiability analysis using field-derived flow magnitudes |
| [case_study](case_study/case_injection_anomaly/README.md) | Field-case selection and historical fault-injection procedures |
| [sensitivity_analysis](sensitivity_analysis/README.md) | Leakage-fraction and residual-threshold experiments |

## Environment

Use a Python environment with NumPy, pandas, PyTorch, PyTorch Geometric, SciPy, scikit-learn, Matplotlib, Seaborn and tqdm. Select a CPU or CUDA build of PyTorch that suits your machine. Run the commands below from the project root. Use a separate local working copy when adding authorised data or regenerating outputs.

## Run the pipeline

```sh
# Stage 0: resample and standardise the authorised field telemetry.
python Preprocess_Timeseries/scripts.py

# Stage I: determine window credibility, fit the opening maps and train PI-GAT.
python PI-GAT/preprocessing/preprocess_imputed.py
python PI-GAT/opening_mapping/build_eom.py
python PI-GAT/train_pi_gat.py

# Stage II: calibrate sensor scales and construct the debiased opening map.
python PI-GAT/bias_calibration/instrument_bias.py
python PI-GAT/opening_mapping/Debiased_build_eom.py

# Stage III: infer states and clean flow observations.
python data_clean/data_cleaning.py
```

The bundled model states and parameter files can be reused for inference. With the authorised five-minute inputs and credibility table present, run the Stage III command directly. To preserve an existing cleaned series, pass a separate `--out-dir` to Stage III. See the module READMEs for baseline commands, the Stage II ablation and evaluation options.

`evaluation/unified_physical_evaluation.py` retains the historical report-set and tolerance options. The historical conductance baseline and the single-window case scripts retain their original methods and are documented in their respective modules.

## Licence

The project-specific code, model configurations, fitted parameters and trained weights are available for non-commercial academic research or teaching under the [Academic Use Only Licence](LICENSE). See [Field data access](DATA_ACCESS.md) for the separate conditions that apply to the topology and field data.
