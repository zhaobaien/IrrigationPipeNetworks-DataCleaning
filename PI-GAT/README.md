# PI-GAT · network model and training

The physics-informed graph attention network estimates flows and pressure heads. Its training objective combines masked reconstruction with hydraulic residuals, including mass and energy terms.

| Component | Role |
| --- | --- |
| `model_pi_gat.py` | Graph attention layers, hydraulic residuals and constrained model parameters |
| `graph_utils.py` | Graph construction and sensor mapping from `network_topology.csv` |
| `timeseries_dataset.py` | Sliding windows of standardised measurements, opening values, masks and credibility labels |
| `train_pi_gat.py` | Stage-I training, task-gradient optimisation, logging and checkpoints |
| [`preprocessing/`](preprocessing/README.md) | Residual-based credibility labels |
| [`opening_mapping/`](opening_mapping/README.md) | Effective-opening mapping (EOM) |
| [`bias_calibration/`](bias_calibration/README.md) | Stage-II instrument bias calibration |

The `Total` sensor supplies a network inflow observation. The graph builder does not assign it to a node. The effective-opening mapping adjusts valve openings used by the energy residual.

## Stage I

Restore authorised field files and run Stage 0 and credibility preprocessing first. Training uses `flow_5min_imputed.csv`, `flow_5min_zscore.csv`, `mask_rest.csv` and `zscore_stats.json` from `Preprocess_Timeseries/outputs/preprocessed/`. The credibility labels are read from `PI-GAT/preprocessing/credibility_windows.csv`.

From the repository root, use the supplied calibration-derived EOM and write a separate checkpoint:

```bash
python PI-GAT/train_pi_gat.py --device cpu --effective_opening_mapping_json PI-GAT/opening_mapping/Debiased_eom_full.json --out-dir reproduction/stage1
```

Choose `--device cuda` to train on a supported GPU. The script writes `pi_gat_best.pth`, `training_log.csv` and `training_log_debug.csv` to the output directory. Omit `--out-dir` to write beside the script.

| Option | Script default |
| --- | --- |
| `--seed` | 42 |
| `--window` | 12 five-minute steps |
| `--batch-size` | 64 |
| `--max-epochs` | 200 |
| `--lr` | 0.001 |
| `--physics-lr-scale` | 3.0 |
| `--patience` | 30 |
| `--effective_opening_mapping_json` | `opening_mapping/Debiased_eom_full.json`, relative to the working directory |

The alternative mappings are `PI-GAT/opening_mapping/eom_high.json` (high-credibility windows) and `PI-GAT/opening_mapping/eom_full.json` (all valid windows, uncorrected). Supply the same mapping used for the checkpoint when replaying an experiment.

For Stage II, see the [instrument calibration instructions](bias_calibration/README.md). Training and calibration require Python, PyTorch, PyTorch Geometric, NumPy and pandas.
