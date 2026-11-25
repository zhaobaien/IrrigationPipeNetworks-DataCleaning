# Stage II · instrument bias calibration

`instrument_bias.py` fits multiplicative meter scales on high-credibility windows with the Stage-I parameters frozen. A scale is fitted for each diagnostic opening bin at each mapped branch. Stage III divides the observed branch flow by the corresponding scale. The calibration JSON also records the leakage rate inherited from Stage I.

## Run

Restore the authorised field inputs and generate the Stage-0 and credibility outputs first. The script reads `network_topology.csv` at the repository root and the Stage-I checkpoint in `PI-GAT/`. Run the following from the repository root to keep the published parameters and checkpoint available for replay:

```bash
python PI-GAT/bias_calibration/instrument_bias.py --device cpu --ckpt pi_gat_best.pth --out-json reproduced_calibration_params.json --out-ckpt reproduced_instrument_bias.pth
```

Relative checkpoint paths are resolved within `PI-GAT/`. Relative output paths are resolved within `PI-GAT/bias_calibration/`.

| Option | Script default | Role |
| --- | --- | --- |
| `--ckpt` | `pi_gat_best.pth` | Stage-I checkpoint |
| `--out-json` | `calibration_params_final.json` | Calibrated scales, associated bias values and leakage rate |
| `--out-ckpt` | `instrument_bias_best.pth` | Checkpoint including the fitted bias parameters |
| `--epochs-delta` | 200 | Maximum number of fitting epochs |
| `--lr-delta` | 0.01 | Learning rate for the bias parameters |
| `--min-delta` | 0.00001 | Minimum validation improvement that resets early stopping |

The resulting JSON can be passed to Stage III with `--calib-json` and to the EOM builder with `--calib_json`. Calibration uses PyTorch, PyTorch Geometric, NumPy and pandas.
