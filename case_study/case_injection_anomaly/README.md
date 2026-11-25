# Historical single-window fault injection

This workflow injects imposed disturbances into a selected historical
reference window and compares PI-GAT with Isolation Forest, Multivariate
Z-score, GAT-only, Kalman and historical Conductance WLS. The archived
single-window analysis and its calibration are distinct from the final
manuscript's 13-window replay. The reference is the recorded series before
injection.

## Inputs and execution

Work in an authorised location. Obtain the five-minute imputed and
standardised observations, rest mask and standardisation parameters under
`Preprocess_Timeseries/outputs/preprocessed/`, the aligned
`PI-GAT/preprocessing/credibility_windows.csv`, and the archived PI-GAT
cleaned observations and flags under `data_clean/outputs/`. The graph methods
also use the anonymised topology, PI-GAT calibration and effective-opening
mapping, and the PI-GAT and GAT-only checkpoints.

```sh
python case_study/case_injection_anomaly/extract_fault_profiles.py
python case_study/case_injection_anomaly/run_cascading_injection.py
```

`extract_fault_profiles.py` derives candidate magnitudes and durations from
the historical cleaning flags and aligned observations. Its local
`fault_profiles.json` covers spike ratios, runs of zero flags, gain-drift
profiles and trunk adjustments. `run_cascading_injection.py` uses those
parameters to construct four scenarios on the selected high-credibility,
active reference interval:

| ID | Disturbance | Application |
| --- | --- | --- |
| B1 | Branch spikes | Multiply eligible nonzero readings outside valve transitions. |
| B2 | Branch dropout | Set one branch flow to zero in a scheduled interval. |
| B3 | Branch gain drift | Ramp a branch gain and restore it before the trunk-bias interval. |
| B4 | Trunk bias | Multiply the trunk reading in the selected interval. |

The evaluation reports root-mean-square error and Pearson correlation
against the pre-injection readings, including scenarios evaluated at their
affected meter. The physical comparison uses the reference trunk flow for
mass closure and counts closed branch–timestep pairs with opening below 5%
and absolute flow below 5 m³/h. The historical spatial false alarm rate
counts changes exceeding both its 5% relative and 2 m³/h absolute thresholds
on branch readings at wholly un-injected timesteps.

The workflow generates a local CSV and Markdown report and intermediate
replay files. Those reports, profiles and per-timestep results are held
separately from the public release.
