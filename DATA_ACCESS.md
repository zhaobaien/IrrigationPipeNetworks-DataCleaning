# Field data access

The original operational telemetry (`PipelineDataset-F.csv`), the field-derived files generated under `Preprocess_Timeseries/outputs/preprocessed/`, and record-level artefacts such as prediction series, split manifests, logs and case traces are held separately. Access and transfer depend on the data owner's authorisation. Direct enquiries to the corresponding author using the contact details in the article.

Authorised field files can be placed at the corresponding paths in a separate local working copy. Use files prepared for the public channel labels and retain their sampling order, missing-value markers and true zero readings. The following paths are the default inputs to the scripts.

| Relative path | Contents |
| --- | --- |
| `PipelineDataset-F.csv` | Time-indexed opening, pressure and flow readings. Headers include `date`, `open_degree_<meter>` (%), `pressure(MPa)_<meter>` (MPa) and `flow(m^3/h)_<meter>` (m³/h). |
| `Preprocess_Timeseries/outputs/preprocessed/` | Aligned five-minute flows, valve openings, pressure head, standardised values, normalisation statistics and Boolean masks. Generate these with Stage 0 after placing the authorised telemetry at the root. |
| `PI-GAT/preprocessing/` | Credibility and pressure-availability tables. Generate these with `preprocess_imputed.py`. |
| `evaluation/outputs/evaluation_split_manifest.json` | Training, calibration and test endpoints used by the historical evaluator. Generate the manifest with `evaluation/split_manifest.py` or use the authorised version. |
| `data_clean/outputs/`, `Baseline/*/cleaning_results/`, `seeds/runs/` | Cleaned field series and associated diagnostic outputs. Generate them with the corresponding pipeline or use authorised archived outputs. |

## Public topology

`network_topology.csv` retains the original schema and directed pipe order. `Node` names the graph node. `Pipe`, `In` and `Out` identify each directed pipe and its endpoints. `EndNode`, `Valve` and `Corresponding pipeline` link a measurement channel to its node and hydrologically associated pipe. The six branch channels are `B01` to `B06`, with `Total` denoting the trunk and `Reservoir` the source. `Elevation`, `Length` and `Diameter` are in metres. `C` is the dimensionless Hazen–Williams coefficient.

Node, pipe and channel identifiers use consistent aliases. Elevation is relative to the source node at zero. The hydraulic connectivity, directions, lengths, diameters, coefficients and relative elevations are preserved. The checkpoint node order and meter-to-pipe relations are therefore available to the graph builder. Empty fields indicate unavailable entries in the combined table. The retained physical characteristics can still be recognisable with independent information.

Fitted leakage, scale and opening-map values are retained as model parameters. Observational identifiers, sample counts in parameter metadata and machine-specific provenance paths are held with the local materials.
