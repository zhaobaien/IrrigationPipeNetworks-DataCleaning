from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List, cast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import json


class IrrigationTimeWindowDataset(Dataset):
    """Represent time-windowed telemetry as PyTorch Geometric graph samples."""

    def __init__(
        self,
        data_graph: Data,
        zscore_path: str | Path,
        restmask_path: str | Path,
        sensor2node_idx: Dict[str, int],
        sensor2edge_idx: Dict[str, int],
        window: int = 12,
        random_mask_range: Optional[Tuple[float, float]] = (0.10, 0.20),
        sensors_order: Optional[List[str]] = None,
        open_paths: Optional[str | Path] = None,
        zscore_stats_path: Optional[str | Path] = None,
        pressure_head_path: Optional[str | Path] = None,
        credibility_path: Optional[str | Path] = None,
        verbose: bool = True,
    ):
        def _log(msg: str):
            if verbose:
                print(msg)

        self.g = data_graph
        self.window = int(window)
        assert self.window >= 1

        Z = pd.read_csv(zscore_path, index_col=0, parse_dates=True, low_memory=False)
        Z = Z.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        R = pd.read_csv(restmask_path, index_col=0, parse_dates=True)

        if sensors_order:
            available_cols = [c for c in sensors_order if c in Z.columns]
            Z = Z.reindex(columns=available_cols)
            R = R.reindex(columns=available_cols)

        self.time_index = Z.index
        self.sensor_columns = Z.columns.tolist()

        _log(f"[Dataset] Time range: {self.time_index[0]} - {self.time_index[-1]}")
        _log(f"[Dataset] Sensor columns: {self.sensor_columns}")

        N = self.g.num_nodes
        assert N is not None
        node_names = list(self.g.node_names)

        X_np = np.full((len(Z), N), np.nan, dtype=np.float32)
        Rest_np = np.zeros((len(R), N), dtype=np.bool_)

        for sensor, ni in sensor2node_idx.items():
            if sensor in Z.columns:
                X_np[:, ni] = Z[sensor].to_numpy(dtype=np.float32)
                Rest_np[:, ni] = R[sensor].astype(bool).to_numpy(dtype=np.bool_)
                _log(f"[Dataset] Sensor {sensor} -> node {node_names[ni]} (idx={ni})")

        self.X_tensor = torch.from_numpy(X_np).to(torch.float32)
        self.Rest_tensor = torch.from_numpy(Rest_np).to(torch.bool)

        self.edge_meas_cols = list({s for s in sensor2edge_idx.keys() if s in Z.columns})
        self.edge_meas_map = sensor2edge_idx

        self.Z_tensor = torch.from_numpy(Z.values).to(torch.float32)
        self.Z_columns = Z.columns

        self.edge_meas_pairs: List[Tuple[int, int]] = []  # (edge_idx, z_col_idx)
        for s in self.edge_meas_cols:
            try:
                self.edge_meas_pairs.append((int(self.edge_meas_map[s]), cast(int, self.Z_columns.get_loc(s))))
            except Exception:
                continue

        means, stds = {}, {}
        if zscore_stats_path and Path(zscore_stats_path).exists():
            stats = json.load(open(zscore_stats_path, "r", encoding="utf-8"))
            means = stats.get("mean", {})
            stds = stats.get("std", {})
            _log(f"[Dataset] Loaded z-score statistics: {list(means.keys())}")

        self.total_inflow_sensor = "Total" if ("Total" in self.Z_columns) else None
        self.total_inflow_col_idx: Optional[int] = None
        self.total_inflow_mean_m3ph: Optional[float] = None
        self.total_inflow_std_m3ph: Optional[float] = None
        if self.total_inflow_sensor is not None:
            self.total_inflow_col_idx = cast(int, self.Z_columns.get_loc(self.total_inflow_sensor))
            if self.total_inflow_sensor in means and self.total_inflow_sensor in stds:
                self.total_inflow_mean_m3ph = float(means[self.total_inflow_sensor])
                self.total_inflow_std_m3ph = max(float(stds[self.total_inflow_sensor]), 1e-6)
                _log(
                    f"[Dataset] Global Total inflow observation enabled: mean={self.total_inflow_mean_m3ph:.2f}, "
                    f"std={self.total_inflow_std_m3ph:.2f} (m3/h)"
                )
            else:
                _log(f"[Dataset] ⚠️ Global inflow constraint disabled because z-score statistics are missing for Total.")

        N = self.g.num_nodes
        assert N is not None
        self.node_mean_m3ph = torch.zeros(N, dtype=torch.float32)
        self.node_std_m3ph = torch.ones(N, dtype=torch.float32)
        self.node_has_stats = torch.zeros(N, dtype=torch.bool)

        for sensor, ni in sensor2node_idx.items():
            if sensor in means and sensor in stds:
                self.node_mean_m3ph[ni] = float(means[sensor])
                std_val = float(stds[sensor])
                self.node_std_m3ph[ni] = max(std_val, 1e-6)
                self.node_has_stats[ni] = True
                _log(f"[Dataset] Node {node_names[ni]} (idx={ni}) -> Sensor {sensor}: mean={means[sensor]:.2f}, std={std_val:.2f}")

        self.has_open = False
        self.OpenNode = None
        self.OpenEdge = None
        self.OpenNode_tensor: Optional[torch.Tensor] = None  # [T, N]
        self.OpenEdge_tensor: Optional[torch.Tensor] = None  # [T, E]

        if open_paths:
            open_file = Path(open_paths)
            if open_file.exists():
                df = pd.read_csv(open_file, index_col=0, parse_dates=True)
                _log(f"[Dataset] Original opening data shape: {df.shape}")
                _log(f"[Dataset] Opening data columns: {df.columns.tolist()}")

                open_cols = [c for c in df.columns if c.startswith("OD_")]
                if not open_cols:
                    open_cols = [c for c in df.columns if "open_degree" in c.lower()]

                def _clean_name(c):
                    """Extract the meter name from a supported opening column."""
                    c = str(c)
                    if c.startswith("OD_"):
                        return c[3:]
                    if c.startswith("open_degree_"):
                        return c[len("open_degree_"):]
                    return c

                if open_cols:
                    Open = df[open_cols].copy()
                    Open.columns = [_clean_name(c) for c in open_cols]
                    Open = Open.reindex(self.time_index, method='ffill').fillna(0.0)
                    _log(f"[Dataset] Cleaned opening data columns: {Open.columns.tolist()}")
                    _log(f"[Dataset] Aligned opening data shape: {Open.shape}")
                    _log(f"[Dataset] Opening value range: {Open.min().min():.1f} - {Open.max().max():.1f}")

                    self.OpenNode = pd.DataFrame(
                        index=self.time_index,
                        columns=node_names,
                        data=1.0
                    )

                    for s, ni in sensor2node_idx.items():
                        if s in Open.columns:
                            self.OpenNode.iloc[:, ni] = np.divide(Open[s].to_numpy(dtype=float), 100.0)
                            _log(f"[Dataset] Node {node_names[ni]} ({s}) opening values loaded")

                    assert self.g.edge_index is not None
                    assert self.g.edge_attr is not None
                    E_count = self.g.edge_index.size(1)
                    EdgeOpen = np.ones((len(self.time_index), E_count), dtype=float)

                    is_self_tmp = (self.g.edge_attr[:, 5].numpy() > 0.5)
                    EdgeOpen[:, is_self_tmp] = 0.0

                    valve_mapping = getattr(self.g, 'valve_mapping', None)
                    if valve_mapping is not None:
                        for valve_name, edge_idx in valve_mapping.items():
                            if str(valve_name).strip().lower() == "total":
                                continue
                            if valve_name in Open.columns:
                                EdgeOpen[:, edge_idx] = Open[valve_name].values / 100.0
                                _log(f"[Dataset] Edge {edge_idx} ({valve_name}) opening values loaded")
                    else:
                        for s, ei in sensor2edge_idx.items():
                            if str(s).strip().lower() == "total":
                                continue
                            if s in Open.columns:
                                EdgeOpen[:, ei] = np.divide(Open[s].to_numpy(dtype=float), 100.0)
                                _log(f"[Dataset] Edge {ei} ({s}) opening values loaded")
                    self.OpenEdge = EdgeOpen

                    # OpenNode: [T, N], OpenEdge: [T, E]
                    try:
                        self.OpenNode_tensor = torch.from_numpy(
                            self.OpenNode.to_numpy(dtype=np.float32, copy=True)
                        ).to(torch.float32)
                    except Exception:
                        self.OpenNode_tensor = None

                    self.OpenEdge_tensor = torch.from_numpy(
                        np.asarray(self.OpenEdge, dtype=np.float32)
                    ).to(torch.float32)

                    self.has_open = True
                    _log(f"[Dataset] Opening data loaded. Nodes with variable openings: {(self.OpenNode != 1.0).any().sum()}")
                else:
                    _log(f"[Dataset] Warning: No valve opening columns found.")
            else:
                _log(f"[Dataset] Warning: Valve opening file not found: {open_paths}")

        assert self.g.edge_index is not None
        assert self.g.edge_attr is not None
        E = self.g.edge_index.size(1)
        self.edge_mean_m3ph = np.zeros(E, dtype=float)
        self.edge_std_m3ph = np.ones(E, dtype=float)

        diameters = self.g.edge_attr[:, 1].numpy()

        # Q = v * A = v * π * (d/2)^2
        design_velocity = 1.0
        nominal_Q_m3s = design_velocity * np.pi * np.divide(diameters, 2.0) ** 2
        nominal_Q_m3ph = nominal_Q_m3s * 3600.0

        self.edge_mean_m3ph = nominal_Q_m3ph.copy()
        self.edge_std_m3ph = nominal_Q_m3ph / 3.0
        self.edge_std_m3ph = np.maximum(self.edge_std_m3ph, 1.0)

        for s in self.edge_meas_cols:
            ei = self.edge_meas_map[s]
            if s in means and s in stds:
                self.edge_mean_m3ph[ei] = float(means[s])
                std_val = stds[s]
                self.edge_std_m3ph[ei] = max(float(std_val) if std_val else 1.0, 1e-6)
                _log(f"[Dataset] Edge {ei} ({s}): mean={means[s]:.2f}, std={float(std_val):.4f}")

        is_self = (self.g.edge_attr[:, 5].numpy() > 0.5)
        self.edge_mean_m3ph[is_self] = 0.0
        self.edge_std_m3ph[is_self] = 1.0

        if hasattr(self.g, 'node_type'):
            self.node_type = self.g.node_type
        else:
            self.node_type = self._infer_node_types()

        self.random_mask_range = random_mask_range
        self.valid_t = list(range(self.window - 1, len(self.time_index)))

        self.has_credibility = False
        self.cred_weight_by_t: Dict[int, float] = {}
        self.cred_level_by_t: Dict[int, int] = {}
        self.total_anchor_gate_level = 0

        if credibility_path:
            cred_path = Path(credibility_path)
            if cred_path.exists():
                cred_df = pd.read_csv(cred_path)
                if "t_index" not in cred_df.columns:
                    raise ValueError(f"Missing t_index column in the credibility file: {cred_path}")
                if "credibility" not in cred_df.columns:
                    raise ValueError(f"Missing credibility column in the credibility file: {cred_path}")
                if "weight" not in cred_df.columns:
                    raise ValueError(f"Missing weight column in the credibility file: {cred_path}")

                def _level_from_label(lbl: str) -> int:
                    s = str(lbl).strip().lower()
                    if s == "high":
                        return 2
                    if s == "mid":
                        return 1
                    return 0

                for row in cred_df.itertuples(index=False):
                    t_idx = int(getattr(row, "t_index"))
                    lvl = _level_from_label(getattr(row, "credibility"))
                    w = float(getattr(row, "weight"))
                    self.cred_level_by_t[t_idx] = lvl
                    self.cred_weight_by_t[t_idx] = w

                allowed_t = set(self.cred_level_by_t.keys())
                before = len(self.valid_t)
                self.valid_t = [t for t in self.valid_t if int(t) in allowed_t]
                after = len(self.valid_t)

                self.has_credibility = True
                _log(
                    f"[Dataset] Credibility filter enabled: entries={len(allowed_t)}, valid_t {before} -> {after} (excluding incomplete or all-zero windows)"
                )
            else:
                _log(f"[Dataset] ⚠️ Credibility file missing, skipping the filter: {cred_path}")

        self.edge_mean_m3ph = torch.tensor(self.edge_mean_m3ph, dtype=torch.float32)
        self.edge_std_m3ph = torch.tensor(self.edge_std_m3ph, dtype=torch.float32)

        self.has_pressure = False
        self.P_tensor = None
        self.P_mask_tensor = None

        if pressure_head_path and Path(pressure_head_path).exists():
            self.has_pressure = True
            _log(f"[Dataset] Loading pressure-head data: {pressure_head_path}")

            pressure_df = pd.read_csv(pressure_head_path, index_col=0, parse_dates=True)

            pressure_df = pressure_df.reindex(self.time_index)

            N = self.g.num_nodes
            assert N is not None
            P_np = np.full((len(self.time_index), N), np.nan, dtype=np.float32)

            for col in pressure_df.columns:
                sensor_name = col.replace("PH_", "")
                if sensor_name in sensor2node_idx:
                    node_idx = sensor2node_idx[sensor_name]
                    P_np[:, node_idx] = pressure_df[col].to_numpy(dtype=np.float32)

            self.P_tensor = torch.from_numpy(P_np).to(torch.float32)
            self.P_mask_tensor = ~torch.isnan(self.P_tensor)

            _log(f"[Dataset] Pressure-head data loaded. Valid observation proportion: {self.P_mask_tensor.float().mean().item():.2%}")

        _log(f"[Dataset] Initialisation complete: {len(self.valid_t)} valid samples")

    def _infer_node_types(self):
        """Infer intermediate, source and terminal node types from topology."""
        N = self.g.num_nodes
        assert N is not None
        ei = self.g.edge_index
        assert ei is not None
        ea = self.g.edge_attr
        assert ea is not None
        is_self = ea[:, 5] > 0.5
        ei_real = ei[:, ~is_self]

        src, dst = ei_real
        has_in = torch.zeros(N, dtype=torch.bool)
        has_out = torch.zeros(N, dtype=torch.bool)

        has_in.scatter_(0, dst, torch.ones_like(dst, dtype=torch.bool))
        has_out.scatter_(0, src, torch.ones_like(src, dtype=torch.bool))

        node_type = torch.zeros(N, dtype=torch.long)
        node_type[has_out & ~has_in] = 1
        node_type[has_in & ~has_out] = 2

        return node_type

    def __len__(self):
        """Return the number of sample endpoints."""
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        """Return the graph sample and supervision masks at a selected endpoint."""
        t = self.valid_t[idx]
        t0 = t - self.window + 1

        x_flow_window = self.X_tensor[t0:t+1].t().contiguous()  # [N, win]
        input_mask_window = ~torch.isnan(x_flow_window)
        x_flow = torch.nan_to_num(x_flow_window, nan=0.0)

        if self.has_open and (self.OpenNode_tensor is not None):
            # OpenNode_tensor: [T, N] -> slice [win, N] -> transpose [N, win]
            x_open = self.OpenNode_tensor[t0:t+1].t().contiguous()
        elif self.has_open and (self.OpenNode is not None):
            Ow = self.OpenNode.iloc[t0:t+1]
            x_open = torch.tensor(Ow.to_numpy(dtype=np.float32), dtype=torch.float32).t().contiguous()
        else:
            x_open = torch.ones_like(x_flow)

        x_node = torch.cat([x_flow, x_open], dim=1)  # [N, win*2]

        rest_last = self.Rest_tensor[t]
        y_node = self.X_tensor[t]
        node_label_mask = ~torch.isnan(y_node)

        node_inpaint_mask = torch.zeros_like(node_label_mask, dtype=torch.bool)
        if self.random_mask_range is not None:
            lo, hi = self.random_mask_range
            old_state = torch.random.get_rng_state()

            seed_val = idx + int(torch.nan_to_num(y_node).sum().item() * 100)
            torch.manual_seed(seed_val % (2**32 - 1))

            ratio = float(torch.rand(1).item() * (hi - lo) + lo)
            obs_idx = node_label_mask.nonzero(as_tuple=False).view(-1)
            k = int(ratio * len(obs_idx))
            if k > 0:
                perm = torch.randperm(len(obs_idx))
                drop = obs_idx[perm[:k]]
                node_inpaint_mask[drop] = True

            torch.random.set_rng_state(old_state)

        _g_ei = self.g.edge_index
        _g_ea = self.g.edge_attr
        assert _g_ei is not None
        assert _g_ea is not None
        E = _g_ei.size(1)
        y_edge = torch.full((E,), float("nan"), dtype=torch.float32)
        edge_label_mask = torch.zeros(E, dtype=torch.bool)

        for ei, col_idx in self.edge_meas_pairs:
            val = self.Z_tensor[t, col_idx]
            if not torch.isnan(val):
                y_edge[ei] = val
                edge_label_mask[ei] = True

        total_inflow_mask = torch.tensor(False, dtype=torch.bool)
        # Apply the observed inlet flow as a fixed boundary.
        total_inflow_hard_mask = torch.tensor(False, dtype=torch.bool)
        total_inflow_z = torch.tensor(0.0, dtype=torch.float32)
        total_inflow_m3ps = torch.tensor(0.0, dtype=torch.float32)
        if (
            self.total_inflow_sensor is not None
            and self.total_inflow_col_idx is not None
            and self.total_inflow_mean_m3ph is not None
            and self.total_inflow_std_m3ph is not None
        ):
            val = self.Z_tensor[t, self.total_inflow_col_idx]
            if not torch.isnan(val):
                total_inflow_mask = torch.tensor(True, dtype=torch.bool)
                total_inflow_z = val.to(torch.float32)
                total_inflow_m3ps = ((total_inflow_z * self.total_inflow_std_m3ph + self.total_inflow_mean_m3ph) / 3600.0).to(torch.float32)

                total_inflow_hard_mask = torch.tensor(True, dtype=torch.bool)

        cred_level = 2
        cred_weight = 1.0
        if self.has_credibility:
            t_int = int(t)
            cred_level = int(self.cred_level_by_t.get(t_int, 0))
            cred_weight = float(self.cred_weight_by_t.get(t_int, 0.1 if cred_level <= 0 else 1.0))

            if cred_level <= self.total_anchor_gate_level:
                total_inflow_mask = torch.tensor(False, dtype=torch.bool)
                total_inflow_z = torch.tensor(0.0, dtype=torch.float32)
                total_inflow_m3ps = torch.tensor(0.0, dtype=torch.float32)

        total_inflow_hard_mask = total_inflow_hard_mask & total_inflow_mask

        if self.has_open and (self.OpenEdge_tensor is not None):
            edge_opening = self.OpenEdge_tensor[t]
        elif self.has_open and (self.OpenEdge is not None):
            edge_opening = torch.tensor(np.asarray(self.OpenEdge[t], dtype=np.float32), dtype=torch.float32)
        else:
            edge_opening = torch.ones(E, dtype=torch.float32)
            edge_opening[_g_ea[:, 5] > 0.5] = 0.0

        g = Data(
            x=x_node,
            edge_index=_g_ei,
            edge_attr=_g_ea,
            node_elev=self.g.node_elev,
            node_type=self.node_type,
            y_node=y_node.nan_to_num(nan=0.0),
            node_label_mask=node_label_mask,
            node_inpaint_mask=node_inpaint_mask,
            rest_last=rest_last,
            y_edge=y_edge.nan_to_num(nan=0.0),
            edge_label_mask=edge_label_mask,
            input_mask_window=input_mask_window,
            t_index=int(t),
            credibility_level=torch.tensor(int(cred_level), dtype=torch.long),
            credibility_weight=torch.tensor(float(cred_weight), dtype=torch.float32),
            edge_mean=self.edge_mean_m3ph,
            edge_std=self.edge_std_m3ph,
            edge_opening=edge_opening,
            node_mean=self.node_mean_m3ph,
            node_std=self.node_std_m3ph,
            node_has_stats=self.node_has_stats,
            total_inflow_mask=total_inflow_mask,
            total_inflow_hard_mask=total_inflow_hard_mask,
            total_inflow_z=total_inflow_z,
            total_inflow_m3ps=total_inflow_m3ps,
            total_inflow_edge_idx=int(getattr(self.g, 'total_inflow_edge_idx', -1)),
        )

        if self.has_pressure and self.P_tensor is not None and self.P_mask_tensor is not None:
            y_press = self.P_tensor[t]
            # Identify prolonged missing runs as rest periods.
            press_label_mask = self.P_mask_tensor[t] & (~rest_last)

            g.y_press = y_press.nan_to_num(nan=0.0)
            g.press_label_mask = press_label_mask
        g.num_nodes = self.g.num_nodes

        return g
