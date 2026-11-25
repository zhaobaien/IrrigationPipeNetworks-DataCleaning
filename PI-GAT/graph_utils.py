from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _to_float(x, default=np.nan):
    """Convert a value to float, returning the default on failure."""
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip())
    except Exception:
        return default


def compute_hw_weight(
    L: torch.Tensor, d: torch.Tensor, C: torch.Tensor,
    mode: str = "conductance", eps: float = 1e-9
) -> torch.Tensor:
    """Compute a Hazen–Williams pipe weight from length, diameter and roughness.

    The mode selects the conductance feature or the legacy loss weight.
    Clamp physical inputs to eps to avoid division by zero."""
    C = torch.clamp(C, min=eps)
    d = torch.clamp(d, min=eps)
    L = torch.clamp(L, min=eps)

    if mode == "loss":
        return 4.727 * (C ** 1.852) * (d ** 4.871) / L
    return (C ** 1.852) * (d ** 4.871) / (L ** 1.852)


def _build_node_name_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Map raw endpoint names or numeric IDs to the canonical graph node names."""
    mapping = {}

    for _, r in df.iterrows():
        node_name = str(r["Node"]).strip()
        if not node_name:
            continue

        if node_name.lower() == "reservoir":
            mapping["Reservoir"] = "Reservoir"
            mapping["reservoir"] = "Reservoir"
            continue

        if node_name.startswith("J-"):
            try:
                num_part = node_name[2:]
                mapping[num_part] = node_name
                mapping[node_name] = node_name
            except:
                mapping[node_name] = node_name
        else:
            mapping[node_name] = node_name

    return mapping


def build_graph_from_csv(
    csv_path: str | Path,
    weight_mode: str = "conductance",
    add_self_loops: bool = True,
    verbose: bool = True,
) -> Tuple[Data, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Build a directed PyTorch Geometric graph from the topology CSV.

    Return the graph and the node, pipe and sensor-to-edge index mappings.
    Optional self-loops are used by the graph encoder."""
    df = pd.read_csv(csv_path).fillna("")

    required_cols = ["Node", "Elevation", "Pipe", "In", "Out", "Length", "Diameter", "C"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    name_map = _build_node_name_mapping(df)

    all_nodes = set()

    for _, r in df.iterrows():
        n = str(r["Node"]).strip()
        if n:
            all_nodes.add(n)

    for _, r in df.iterrows():
        pin = str(r["In"]).strip()
        pout = str(r["Out"]).strip()
        if pin:
            mapped_in = name_map.get(pin, pin)
            all_nodes.add(mapped_in)
        if pout:
            mapped_out = name_map.get(pout, pout)
            all_nodes.add(mapped_out)

    all_nodes.discard("")

    # Keep the source first and sort the remaining node names for checkpoint compatibility.
    node_names = sorted(all_nodes, key=lambda x: (0 if x.lower() == "reservoir" else 1, x))
    node2idx = {n: i for i, n in enumerate(node_names)}

    def _log(msg: str):
        if verbose:
            print(msg)

    _log(f"[graph_utils] Detected {len(node_names)} nodes: {node_names}")

    elev_map: Dict[str, float] = {}
    for _, r in df.iterrows():
        n = str(r["Node"]).strip()
        if n:
            elev_map[n] = _to_float(r["Elevation"], default=np.nan)

    for raw_name, mapped_name in name_map.items():
        if mapped_name in elev_map and raw_name not in elev_map:
            elev_map[raw_name] = elev_map[mapped_name]

    node_elev = np.array([elev_map.get(n, np.nan) for n in node_names], dtype=float)
    median_elev = float(np.nanmedian(node_elev)) if not np.all(np.isnan(node_elev)) else 350.0
    node_elev = torch.tensor(np.nan_to_num(node_elev, nan=median_elev), dtype=torch.float)

    _log(f"[graph_utils] Node elevation range: {node_elev.min():.2f} - {node_elev.max():.2f} m")

    edges: List[Tuple[int, int]] = []
    eL, ed, eC, ePipe, dz = [], [], [], [], []

    for _, r in df.iterrows():
        pin_raw = str(r["In"]).strip()
        pout_raw = str(r["Out"]).strip()
        pipe = str(r["Pipe"]).strip()

        if not (pin_raw and pout_raw and pipe):
            continue

        # Resolve the pipe endpoints to the canonical graph nodes.
        pin = name_map.get(pin_raw, pin_raw)
        pout = name_map.get(pout_raw, pout_raw)

        if pin not in node2idx or pout not in node2idx:
            _log(f"[graph_utils] Warning: Skipping edge {pipe} ({pin_raw}->{pout_raw}) because a node is missing")
            continue

        u = node2idx[pin]
        v = node2idx[pout]

        L = _to_float(r["Length"], 100.0)
        d = _to_float(r["Diameter"], 0.4)
        C = _to_float(r["C"], 130.0)

        zu = float(node_elev[u])
        zv = float(node_elev[v])
        dz_val = zu - zv

        edges.append((u, v))
        eL.append(L)
        ed.append(d)
        eC.append(C)
        ePipe.append(pipe)
        dz.append(dz_val)

        _log(
            f"[graph_utils] Edge {pipe}: {pin}({zu:.1f}m) -> {pout}({zv:.1f}m), "
            f"L={L:.1f}m, D={d:.3f}m, dz={dz_val:.2f}m"
        )

    if not edges:
        raise ValueError("No edges constructed from CSV. Check 'Pipe','In','Out' columns.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    eL = torch.tensor(eL, dtype=torch.float)
    ed = torch.tensor(ed, dtype=torch.float)
    eC = torch.tensor(eC, dtype=torch.float)
    dz = torch.tensor(dz, dtype=torch.float)

    w = compute_hw_weight(eL, ed, eC, mode=weight_mode)
    is_self = torch.zeros(len(edges), dtype=torch.float)

    edge_attr = torch.stack([eL, ed, eC, w, dz, is_self], dim=1)  # [E,6]

    pipe2idx = {name: i for i, name in enumerate(ePipe)}

    N = len(node_names)

    src_nodes, dst_nodes = edge_index[0], edge_index[1]

    has_in = torch.zeros(N, dtype=torch.bool)
    has_out = torch.zeros(N, dtype=torch.bool)

    has_out.index_fill_(0, src_nodes, True)
    has_in.index_fill_(0, dst_nodes, True)

    node_type = torch.zeros(N, dtype=torch.long)

    node_type[has_out & ~has_in] = 1
    node_type[has_in & ~has_out] = 2

    unique, counts = torch.unique(node_type, return_counts=True)
    type_stats = dict(zip(unique.tolist(), counts.tolist()))
    _log(f"[graph_utils] Node type counts: {type_stats}")
    _log(f"  - Intermediate node (Type 0): {type_stats.get(0, 0)} (typically the majority)")
    _log(f"  - Source node   (Type 1): {type_stats.get(1, 0)}")
    _log(f"  - Terminal node (Type 2): {type_stats.get(2, 0)}")

    if type_stats.get(0, 0) == 0:
        _log("[graph_utils] ⚠️ Warning: No intermediate nodes detected. Check the graph connectivity.")

    type_names = {0: "Intermediate", 1: "Source", 2: "Terminal"}
    for i, n in enumerate(node_names):
        _log(f"[graph_utils] Node {n}: type={type_names[int(node_type[i].item())]}, elevation={node_elev[i]:.2f}m")

    sensor2edge_idx: Dict[str, int] = {}
    sensor2node_idx: Dict[str, int] = {}

    endnode_map: Dict[str, str] = {}
    for _, r in df.iterrows():
        valve = str(r.get("Valve", "")).strip()
        endnode = str(r.get("EndNode", "")).strip()
        if valve and endnode:
            endnode_map[valve] = endnode

    for _, r in df.iterrows():
        valve = str(r.get("Valve", "")).strip()
        if not valve:
            continue

        cp = str(r.get("Corresponding pipeline", "")).strip()
        if cp and cp in pipe2idx:
            ei = pipe2idx[cp]
            sensor2edge_idx[valve] = ei

            # The trunk meter provides an inlet boundary rather than a node label.
            if valve.strip().lower() == "total":
                _log(f"[graph_utils] Sensor {valve}: edge={cp}, node=SKIP (global inflow only)")
                continue

            if valve in endnode_map:
                endnode = endnode_map[valve]
                if endnode in node2idx:
                    sensor2node_idx[valve] = node2idx[endnode]
                else:
                    for nn in node_names:
                        if nn == endnode or nn.endswith(f"-{endnode}") or endnode.endswith(nn.split("-")[-1] if "-" in nn else ""):
                            sensor2node_idx[valve] = node2idx[nn]
                            break

            if valve not in sensor2node_idx:
                v = edge_index[1, ei].item()
                sensor2node_idx[valve] = int(v)

        _log(f"[graph_utils] Sensor {valve}: edge={cp}, node index={sensor2node_idx.get(valve, 'N/A')}")

    n_real_edges = len(edges)
    if add_self_loops:
        self_src = torch.arange(N, dtype=torch.long)
        self_dst = torch.arange(N, dtype=torch.long)
        self_index = torch.stack([self_src, self_dst], dim=0)

        edge_index = torch.cat([edge_index, self_index], dim=1)

        sl = torch.zeros(N, dtype=torch.float)
        sd = torch.ones(N, dtype=torch.float)
        sC = torch.ones(N, dtype=torch.float)
        sw = torch.ones(N, dtype=torch.float)
        sdz = torch.zeros(N, dtype=torch.float)
        sis = torch.ones(N, dtype=torch.float)
        self_attr = torch.stack([sl, sd, sC, sw, sdz, sis], dim=1)
        edge_attr = torch.cat([edge_attr, self_attr], dim=0)
        ePipe += [f"__self_{i}" for i in range(N)]

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N
    )
    data.node_elev = node_elev
    data.node_names = node_names
    data.node_type = node_type
    data.pipe_names = ePipe
    data.sensor2edge_idx = sensor2edge_idx
    data.sensor2node_idx = sensor2node_idx
    data.total_inflow_sensor = "Total" if "Total" in sensor2edge_idx else ""
    data.total_inflow_edge_idx = int(sensor2edge_idx.get("Total", -1))
    data.n_real_edges = n_real_edges
    data.name_map = name_map

    _log(f"[graph_utils] Graph built: {N} nodes, {n_real_edges} physical edges, {len(sensor2edge_idx)} sensors")

    return data, node2idx, pipe2idx, sensor2edge_idx
