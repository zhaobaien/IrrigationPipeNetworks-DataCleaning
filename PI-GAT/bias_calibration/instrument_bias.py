from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Dict, List

import copy
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

import sys

CURRENT_DIR = Path(__file__).parent
PI_GAT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PI_GAT_DIR))

from graph_utils import build_graph_from_csv
from model_pi_gat import PI_GAT
from timeseries_dataset import IrrigationTimeWindowDataset
import functools


def _worker_init_fn_impl(worker_id: int, seed: int) -> None:
    import random as _r
    ws = seed + worker_id
    _r.seed(ws)
    np.random.seed(ws)
    torch.manual_seed(ws)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def export_calibration_params_json(
    model: nn.Module,
    data_graph: Any,
    out_path: Path,
    *,
    tag: str,
    loss: float,
    best_val_loss: float,
    ckpt_path: str,
):
    """Write the fitted meter scales and frozen Stage-I leakage estimate.

    The JSON layout is the calibration input for effective-opening mapping
    and Stage-III cleaning."""

    orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    orig = cast(Any, orig)

    leakage = float(orig.get_leakage_rate().detach().cpu().item()) if hasattr(orig, "get_leakage_rate") else 0.0

    delta = None
    if hasattr(orig, "get_bias_params"):
        _, delta = orig.get_bias_params()
        delta = delta.detach().cpu()

    bin_names = ["6-33%", "34-66%", "67-100%"]

    valves: Dict[str, Any] = {}
    sensor2node = getattr(data_graph, "sensor2node_idx", {}) or {}
    node_names = getattr(data_graph, "node_names", None)
    node_type = getattr(data_graph, "node_type", None)

    for valve_name, node_idx in sensor2node.items():
        if str(valve_name).strip().lower() == "total":
            continue
        try:
            node_idx_int = int(node_idx)
        except Exception:
            continue

        if isinstance(node_type, torch.Tensor):
            if node_idx_int < 0 or node_idx_int >= int(node_type.numel()):
                continue
            if int(node_type[node_idx_int].item()) != 2:
                continue

        node_name = None
        if isinstance(node_names, list) and 0 <= node_idx_int < len(node_names):
            node_name = node_names[node_idx_int]

        entry: Dict[str, Any] = {
            "node_idx": node_idx_int,
            "node_name": node_name,
        }

        if (
            delta is not None
            and isinstance(delta, torch.Tensor)
            and delta.ndim == 2
            and int(delta.shape[1]) == 3
            and node_idx_int < int(delta.shape[0])
        ):
            d = delta[node_idx_int].flatten()
            d_list = [float(x) for x in d.tolist()]
            bias_list = d_list
            scale_list = [float(1.0 + b) for b in bias_list]
            entry.update({
                "delta": d_list,
                "bias": bias_list,
                "scale": scale_list,
            })

        valves[str(valve_name)] = entry

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "loss": float(loss),
            "best_val_loss": float(best_val_loss),
            "ckpt": str(ckpt_path),
        },
        "leakage": {
            "rate": float(leakage),
            "range": [0.05, 0.15],
        },
        "bias": {
            "bin_names": bin_names,
            "valves": valves,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_model_weights(model: nn.Module, ckpt_path: Path) -> None:
    # Mark as Any to avoid static-analysis errors about unknown checkpoint types
    obj: Any = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        state: Any = obj["model_state"]
    else:
        state = obj

    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        pass

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = str(k)
            for prefix in ("module.", "_orig_mod."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
        return

    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def _set_requires_grad_bias_only(model: nn.Module, *, train_mu: bool, train_delta: bool) -> None:
    orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    orig = cast(Any, orig)
    for name, p in orig.named_parameters():
        if "bias_mu_raw" in name:
            p.requires_grad = bool(train_mu)
        elif "bias_delta_raw" in name:
            p.requires_grad = bool(train_delta)
        else:
            p.requires_grad = False


def _load_credibility_table(credibility_csv: Path) -> Dict[int, str]:
    """Map time indices to the credibility labels stored in the CSV."""
    if not credibility_csv.exists():
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(credibility_csv)
        if "t_index" not in df.columns or "credibility" not in df.columns:
            return {}
        return {int(t): str(c).strip().lower() for t, c in zip(df["t_index"].tolist(), df["credibility"].tolist())}
    except Exception:
        import csv
        out: Dict[int, str] = {}
        with open(credibility_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row["t_index"])] = str(row.get("credibility", "")).strip().lower()
                except Exception:
                    continue
        return out


def _make_high_subset_indices(dataset: Any, cred_by_t: Dict[int, str]) -> List[int]:
    """Return dataset indices of the high-credibility windows."""
    out: List[int] = []
    for di in range(len(dataset)):
        try:
            t = int(dataset.valid_t[int(di)])
        except Exception:
            continue
        if cred_by_t.get(t, "") == "high":
            out.append(di)
    return out


def _bias_fit_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    *,
    use_leakage_rate: bool = True,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_weight = 0.0
    total_n = 0
    total_used = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        mask = batch.node_label_mask & (~batch.rest_last) & (batch.node_type == 2)
        n = int(mask.sum().item())
        total_used += n
        total_n += int(batch.y_node.numel())
        if n <= 0:
            continue

        obs_z = batch.y_node[mask].to(torch.float32)
        node_mean = batch.node_mean[mask].to(torch.float32)
        node_std = batch.node_std[mask].to(torch.float32)
        obs_m3ps = torch.clamp(obs_z * node_std + node_mean, min=0.0) / 3600.0

        meter_scale = out.get("meter_scale", None)
        if meter_scale is None:
            meter_scale = torch.ones_like(obs_m3ps)
        else:
            meter_scale = torch.clamp(meter_scale[mask].to(torch.float32), min=1e-6)
        true_m3ps = obs_m3ps / meter_scale

        if (not hasattr(batch, "total_inflow_m3ps")) or (not hasattr(batch, "total_inflow_mask")):
            continue

        total_in_m3ps = batch.total_inflow_m3ps.to(torch.float32).view(-1)
        total_mask = batch.total_inflow_mask.to(torch.bool).view(-1)
        num_graphs = int(getattr(batch, "num_graphs", int(total_in_m3ps.numel())))

        g_idx = batch.batch[mask]
        sum_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
        sum_true.index_add_(0, g_idx, true_m3ps)
        cnt_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
        cnt_true.index_add_(0, g_idx, torch.ones_like(true_m3ps))

        if use_leakage_rate:
            leak_rate = out.get("leakage_rate", None)
            leak_r = float(leak_rate.detach().mean().cpu().item()) if isinstance(leak_rate, torch.Tensor) else 0.0
            leak_r = float(np.clip(leak_r, 0.0, 0.5))
        else:
            leak_r = 0.0

        target = total_in_m3ps[:num_graphs] * (1.0 - leak_r)
        valid_g = total_mask[:num_graphs] & (target > 1e-6) & (cnt_true > 0.5)
        if not bool(valid_g.any().item()):
            continue

        rel = (sum_true[valid_g] - target[valid_g]) / torch.clamp(target[valid_g], min=1e-6)
        loss = torch.mean(rel ** 2)
        n_valid = float(valid_g.sum().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * n_valid
        total_weight += n_valid

    avg_loss = total_loss / max(1e-12, total_weight)
    keep_rate = float(total_used) / float(max(1, total_n))
    return {"loss": float(avg_loss), "keep_rate": keep_rate}


def main():
    parser = argparse.ArgumentParser(
        description="Fit instrument bias with Stage-I model frozen (bias-only optimisation, high-credibility data only)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--ckpt", type=str, default="pi_gat_best.pth",
        help="Stage-I trained weights: pi_gat_best.pth / pi_gat_final.pth / pi_gat_ckpt.pth",
    )
    parser.add_argument(
        "--out-json", type=str, default="calibration_params_final.json",
        help="Output JSON filename (created in bias_calibration dir by default)",
    )
    parser.add_argument(
        "--out-ckpt", type=str, default="instrument_bias_best.pth",
        help="Output checkpoint filename (created in bias_calibration dir by default)",
    )
    parser.add_argument("--epochs-delta", type=int, default=200)
    parser.add_argument("--lr-delta", type=float, default=1e-2)
    parser.add_argument(
        "--min-delta", type=float, default=1e-5,
        help="Minimum reduction in validation loss required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--no-use-leakage-rate", action="store_true",
        help="Use the observed Total as the target without a Stage-I leakage correction.",
    )
    parser.add_argument(
        "--fixed-leakage", type=float, default=None,
        help="Fixed leakage rate shared with Stage I.",
    )

    args = parser.parse_args()

    if platform.system() == "Windows" and args.num_workers > 0:
        print(f"[BiasFit] On Windows, num_workers {args.num_workers} → 0 (to avoid multiprocessing spawn overhead)")
        args.num_workers = 0

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = PI_GAT_DIR
    preprocess_dir = base_dir.parent / "Preprocess_Timeseries" / "outputs" / "preprocessed"

    topo_csv = base_dir.parent / "network_topology.csv"
    zscore_csv = preprocess_dir / "flow_5min_zscore.csv"
    restmask_csv = preprocess_dir / "mask_rest.csv"
    zscore_stats = preprocess_dir / "zscore_stats.json"
    open_csv = preprocess_dir / "flow_5min_imputed.csv"
    credibility_csv = base_dir / "preprocessing" / "credibility_windows.csv"

    data_graph, _, _, sensor2edge_idx = build_graph_from_csv(
        topo_csv, weight_mode="conductance", add_self_loops=True
    )
    sensor2node_idx = data_graph.sensor2node_idx

    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=zscore_csv,
        restmask_path=restmask_csv,
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=int(args.window),
        random_mask_range=None,
        open_paths=open_csv,
        zscore_stats_path=zscore_stats,
        credibility_path=(credibility_csv if credibility_csv.exists() else None),
        verbose=False,
    )

    cred_by_t = _load_credibility_table(credibility_csv)
    idx_sel = _make_high_subset_indices(dataset, cred_by_t)
    if len(idx_sel) == 0:
        raise RuntimeError(
            "No high-credibility time windows were found. "
            "Run preprocessing/preprocess_imputed.py to generate credibility_windows.csv."
        )

    rng = random.Random(int(args.seed) + 123)
    idx_s = idx_sel.copy()
    rng.shuffle(idx_s)
    n_val = max(1, int(0.2 * len(idx_s)))
    val_idx = idx_s[:n_val]
    train_idx = idx_s[n_val:]

    _train_g = torch.Generator()
    _train_g.manual_seed(int(args.seed))
    _worker_init_fn = functools.partial(_worker_init_fn_impl, seed=int(args.seed))

    train_loader = DataLoader(
        cast(Any, torch.utils.data.Subset(dataset, train_idx)),
        batch_size=int(args.batch_size), shuffle=True,
        num_workers=int(args.num_workers), pin_memory=(device == "cuda"),
        generator=_train_g, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        cast(Any, torch.utils.data.Subset(dataset, val_idx)),
        batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=_worker_init_fn,
    )

    assert data_graph.num_nodes is not None
    model: Any = PI_GAT(
        win=int(args.window), num_nodes=int(data_graph.num_nodes),
        hidden=64, heads=8, dropout=0.1, in_feat=2,
    ).to(device)

    if args.fixed_leakage is not None:
        if hasattr(model, '_orig_mod'):
            model._orig_mod._leakage_range = (args.fixed_leakage, args.fixed_leakage)
        else:
            model._leakage_range = (args.fixed_leakage, args.fixed_leakage)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _load_model_weights(cast(nn.Module, model), ckpt_path)

    use_leakage = not bool(args.no_use_leakage_rate)
    print(f"[BiasFit] device={device}, high windows={len(idx_sel)} (train={len(train_idx)}, val={len(val_idx)})")

    _set_requires_grad_bias_only(model, train_mu=False, train_delta=True)
    opt_delta = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr_delta), weight_decay=0.0)
    min_delta = float(args.min_delta)
    print(f"[BiasFit] Fitting node delta, epochs={args.epochs_delta}, lr={args.lr_delta:.2e}, patience=10, min_delta={min_delta:.1e}")

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    for ep in range(1, int(args.epochs_delta) + 1):
        t0 = time.time()
        train_metrics = _bias_fit_one_epoch(model, train_loader, device, opt_delta, use_leakage_rate=use_leakage)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_weight = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)

                if (not hasattr(batch, "total_inflow_m3ps")) or (not hasattr(batch, "total_inflow_mask")):
                    continue

                mask = batch.node_label_mask & (~batch.rest_last) & (batch.node_type == 2)
                if int(mask.sum().item()) <= 0:
                    continue

                obs_z = batch.y_node[mask].to(torch.float32)
                node_mean = batch.node_mean[mask].to(torch.float32)
                node_std = batch.node_std[mask].to(torch.float32)
                obs_m3ps = torch.clamp(obs_z * node_std + node_mean, min=0.0) / 3600.0

                meter_scale = out.get("meter_scale", None)
                if meter_scale is None:
                    meter_scale = torch.ones_like(obs_m3ps)
                else:
                    meter_scale = torch.clamp(meter_scale[mask].to(torch.float32), min=1e-6)
                true_m3ps = obs_m3ps / meter_scale

                total_in_m3ps = batch.total_inflow_m3ps.to(torch.float32).view(-1)
                total_mask = batch.total_inflow_mask.to(torch.bool).view(-1)
                num_graphs = int(getattr(batch, "num_graphs", int(total_in_m3ps.numel())))

                g_idx = batch.batch[mask]
                sum_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
                sum_true.index_add_(0, g_idx, true_m3ps)
                cnt_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
                cnt_true.index_add_(0, g_idx, torch.ones_like(true_m3ps))

                if use_leakage:
                    leak_rate = out.get("leakage_rate", None)
                    leak_r = float(leak_rate.detach().mean().cpu().item()) if isinstance(leak_rate, torch.Tensor) else 0.0
                    leak_r = float(np.clip(leak_r, 0.0, 0.5))
                else:
                    leak_r = 0.0

                target = total_in_m3ps[:num_graphs] * (1.0 - leak_r)
                valid_g = total_mask[:num_graphs] & (target > 1e-6) & (cnt_true > 0.5)
                if not bool(valid_g.any().item()):
                    continue

                rel = (sum_true[valid_g] - target[valid_g]) / torch.clamp(target[valid_g], min=1e-6)
                loss = torch.mean(rel ** 2)

                val_loss_sum += float(loss.detach().cpu().item())
                val_weight += 1.0

            current_val_loss = val_loss_sum / max(1e-12, val_weight)

        if ep == 1 or ep % 10 == 0 or ep == int(args.epochs_delta):
            print(f"  [delta {ep:03d}] train_loss={train_metrics['loss']:.6f}, val_loss={current_val_loss:.6f}, keep={train_metrics['keep_rate']:.2%}, time={time.time()-t0:.1f}s")

        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
            orig = model._orig_mod if hasattr(model, "_orig_mod") else model
            orig = cast(Any, orig)
            best_model_state = copy.deepcopy(orig.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[BiasFit] Early stopping at epoch {ep}, best val loss: {best_val_loss:.6f}")
                break

    if best_model_state is not None:
        orig = model._orig_mod if hasattr(model, "_orig_mod") else model
        orig = cast(Any, orig)
        orig.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        val_loss_sum = 0.0
        val_weight = 0.0
        val_resid_sum = 0.0
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)

            if (not hasattr(batch, "total_inflow_m3ps")) or (not hasattr(batch, "total_inflow_mask")):
                continue

            mask = batch.node_label_mask & (~batch.rest_last) & (batch.node_type == 2)
            if int(mask.sum().item()) <= 0:
                continue

            obs_z = batch.y_node[mask].to(torch.float32)
            node_mean = batch.node_mean[mask].to(torch.float32)
            node_std = batch.node_std[mask].to(torch.float32)
            obs_m3ps = torch.clamp(obs_z * node_std + node_mean, min=0.0) / 3600.0

            meter_scale = out.get("meter_scale", None)
            if meter_scale is None:
                meter_scale = torch.ones_like(obs_m3ps)
            else:
                meter_scale = torch.clamp(meter_scale[mask].to(torch.float32), min=1e-6)
            true_m3ps = obs_m3ps / meter_scale

            total_in_m3ps = batch.total_inflow_m3ps.to(torch.float32).view(-1)
            total_mask = batch.total_inflow_mask.to(torch.bool).view(-1)
            num_graphs = int(getattr(batch, "num_graphs", int(total_in_m3ps.numel())))

            g_idx = batch.batch[mask]
            sum_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
            sum_true.index_add_(0, g_idx, true_m3ps)
            cnt_true = torch.zeros(num_graphs, device=true_m3ps.device, dtype=torch.float32)
            cnt_true.index_add_(0, g_idx, torch.ones_like(true_m3ps))

            if use_leakage:
                leak_rate = out.get("leakage_rate", None)
                leak_r = float(leak_rate.detach().mean().cpu().item()) if isinstance(leak_rate, torch.Tensor) else 0.0
                leak_r = float(np.clip(leak_r, 0.0, 0.5))
            else:
                leak_r = 0.0

            target = total_in_m3ps[:num_graphs] * (1.0 - leak_r)
            valid_g = total_mask[:num_graphs] & (target > 1e-6) & (cnt_true > 0.5)
            if not bool(valid_g.any().item()):
                continue

            resid = sum_true[valid_g] - target[valid_g]
            rel = resid / torch.clamp(target[valid_g], min=1e-6)
            loss = torch.mean(rel ** 2)

            val_loss_sum += float(loss.detach().cpu().item())
            val_weight += 1.0
            val_resid_sum += float(resid.mean().detach().cpu().item())

        val_loss = val_loss_sum / max(1e-12, val_weight)
        val_resid_mean = val_resid_sum / max(1e-12, val_weight)

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = CURRENT_DIR / out_json
    out_ckpt = Path(args.out_ckpt)
    if not out_ckpt.is_absolute():
        out_ckpt = CURRENT_DIR / out_ckpt

    export_calibration_params_json(
        cast(nn.Module, model), data_graph, out_json,
        tag="instrument_bias_total_anchor", loss=float(val_loss), best_val_loss=float(best_val_loss), ckpt_path=str(ckpt_path),
    )
    orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    orig = cast(Any, orig)
    torch.save(orig.state_dict(), out_ckpt)

    print(f"[BiasFit] val_loss={val_loss:.6f} (graphs={int(val_weight)})")
    print(f"[BiasFit] Mean corrected closure residual: {val_resid_mean*3600:.4f} m³/h")
    print(f"[BiasFit] exported JSON: {out_json}")
    print(f"[BiasFit] exported checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
