from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import os
import sys
import numpy as np
import time
import json
from datetime import datetime
import argparse
from typing import Any, cast

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch_geometric.loader import DataLoader
from graph_utils import build_graph_from_csv
from timeseries_dataset import IrrigationTimeWindowDataset
from model_pi_gat import PI_GAT
import platform
import functools

os.environ["TQDM_DISABLE"] = "False"
os.environ["TQDM_COLOR"] = "True"


def _worker_init_fn_impl(worker_id: int, seed: int) -> None:
    import random as _r
    ws = seed + worker_id
    _r.seed(ws)
    np.random.seed(ws)
    torch.manual_seed(ws)


class PCGrad:
    """Gradient-projection optimiser used in the submitted training."""
    def __init__(self, optimizer):
        self._optim = optimizer
        self.param_groups = self._optim.param_groups

    @property
    def state(self):
        return self._optim.state

    def zero_grad(self, set_to_none=False):
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self):
        self._optim.step()

    def state_dict(self):
        """Return the underlying optimiser state."""
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        """Restore the underlying optimiser state."""
        return self._optim.load_state_dict(state_dict)

    def pc_backward(self, objectives):
        """Project conflicting task gradients and accumulate the submitted update."""
        grads = []
        params = []

        for group in self._optim.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)

            grad_task = []
            for p in params:
                if p.grad is not None:
                    grad_task.append(p.grad.clone().flatten())
                else:
                    grad_task.append(torch.zeros_like(p).flatten())
            grads.append(torch.cat(grad_task))

        _perm = torch.randperm(len(grads)).tolist()
        grads = [grads[_pi] for _pi in _perm]

        num_tasks = len(grads)
        proj_grads = [g.clone() for g in grads]

        for i in range(num_tasks):
            for j in range(int(torch.randint(0, num_tasks + 1, (1,)).item()), num_tasks): # Choose the gradient comparison tasks using the seeded random sequence.
                if i == j: continue

                g_i = proj_grads[i]
                g_j = grads[j]

                inner_product = torch.dot(g_i, g_j)

                if inner_product < 0:
                    proj_grads[i] -= (inner_product / (g_j.norm()**2 + 1e-8)) * g_j

        final_grad = torch.stack(proj_grads).sum(dim=0)

        # Retain the most recent task gradient before adding the projected sum.

        idx = 0
        for p in params:
            num_el = p.numel()
            if p.grad is None:
                 p.grad = torch.zeros_like(p)

            p.grad.add_(final_grad[idx:idx+num_el].view_as(p))
            idx += num_el

        return


class LossPI(nn.Module):
    """Combine the data-fit and physical losses for PI-GAT."""

    def __init__(self):
        """Construct the data-fit loss functions."""
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.huber = nn.HuberLoss(reduction="mean", delta=1.0)

    def __call__(
        self,
        out,
        batch,
        model=None,
        lambda_mass=0.0,
        lambda_energy=0.0,
        lambda_pressure=0.0,
        lambda_buffer=0.0,
        lambda_inflow=0.0,
        epoch=0
    ):
        """Evaluate the weighted losses and return the loss and diagnostic dictionary."""
        node_hat_phys = out["node_hat"]
        node_hat_biased = out.get("node_hat_biased", out["node_hat"])
        edge_hat = out["edge_hat"]
        device = node_hat_phys.device

        has_nan = (torch.isnan(node_hat_phys).any() or
                  torch.isnan(edge_hat).any() or
                  torch.isinf(edge_hat).any())

        has_extreme = (torch.max(torch.abs(node_hat_phys)) > 1e6 or
                      torch.max(torch.abs(edge_hat)) > 1e6)

        if has_nan or has_extreme:
            return {k: torch.tensor(0.0, device=device, requires_grad=True) for k in
                    ["L_total", "L_data", "L_rec_main", "L_rec_inp", "L_edge",
                     "L_mass", "L_energy", "L_pressure", "L_press_data",
                     "L_leakage", "L_global_mass", "L_inflow_obs",
                     "L_total_closure",
                     "L_leak_mass", "L_leak_prior",
                     "L_C_constraint", "L_beta_reg", "L_demand_reg"]}

        mask_obs = batch.node_label_mask & (~batch.rest_last)
        mask_inp = batch.node_inpaint_mask & (~batch.rest_last)

        # - credibility_weight: [num_graphs]
        w_graph = None
        w_node = None
        if hasattr(batch, 'credibility_weight') and hasattr(batch, 'batch'):
            try:
                w_graph = batch.credibility_weight.to(device=device, dtype=torch.float32).view(-1)
                if w_graph.numel() > 0 and isinstance(batch.batch, torch.Tensor) and batch.batch.numel() == batch.y_node.numel():
                    w_node = w_graph[batch.batch].clamp(0.0, 1.0)
            except Exception:
                w_graph = None
                w_node = None
        if w_node is None:
            w_node = torch.ones_like(batch.y_node, dtype=torch.float32, device=device)

        weight_sink = 0.5
        weight_source = 1.0
        weight_mid = 1.0

        weights = torch.ones_like(batch.y_node)

        if hasattr(batch, 'node_type'):
            # node_type: 0=Mid, 1=Source, 2=Sink
            weights[batch.node_type == 2] = weight_sink
            weights[batch.node_type == 1] = weight_source

        if mask_obs.any():
            diff_sq = (node_hat_biased[mask_obs] - batch.y_node[mask_obs]) ** 2

            w_obs = weights[mask_obs].to(device=device, dtype=torch.float32)
            w_cred = w_node[mask_obs].to(device=device, dtype=torch.float32)
            w_total = w_obs * w_cred
            L_rec_main = (diff_sq * w_total).sum() / (w_total.sum() + 1e-9)
        else:
            L_rec_main = torch.tensor(0.0, device=device)

        if mask_inp.any():
            diff_sq = (node_hat_biased[mask_inp] - batch.y_node[mask_inp]) ** 2
            w_total = w_node[mask_inp].to(device=device, dtype=torch.float32)
            L_rec_inp = (diff_sq * w_total).sum() / (w_total.sum() + 1e-9)
        else:
            L_rec_inp = torch.tensor(0.0, device=device)

        mask_edge_valid = batch.edge_label_mask

        if hasattr(batch, 'edge_opening'):
            edge_open = batch.edge_opening
            mask_valve_open = edge_open > 0.05
            mask_edge_valid = batch.edge_label_mask & mask_valve_open

        if mask_edge_valid.any():
            diff_sq = (edge_hat[mask_edge_valid] - batch.y_edge[mask_edge_valid]) ** 2

            w_edge = None
            if w_graph is not None and hasattr(batch, 'edge_index') and hasattr(batch, 'batch'):
                try:
                    edge_graph = batch.batch[batch.edge_index[0]]
                    w_edge = w_graph[edge_graph].clamp(0.0, 1.0)
                except Exception:
                    w_edge = None
            if w_edge is None:
                w_edge = torch.ones_like(batch.y_edge, dtype=torch.float32, device=device)

            w_total = w_edge[mask_edge_valid].to(device=device, dtype=torch.float32)
            L_edge = (diff_sq * w_total).sum() / (w_total.sum() + 1e-9)
        else:
            L_edge = torch.tensor(0.0, device=device)

        L_data = L_rec_main + 0.5 * L_rec_inp + 0.5 * L_edge

        L_mass = torch.tensor(0.0, device=device)
        L_energy = torch.tensor(0.0, device=device)
        L_pressure = torch.tensor(0.0, device=device)
        L_press_data = torch.tensor(0.0, device=device)

        if lambda_mass > 0 or lambda_energy > 0 or lambda_pressure > 0:
            r_mass = out["r_mass"]
            r_energy = out["r_energy"]
            valid_mass_mask = out.get("valid_mass_mask", None)
            p_head = out.get("p_head", None)
            edge_hat_m3ps = out.get("edge_hat_m3ps", None)
            h_loss_pos = out.get("h_loss_pos", None)

            eps = 1e-9
            S_Q = torch.tensor(1.0, device=device)
            S_H = torch.tensor(1.0, device=device)

            is_self = batch.edge_attr[:, 5] > 0.5

            if edge_hat_m3ps is not None:
                q_real = edge_hat_m3ps[~is_self]
                if q_real.numel() > 0:
                    S_Q = torch.median(torch.abs(q_real)).clamp(min=eps)

            if h_loss_pos is not None:
                h_real = None
                if isinstance(h_loss_pos, torch.Tensor):
                    if h_loss_pos.numel() == is_self.numel():
                        h_real = h_loss_pos[~is_self]
                    elif h_loss_pos.numel() == int((~is_self).sum().item()):
                        h_real = h_loss_pos
                if isinstance(h_real, torch.Tensor) and h_real.numel() > 0:
                    S_H = torch.median(torch.abs(h_real)).clamp(min=eps)

            # Normalise the residual before computing its loss.
            if lambda_mass > 0:
                if valid_mass_mask is None:
                    valid_mass_mask = torch.ones_like(r_mass, dtype=torch.bool)
                if isinstance(valid_mass_mask, torch.Tensor) and valid_mass_mask.any():
                    r_mass_scaled = r_mass[valid_mass_mask] / torch.clamp(S_Q, min=1e-6)
                    L_mass = self.huber(r_mass_scaled, torch.zeros_like(r_mass_scaled))

            # Normalise the residual before computing its loss.
            if lambda_energy > 0:
                r_energy_real = r_energy[~is_self]
                if r_energy_real.numel() > 0:
                    r_energy_scaled = r_energy_real / torch.clamp(S_H, min=1e-6)
                    L_energy = self.huber(r_energy_scaled, torch.zeros_like(r_energy_scaled))

            if lambda_pressure > 0 and p_head is not None:
                L_pressure = torch.relu(-p_head + 5.0).mean()

            if hasattr(batch, "press_label_mask") and hasattr(batch, "y_press"):
                mask_p = batch.press_label_mask
                if mask_p.any() and p_head is not None:
                    r = mask_p.float().mean().item()
                    r0 = 0.3
                    lambda_p_eff = lambda_pressure * min(max(r / r0, 0.2), 1.0)

                    p_pred = p_head[mask_p]
                    p_obs = batch.y_press[mask_p]
                    L_press_data = self.huber(p_pred, p_obs) * lambda_p_eff

            out['S_Q'] = S_Q
            out['S_H'] = S_H
            if valid_mass_mask is not None and valid_mass_mask.any():
                out['r_mass_nd'] = r_mass[valid_mass_mask] / S_Q
            out['r_energy_nd'] = r_energy[~is_self] / S_H if r_energy[~is_self].numel() > 0 else r_energy

        L_leakage = torch.tensor(0.0, device=device)

        if hasattr(batch, 'edge_opening') and ('edge_hat_m3ps' in out):
            edge_open = batch.edge_opening
            is_self = batch.edge_attr[:, 5] > 0.5

            mask_closed = (edge_open <= 0.05) & (~is_self)

            q_pred_closed = out['edge_hat_m3ps'][mask_closed]

            if q_pred_closed.numel() > 0:
                S_Q = out.get('S_Q', torch.tensor(1.0, device=device)).detach().clamp(min=1e-6)
                q_nd = q_pred_closed / S_Q
                L_leakage = torch.mean(q_nd ** 2) * 10.0

        # total_system_out ≈ (1 - leakage) * total_system_in
        L_global_mass = torch.tensor(0.0, device=device)
        if ("total_system_in" in out) and ("total_system_out" in out):
            total_in = out["total_system_in"]
            total_out = out["total_system_out"]
            leak = out.get('leakage_rate', torch.tensor(0.10, device=device))

            scale_factor = torch.as_tensor(total_in, device=device).detach().clamp(min=1.0)
            r_global = torch.as_tensor(total_out, device=device) - (1.0 - leak) * torch.as_tensor(total_in, device=device)
            L_global_mass = torch.abs(r_global) / scale_factor

        params = out
        C_main = params['C_main']
        C_branch = params['C_branch']
        alpha = params['alpha']

        alpha_mid = 0.15
        C_main_mid = 130.0
        C_branch_mid = 150.0

        # ============================================================
        # ============================================================

        alpha_gap = torch.relu(alpha_mid - alpha)

        w_c_lock = alpha_gap * 10.0

        c_deviation = torch.relu(C_main_mid - C_main) + torch.relu(C_branch_mid - C_branch)

        L_C_constraint = w_c_lock * (c_deviation ** 2)


        # ============================================================
        # ============================================================

        L_beta_reg = torch.tensor(0.0, device=device)
        orig: Any
        if hasattr(model, '_orig_mod'):
            orig = cast(Any, model)._orig_mod
        else:
            orig = model

        if hasattr(orig, 'resistance_beta_logit'):
            beta = torch.sigmoid(orig.resistance_beta_logit)
            L_beta_reg = calculate_barrier_loss(beta, 0.15, 0.85, k=20.0)

        L_leak_prior = torch.tensor(0.0, device=device)
        if 'leakage_rate' in out:
            leak = out['leakage_rate']
            if isinstance(leak, torch.Tensor):
                lo, hi = 0.05, 0.15
                u = (leak - lo) / (hi - lo)
                u = torch.clamp(u, 1e-4, 1.0 - 1e-4)
                a = 4.0
                b = 4.0
                nll_beta = -((a - 1.0) * torch.log(u) + (b - 1.0) * torch.log(1.0 - u))
                nll_center = ((u - 0.5) / 0.25) ** 2
                L_leak_prior = 0.02 * nll_beta + 0.02 * nll_center


        L_demand_reg = torch.tensor(0.0, device=device)
        if 'latent_demand' in out:
            demand = out['latent_demand']
            L_demand_reg = 0.001 * torch.mean(torch.abs(demand))

        # ============================================================
        L_inflow_obs = torch.tensor(0.0, device=device)
        L_total_closure = torch.tensor(0.0, device=device)
        L_leak_mass = torch.tensor(0.0, device=device)
        inflow_n_total = torch.tensor(0.0, device=device)
        inflow_n_kept = torch.tensor(0.0, device=device)
        inflow_keep_rate = torch.tensor(0.0, device=device)
        inflow_logratio_median = torch.tensor(0.0, device=device)
        if (
            lambda_inflow > 0
            and hasattr(batch, "total_inflow_mask")
            and hasattr(batch, "total_inflow_m3ps")
            and hasattr(batch, "total_inflow_edge_idx")
            and ("edge_hat_m3ps" in out)
            and hasattr(batch, "edge_index")
            and hasattr(batch, "batch")
        ):
            mask = batch.total_inflow_mask
            if isinstance(mask, torch.Tensor) and mask.any():
                if hasattr(batch, "num_graphs") and isinstance(batch.num_graphs, int) and batch.num_graphs > 0:
                    num_graphs = int(batch.num_graphs)
                elif isinstance(batch.batch, torch.Tensor) and batch.batch.numel() > 0:
                    num_graphs = int(batch.batch.max().item()) + 1
                else:
                    num_graphs = 1

                edge_graph = batch.batch[batch.edge_index[0]]  # [E]
                uniq, counts = torch.unique_consecutive(edge_graph, return_counts=True)

                edge_starts = torch.zeros(num_graphs, device=device, dtype=torch.long)
                if uniq.numel() > 0:
                    csum = counts.cumsum(0)
                    starts_for_uniq = torch.cat([
                        torch.zeros(1, device=device, dtype=torch.long),
                        csum[:-1],
                    ], dim=0)
                    edge_starts[uniq] = starts_for_uniq

                edge_counts = torch.zeros(num_graphs, device=device, dtype=torch.long)
                if uniq.numel() > 0:
                    edge_counts[uniq] = counts

                inflow_edge_idx = batch.total_inflow_edge_idx
                if isinstance(inflow_edge_idx, int):
                    inflow_edge_idx = torch.full((num_graphs,), inflow_edge_idx, device=device, dtype=torch.long)
                elif isinstance(inflow_edge_idx, torch.Tensor):
                    inflow_edge_idx = inflow_edge_idx.to(device).view(-1).long()
                    if inflow_edge_idx.numel() == 1 and num_graphs > 1:
                        inflow_edge_idx = inflow_edge_idx.repeat(num_graphs)
                else:
                    inflow_edge_idx = torch.full((num_graphs,), -1, device=device, dtype=torch.long)

                valid = mask.to(device).view(-1)
                if valid.numel() != num_graphs:
                    valid = torch.zeros((num_graphs,), device=device, dtype=torch.bool)

                high_graph_mask = torch.ones((num_graphs,), device=device, dtype=torch.bool)
                if hasattr(batch, 'credibility_level'):
                    try:
                        lvl = batch.credibility_level
                        if isinstance(lvl, torch.Tensor):
                            lvl = lvl.to(device).view(-1).long()
                            if lvl.numel() == num_graphs:
                                high_graph_mask = lvl >= 2
                    except Exception:
                        pass
                elif w_graph is not None:
                    try:
                        if isinstance(w_graph, torch.Tensor) and w_graph.numel() == num_graphs:
                            high_graph_mask = w_graph.to(device).view(-1) >= 0.999
                    except Exception:
                        pass

                valid = valid & high_graph_mask & (inflow_edge_idx >= 0) & (inflow_edge_idx < edge_counts)
                if valid.any():
                    global_edge_idx = edge_starts + inflow_edge_idx
                    pred_in_all = out["edge_hat_m3ps"][global_edge_idx[valid]]
                    obs_in_all = batch.total_inflow_m3ps.to(device).view(-1)[valid]

                    abs_obs_min = 0.02
                    keep_abs = obs_in_all.abs() >= abs_obs_min
                    pred_in = pred_in_all[keep_abs]
                    obs_in = obs_in_all[keep_abs]
                    valid_idx_all = torch.nonzero(valid, as_tuple=False).view(-1)
                    valid_idx = valid_idx_all[keep_abs]

                    w_valid = None
                    if w_graph is not None:
                        try:
                            w_valid = w_graph[valid_idx].clamp(0.0, 1.0)
                        except Exception:
                            w_valid = None
                    if w_valid is None:
                        w_valid = torch.ones((valid_idx.numel(),), device=device, dtype=torch.float32)

                    inflow_n_total = torch.tensor(float(obs_in.numel()), device=device)

                    if obs_in.numel() >= 5:
                        eps = 1e-6
                        pred_raw_full = out.get("edge_hat_m3ps_raw", out["edge_hat_m3ps"])
                        pred_in_raw = pred_raw_full[global_edge_idx[valid]][keep_abs]

                        y = torch.log(torch.clamp(pred_in_raw.abs(), min=eps)) - torch.log(torch.clamp(obs_in.abs(), min=eps))
                        inflow_logratio_median = torch.median(y).detach()
                        med = inflow_logratio_median
                        mad = torch.median(torch.abs(y - med))
                        scale = torch.clamp(1.4826 * mad, min=eps)
                        z = torch.abs(y - med) / scale
                        keep_mask = z <= 6.0

                        min_keep = max(1, int(0.8 * float(obs_in.numel())))
                        if int(keep_mask.sum().item()) >= min_keep and int(keep_mask.sum().item()) < int(obs_in.numel()):
                            pred_in = pred_in[keep_mask]
                            obs_in = obs_in[keep_mask]
                            valid_idx = valid_idx[keep_mask]
                            w_valid = w_valid[keep_mask]

                    inflow_n_kept = torch.tensor(float(obs_in.numel()), device=device)
                    if inflow_n_total.item() > 0:
                        inflow_keep_rate = (inflow_n_kept / inflow_n_total).detach()

                    if pred_in.numel() == 0 or obs_in.numel() == 0 or valid_idx.numel() == 0:
                        pass
                    else:
                        is_self = (batch.edge_attr[:, 5] > 0.5)
                        if hasattr(batch, "edge_std") and (~is_self).any():
                            S_Q = torch.median(batch.edge_std[~is_self]) / 3600.0
                        else:
                            S_Q = torch.tensor(1.0, device=device)
                        S_Q = torch.clamp(S_Q, min=1e-6)

                        hub = F.huber_loss(pred_in / S_Q, obs_in / S_Q, delta=1.0, reduction='none')
                        L_inflow_obs = (hub * w_valid).sum() / (w_valid.sum() + 1e-9)

                        # Compare total predicted terminal flow with effective inflow.
                        if (
                            hasattr(batch, 'node_type')
                            and ('leakage_rate' in out)
                            and ('sum_in' in out)
                        ):
                            try:
                                sink_mask = (batch.node_type == 2)
                                if 'sink_open_mask' in out and isinstance(out['sink_open_mask'], torch.Tensor):
                                    sink_mask = sink_mask & out['sink_open_mask']

                                if sink_mask.any():
                                    sink_graph = batch.batch[sink_mask]
                                    sink_in = out['sum_in'][sink_mask]
                                    sink_out_per_graph = torch.zeros(num_graphs, device=device)
                                    sink_out_per_graph.index_add_(0, sink_graph, sink_in)

                                    leak = out['leakage_rate']
                                    denom = torch.clamp(1.0 - leak, min=1e-3)
                                    closure_pred = sink_out_per_graph[valid_idx] / denom
                                    hub_c = F.huber_loss(closure_pred / S_Q, obs_in / S_Q, delta=1.0, reduction='none')
                                    L_total_closure = (hub_c * w_valid).sum() / (w_valid.sum() + 1e-9)
                            except Exception:
                                L_total_closure = torch.tensor(0.0, device=device)

                        if (
                            ('sum_in' in out)
                            and hasattr(batch, 'node_type')
                            and ('leakage_rate' in out)
                        ):
                            sink_mask = (batch.node_type == 2)
                            if sink_mask.any():
                                sink_in = out['sum_in'][sink_mask]
                                sink_graph = batch.batch[sink_mask]
                                sink_in_per_graph = torch.zeros(num_graphs, device=device)
                                sink_in_per_graph.index_add_(0, sink_graph, sink_in)

                                leak = out['leakage_rate']
                                expected_out = (1.0 - leak) * obs_in
                                hub_lm = F.huber_loss(sink_in_per_graph[valid_idx] / S_Q, expected_out / S_Q, delta=1.0, reduction='none')
                                L_leak_mass = (hub_lm * w_valid).sum() / (w_valid.sum() + 1e-9)

        L_total = (
            L_data +
            lambda_mass * L_mass +
            lambda_energy * L_energy +
            lambda_pressure * L_pressure +
            lambda_inflow * L_inflow_obs +
            1.0 * L_total_closure +
            L_press_data +
            L_leakage +
            1.0 * L_global_mass +
            1.0 * L_leak_mass +
            1.0 * L_leak_prior +
            L_C_constraint +
            L_beta_reg +
            L_demand_reg
        )

        return {
            "L_total": L_total,
            "L_data": L_data,
            "L_rec_main": L_rec_main,
            "L_rec_inp": L_rec_inp,
            "L_edge": L_edge,
            "L_mass": L_mass,
            "L_energy": L_energy,
            "L_pressure": L_pressure,
            "L_press_data": L_press_data,
            "L_leakage": L_leakage,
            "L_global_mass": L_global_mass,
            "L_inflow_obs": L_inflow_obs,
            "L_total_closure": L_total_closure,
            "L_leak_mass": L_leak_mass,
            "L_leak_prior": L_leak_prior,
            "L_C_constraint": L_C_constraint,
            "L_beta_reg": L_beta_reg,
            "L_demand_reg": L_demand_reg,
            "inflow_n_total": inflow_n_total,
            "inflow_n_kept": inflow_n_kept,
            "inflow_keep_rate": inflow_keep_rate,
            "inflow_logratio_median": inflow_logratio_median,
        }

def calculate_barrier_loss(tensor, min_val, max_val, k=10.0, reduction='mean'):
    """Apply a parameter barrier with zero loss within the prescribed range
    and exponentially increasing loss outside the range."""
    high_violation = torch.relu(tensor - max_val)
    low_violation = torch.relu(min_val - tensor)

    max_exp_input = 20.0
    safe_high = torch.min(k * high_violation, torch.tensor(max_exp_input, device=tensor.device))
    safe_low = torch.min(k * low_violation, torch.tensor(max_exp_input, device=tensor.device))

    loss_high = torch.expm1(safe_high) # exp(x) - 1
    loss_low = torch.expm1(safe_low)

    total_loss = loss_high + loss_low

    if reduction == 'mean':
        return torch.mean(total_loss)
    return total_loss


def set_seed(seed: int = 42, full_deterministic: bool = False):
    """Set random seeds for reproducible training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if full_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"✓ Random seed set to {seed}")


def get_lambda_schedule(epoch, max_epochs=200):
    """Return the loss weights for the high-credibility training schedule.

    Use weaker physics weighting early in training and increase the nodal
    mass weight after the initial 30 epochs."""
    if epoch <= 30:
        lambda_mass = 0.2
        lambda_energy = 0.25
        lambda_pressure = 0.003
        lambda_buffer = 0.0
        lambda_inflow = 0.5
    else:
        lambda_mass = 0.35
        lambda_energy = 0.25
        lambda_pressure = 0.003
        lambda_buffer = 0.0
        lambda_inflow = 0.5

    return lambda_mass, lambda_energy, lambda_pressure, lambda_buffer, lambda_inflow


def set_physics_params_grad(model, requires_grad=True):
    """Set gradient tracking for the energy-side parameters."""
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    lock_names = [
        'resistance_beta_logit',
        'pressure_scale',
        'pressure_base'
    ]

    for name, param in orig_model.named_parameters():
        if any(ln in name for ln in lock_names):
            param.requires_grad = requires_grad


def set_stage_trainability(model):
    """Freeze the meter offsets during Stage I and train the remaining parameters."""
    orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, p in orig.named_parameters():
        if ('bias_mu_raw' in name) or ('bias_delta_raw' in name):
            p.requires_grad = False
        else:
            p.requires_grad = True

def diagnose_physics(model, batch, out, epoch):
    """Report flow, head and physical residual diagnostics."""
    with torch.no_grad():
        orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        q_m3ps = out['edge_hat_m3ps']
        q_m3ph = out.get('edge_hat_m3ph', q_m3ps * 3600)

        is_self = batch.edge_attr[:, 5] > 0.5
        q_real = q_m3ph[~is_self]
        q_real_m3ps = q_m3ps[~is_self]

        print(f"\n{'='*80}")
        print(f"[Epoch {epoch}] Hydraulic diagnostics")
        print(f"{'='*80}")
        print(f"Flow Q (m³/h): min={q_real.min():.2f}, max={q_real.max():.2f}, mean={q_real.mean():.2f}")

        if 'p_head' in out:
            p = out['p_head']
            print(f"Pressure head p (m): min={p.min():.2f}, max={p.max():.2f}, mean={p.mean():.2f}")

        if 'H_node' in out:
            H = out['H_node']
            print(f"Total head H (m): min={H.min():.2f}, max={H.max():.2f}, mean={H.mean():.2f}")

        print(f"\n=== A. Flow direction ===")
        ratio_rev = (q_real_m3ps < 0).float().mean().item()
        print(f"Proportion of reverse flows: {ratio_rev:.4f}")

        if (
            hasattr(batch, 'total_inflow_mask')
            and hasattr(batch, 'total_inflow_m3ps')
            and hasattr(batch, 'total_inflow_edge_idx')
            and ('edge_hat_m3ps' in out)
            and hasattr(batch, 'edge_index')
            and hasattr(batch, 'batch')
        ):
            mask = batch.total_inflow_mask
            if isinstance(mask, torch.Tensor) and mask.any():
                if hasattr(batch, "num_graphs") and isinstance(batch.num_graphs, int) and batch.num_graphs > 0:
                    num_graphs = int(batch.num_graphs)
                elif isinstance(batch.batch, torch.Tensor) and batch.batch.numel() > 0:
                    num_graphs = int(batch.batch.max().item()) + 1
                else:
                    num_graphs = 1

                edge_graph_all = batch.batch[batch.edge_index[0]]  # [E]
                full_edge_idx = torch.arange(batch.edge_index.size(1), device=q_m3ps.device)
                real_full_edge_idx = full_edge_idx[~is_self]
                edge_graph_real = edge_graph_all[~is_self]
                uniq, counts = torch.unique_consecutive(edge_graph_real, return_counts=True)

                real_edge_starts = torch.zeros(num_graphs, device=q_m3ps.device, dtype=torch.long)
                real_edge_counts = torch.zeros(num_graphs, device=q_m3ps.device, dtype=torch.long)
                if uniq.numel() > 0:
                    csum = counts.cumsum(0)
                    starts_for_uniq = torch.cat([
                        torch.zeros(1, device=q_m3ps.device, dtype=torch.long),
                        csum[:-1],
                    ], dim=0)
                    real_edge_starts[uniq] = starts_for_uniq
                    real_edge_counts[uniq] = counts

                inflow_edge_idx = batch.total_inflow_edge_idx
                if isinstance(inflow_edge_idx, int):
                    inflow_edge_idx = torch.full((num_graphs,), inflow_edge_idx, device=q_m3ps.device, dtype=torch.long)
                elif isinstance(inflow_edge_idx, torch.Tensor):
                    inflow_edge_idx = inflow_edge_idx.to(q_m3ps.device).view(-1).long()
                    if inflow_edge_idx.numel() == 1 and num_graphs > 1:
                        inflow_edge_idx = inflow_edge_idx.repeat(num_graphs)
                else:
                    inflow_edge_idx = torch.full((num_graphs,), -1, device=q_m3ps.device, dtype=torch.long)

                valid = mask.to(q_m3ps.device).view(-1)
                if valid.numel() != num_graphs:
                    valid = torch.zeros((num_graphs,), device=q_m3ps.device, dtype=torch.bool)

                valid = valid & (inflow_edge_idx >= 0) & (inflow_edge_idx < real_edge_counts)
                if valid.any():
                    real_pos = real_edge_starts + inflow_edge_idx
                    full_idx_per_graph = real_full_edge_idx[real_pos]

                    obs_all = batch.total_inflow_m3ps.to(q_m3ps.device).view(-1)
                    pred_all = out['edge_hat_m3ps'][full_idx_per_graph]
                    pred_raw_all = out.get('edge_hat_m3ps_raw', out['edge_hat_m3ps'])[full_idx_per_graph]

                    abs_obs_min = 0.02
                    keep_abs = obs_all.abs() >= abs_obs_min
                    keep = valid & keep_abs

                    if keep.any() and int(keep.sum().item()) >= 5:
                        eps = 1e-6
                        y = torch.log(torch.clamp(pred_raw_all[keep].abs(), min=eps)) - torch.log(torch.clamp(obs_all[keep].abs(), min=eps))
                        med = y.median()
                        mad = (y - med).abs().median()
                        scale = torch.clamp(1.4826 * mad, min=eps)
                        z = (y - med).abs() / scale
                        keep_local = z <= 6.0
                        min_keep = max(1, int(0.8 * float(y.numel())))
                        if int(keep_local.sum().item()) >= min_keep and int(keep_local.sum().item()) < int(y.numel()):
                            keep2 = keep.clone()
                            keep2[keep] = keep_local
                            keep = keep2

                    pred = pred_all[keep]
                    obs = obs_all[keep]
                    pred_raw = pred_raw_all[keep]

                    pred_sum = pred.sum()
                    obs_sum = obs.sum()
                    pred_mean = pred.mean()
                    obs_mean = obs.mean()

                    diff_sum = pred_sum - obs_sum
                    rel_sum = diff_sum / (obs_sum.abs() + 1e-6)
                    diff_mean = pred_mean - obs_mean
                    rel_mean = diff_mean / (obs_mean.abs() + 1e-6)

                    print(
                        f"[Total inflow@P-101] sum:  pred={pred_sum.item():.6f}, obs={obs_sum.item():.6f}, diff={diff_sum.item():+.6f} m³/s, rel={rel_sum.item():+.4%}"
                    )
                    print(
                        f"[Total inflow@P-101] mean: pred={pred_mean.item():.6f}, obs={obs_mean.item():.6f}, diff={diff_mean.item():+.6f} m³/s, rel={rel_mean.item():+.4%} (kept={int(keep.sum().item())}/{int(valid.sum().item())}, graphs={num_graphs})"
                    )
                    if pred_raw.numel() > 0:
                        raw_mean = pred_raw.mean()
                        print(f"[Total inflow@P-101] raw_mean={raw_mean.item():.6f} m³/s, after_mean={pred_mean.item():.6f} m³/s")

        print(f"\n=== B. Mass conservation ===")
        r_mass = out['r_mass']
        valid_mask = out.get('valid_mass_mask', torch.ones_like(r_mass, dtype=torch.bool))

        if valid_mask.any():
            r_mass_valid = r_mass[valid_mask]
            print(f"Mass residual |r_mass|: mean={r_mass_valid.abs().mean():.6f}, max={r_mass_valid.abs().max():.6f}")

        sum_in = out.get('sum_in', torch.zeros_like(r_mass))
        sum_out = out.get('sum_out', torch.zeros_like(r_mass))
        demand = out.get('demand', torch.zeros_like(r_mass))
        total_demand_vec = out.get('total_demand', demand)
        sink_open_mask = out.get('sink_open_mask', None)

        node_type = getattr(batch, 'node_type', None)
        if node_type is not None:
            source_mask = (node_type == 1)
            sink_mask = (node_type == 2)

            if source_mask.any():
                source_out = sum_out[source_mask].sum().item()
                print(f"Total source-node outflow (m³/s): {source_out:.4f}")
            if sink_mask.any():
                sink_in = sum_in[sink_mask].sum().item()
                sink_out = sum_out[sink_mask].sum().item()
                sink_net = (sum_in[sink_mask] - sum_out[sink_mask]).sum().item()
                print(f"Terminal node sum_in  (m³/s): {sink_in:.4f}")
                print(f"Terminal node sum_out (m³/s): {sink_out:.4f}")
                print(f"Terminal node net_flow(m³/s): {sink_net:.4f}  (net=sum_in-sum_out)")

            total_demand = total_demand_vec.sum().item()
            print(f"Total demand (m³/s): {total_demand:.4f}")

            if sink_open_mask is not None and sink_mask.any():
                sink_open = sink_open_mask[sink_mask]
                open_rate = sink_open.float().mean().item() if sink_open.numel() > 0 else 0.0
                print(f"Proportion of terminal openings above 5%: {open_rate:.4f}")
                sink_demand = total_demand_vec[sink_mask]
                if sink_demand.numel() > 0:
                    d_open = sink_demand[sink_open].sum().item() if sink_open.any() else 0.0
                    d_closed = sink_demand[~sink_open].sum().item() if (~sink_open).any() else 0.0
                    print(f"Terminal demand by valve state (m³/s): open={d_open:.4f}, closed={d_closed:.4f}")

        print(f"Nodes subject to the constraint: {valid_mask.sum().item()}/{len(valid_mask)}")

        print(f"\n=== C. Energy conservation ===")
        r_energy = out['r_energy']
        r_energy_real = r_energy[~is_self]

        print(f"Energy residual |r_energy| (m): mean={r_energy_real.abs().mean():.4f}, max={r_energy_real.abs().max():.4f}")

        if 'S_Q' in out and 'S_H' in out:
            S_Q = out['S_Q']
            S_H = out['S_H']
            print(f"Scales: S_Q={S_Q.item():.6f} m³/s, S_H={S_H.item():.4f} m")

        if 'r_mass_nd' in out:
            print(f"Dimensionless mass residual: mean={out['r_mass_nd'].abs().mean():.4f}")
        if 'r_energy_nd' in out and out['r_energy_nd'].numel() > 0:
            print(f"Dimensionless energy residual: mean={out['r_energy_nd'].abs().mean():.4f}")

        print(f"\n=== D. Hydraulic parameters ===")

        params = orig_model.get_physics_params()
        C_main = params['C_main'].item()
        C_branch = params['C_branch'].item()
        alpha = params['alpha'].item()
        cm_ok = "✓" if 120 <= C_main <= 140 else "✗"
        cb_ok = "✓" if 140 <= C_branch <= 160 else "✗"
        al_ok = "✓" if 0.10 <= alpha <= 0.20 else "✗"

        print(f"  {cm_ok} C_main = {C_main:.2f} (range: 120-140)")
        print(f"  {cb_ok} C_branch = {C_branch:.2f} (range: 140-160)")
        print(f"  {al_ok} alpha = {alpha:.4f} (range: 0.10-0.20)")
        if 'leakage_rate' in out:
            leak = float(out['leakage_rate'].detach().cpu().item())
            print(f"  ✓ leakage = {leak:.4f} (range: 0.05-0.15)")


        if hasattr(orig_model, 'get_raw_params'):
            raw = orig_model.get_raw_params()
            print(f"\n  [Untransformed logits]")
            print(f"  resistance_beta_logit = {raw['resistance_beta_logit']:.4f}")
            print(f"  leakage_logit = {raw['leakage_logit']:.4f}")


def create_optimizer_with_param_groups(model, lr=1e-3, physics_lr_scale=3.0, weight_decay=1e-4):
    """Create the optimiser with separate parameter groups and learning rates."""
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    energy_names = ['resistance_beta_logit', 'pressure_scale', 'pressure_base']
    leak_names = ['leakage_logit']
    demand_names = ['latent_demand_logit']

    energy_params = []
    leak_params = []
    demand_params = []
    other_params = []

    for name, param in orig_model.named_parameters():
        if any(pn in name for pn in energy_names):
            energy_params.append(param)
            print(f"  Hydraulic parameter(energy): {name}")
        elif any(pn in name for pn in leak_names):
            leak_params.append(param)
            print(f"  Hydraulic parameter(leak): {name}")
        elif any(pn in name for pn in demand_names):
            demand_params.append(param)
            print(f"  Hydraulic parameter(demand): {name}")
        else:
            other_params.append(param)

    leak_lr_scale = 1.0
    demand_lr_scale = 1.0

    optimizer = optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': energy_params, 'lr': lr * physics_lr_scale, 'weight_decay': 0.0},
        {'params': leak_params, 'lr': lr * leak_lr_scale, 'weight_decay': 0.0},
        {'params': demand_params, 'lr': lr * demand_lr_scale, 'weight_decay': 0.0},
    ])

    print(
        f"✓ Optimiser created: net={len(other_params)}, energy={len(energy_params)}, leak={len(leak_params)}, demand={len(demand_params)}"
    )
    print(
        f"  LR: net={lr}, energy={lr * physics_lr_scale}, leak={lr * leak_lr_scale}, demand={lr * demand_lr_scale}"
    )

    return optimizer


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device="cuda",
    epoch=0,
    clip_grad=1.0,
    accumulation_steps=1,
    lambda_mass=0.0,
    lambda_energy=0.0,
    lambda_pressure=0.0,
    lambda_buffer=0.0,
    lambda_inflow=0.0
):
    """Train for one epoch and aggregate the loss diagnostics."""
    model.train()

    cum = {k: 0.0 for k in
        ["L_total", "L_data", "L_rec_main", "L_rec_inp", "L_edge",
         "L_mass", "L_energy", "L_pressure", "L_press_data",
         "L_inflow_obs", "L_global_mass", "L_leakage",
         "L_total_closure",
         "L_leak_mass", "L_leak_prior",
         "L_C_constraint", "L_beta_reg", "L_demand_reg"]}
    inflow_total_sum = 0.0
    inflow_kept_sum = 0.0
    inflow_logratio_sum = 0.0
    inflow_logratio_w = 0.0
    n = 0
    accum_step = 0
    nan_count = 0
    total_batches = len(loader)

    print_interval = max(1, total_batches // 10)

    for idx, batch in enumerate(loader):
        batch.lambda_mass = lambda_mass
        batch.lambda_energy = lambda_energy
        batch = batch.to(device)

        out = model(batch)
        loss_dict = loss_fn(
            out,
            batch,
            model=model,
            lambda_mass=lambda_mass,
            lambda_energy=lambda_energy,
            lambda_pressure=lambda_pressure,
            lambda_buffer=lambda_buffer,
            lambda_inflow=lambda_inflow,
            epoch=epoch,
        )

        def _as_float(v):
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        nt = _as_float(loss_dict.get('inflow_n_total', 0.0))
        nk = _as_float(loss_dict.get('inflow_n_kept', 0.0))
        med = _as_float(loss_dict.get('inflow_logratio_median', 0.0))
        inflow_total_sum += nt
        inflow_kept_sum += nk
        if nt > 0:
            inflow_logratio_sum += med * nt
            inflow_logratio_w += nt

        task_data = loss_dict['L_data']

        task_mass = lambda_mass * loss_dict['L_mass']
        task_energy = lambda_energy * loss_dict['L_energy']

        task_pressure = (
            lambda_pressure * loss_dict['L_pressure'] +
            loss_dict.get('L_press_data', 0.0)
        )

        task_constraints = (
            loss_dict.get('L_leakage', 0) +
            loss_dict.get('L_leak_mass', 0) +
            loss_dict.get('L_leak_prior', 0) +
            loss_dict.get('L_total_closure', 0) +
            loss_dict.get('L_C_constraint', 0) +
            loss_dict.get('L_demand_reg', 0) +
            loss_dict.get('L_beta_reg', 0) +
            loss_dict.get('L_global_mass', 0) +
            lambda_inflow * loss_dict.get('L_inflow_obs', 0)
        )

        objectives = []
        for loss in [task_data, task_mass, task_energy, task_pressure, task_constraints]:
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                objectives.append(loss)

        L_total = torch.stack(objectives).sum() if objectives else loss_dict['L_total']

        if torch.isnan(L_total) or torch.isinf(L_total):
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.pc_backward(objectives)
        accum_step += 1

        if accum_step >= accumulation_steps or idx == total_batches - 1:
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0

        for k in cum:
            cum[k] += float(loss_dict[k].item())
        n += 1

        if (idx + 1) % print_interval == 0:
            progress = (idx + 1) / total_batches * 100
            print(f"\r  [{progress:5.1f}%] L_tot={cum['L_total']/n:.4f}, "
                  f"L_data={cum['L_data']/n:.4f}, L_mass={cum['L_mass']/n:.4f}, "
                  f"L_energy={cum['L_energy']/n:.4f}", end="")

    print()

    if nan_count > 0:
        print(f"  Warning: {nan_count} batches skipped after NaN detection")

    metrics = {k: v / max(n, 1) for k, v in cum.items()}
    metrics['inflow_n_total'] = inflow_total_sum
    metrics['inflow_n_kept'] = inflow_kept_sum
    metrics['inflow_keep_rate'] = (inflow_kept_sum / inflow_total_sum) if inflow_total_sum > 0 else 0.0
    metrics['inflow_logratio_median'] = (inflow_logratio_sum / inflow_logratio_w) if inflow_logratio_w > 0 else 0.0
    return metrics


@torch.no_grad()
def validate(model, loader, loss_fn, device, lambda_mass, lambda_energy, lambda_pressure, lambda_buffer, lambda_inflow, epoch=0):
    """Evaluate the validation loss and diagnostics."""
    model.eval()

    cum = {k: 0.0 for k in
           ["L_total", "L_data", "L_rec_main", "L_rec_inp", "L_edge",
            "L_mass", "L_energy", "L_pressure", "L_press_data",
            "L_inflow_obs", "L_global_mass", "L_leakage",
            "L_total_closure",
            "L_leak_mass", "L_leak_prior",
            "L_C_constraint", "L_beta_reg", "L_demand_reg"]}
    inflow_total_sum = 0.0
    inflow_kept_sum = 0.0
    inflow_logratio_sum = 0.0
    inflow_logratio_w = 0.0
    n = 0

    for batch in loader:
        batch.lambda_mass = lambda_mass
        batch.lambda_energy = lambda_energy
        batch = batch.to(device)

        out = model(batch)
        losses = loss_fn(
            out,
            batch,
            model=model,
            lambda_mass=lambda_mass,
            lambda_energy=lambda_energy,
            lambda_pressure=lambda_pressure,
            lambda_buffer=lambda_buffer,
            lambda_inflow=lambda_inflow,
            epoch=epoch,
        )

        def _as_float(v):
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        nt = _as_float(losses.get('inflow_n_total', 0.0))
        nk = _as_float(losses.get('inflow_n_kept', 0.0))
        med = _as_float(losses.get('inflow_logratio_median', 0.0))
        inflow_total_sum += nt
        inflow_kept_sum += nk
        if nt > 0:
            inflow_logratio_sum += med * nt
            inflow_logratio_w += nt

        for k in cum:
            cum[k] += float(losses[k].item())
        n += 1

    metrics = {k: v / max(n, 1) for k, v in cum.items()}
    metrics['inflow_n_total'] = inflow_total_sum
    metrics['inflow_n_kept'] = inflow_kept_sum
    metrics['inflow_keep_rate'] = (inflow_kept_sum / inflow_total_sum) if inflow_total_sum > 0 else 0.0
    metrics['inflow_logratio_median'] = (inflow_logratio_sum / inflow_logratio_w) if inflow_logratio_w > 0 else 0.0
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_val, bad, ckpt_path):
    """Save the model, optimiser and scheduler states."""
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    params = orig_model.get_physics_params()

    checkpoint = {
        "model_state_dict": orig_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val": best_val,
        "bad": bad,
        "physics_params": {
            "C_main": params['C_main'].item(),
            "C_branch": params['C_branch'].item(),
            "alpha": params['alpha'].item(),
            "leakage": orig_model.get_leakage_rate().item() if hasattr(orig_model, 'get_leakage_rate') else 0.0,
        }
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\n✓ Checkpoint saved at epoch {epoch}")


def main():
    """Run Stage-I PI-GAT training."""
    parser = argparse.ArgumentParser(description="Train PI-GAT (Physics-Informed GAT) for irrigation network data cleaning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda/cpu")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--physics-lr-scale", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile (Linux + CUDA only; on Windows it will be ignored)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity run: overrides epochs/batch/workers for a fast check",
    )
    parser.add_argument(
        "--effective_opening_mapping_json",
        type=str,
        default="opening_mapping/Debiased_eom_full.json",
        help=(
            "Optional JSON produced by opening_mapping/build_eom.py or Debiased_build_eom.py. "
            "Applies per-valve effective opening mapping ONLY when computing energy residual. "
            "Available options (in opening_mapping/): "
            "Debiased_eom_full.json (default, recommended: full data + Scale debias), "
            "eom_high.json (high-credibility subset), "
            "eom_full.json (full data, no debias). "
            "Pass empty string '' to disable."
        ),
    )
    parser.add_argument(
        "--fixed-leakage",
        type=float,
        default=None,
        help="If set, freezes the leakage rate to this exact value (e.g., 0.05, 0.1) instead of learning it.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory for training logs and checkpoints. Defaults to current script directory.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.batch_size = min(args.batch_size, 8)
        args.train_num_workers = 0
        args.val_num_workers = 0

    SEED = args.seed
    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    PHYSICS_LR_SCALE = 3.0
    PHYSICS_LR_SCALE = args.physics_lr_scale
    PATIENCE = args.patience
    EARLY_STOP_MIN_EPOCH = max(1, int(PATIENCE))
    WINDOW = args.window

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PREPROCESS_DIR = BASE_DIR.parent / "Preprocess_Timeseries" / "outputs" / "preprocessed"

    TOPO_CSV = BASE_DIR.parent / "network_topology.csv"
    ZSCORE_CSV = PREPROCESS_DIR / "flow_5min_zscore.csv"
    RESTMASK_CSV = PREPROCESS_DIR / "mask_rest.csv"
    ZSCORE_STATS = PREPROCESS_DIR / "zscore_stats.json"
    OPEN_CSV = PREPROCESS_DIR / "flow_5min_imputed.csv"
    CREDIBILITY_CSV = BASE_DIR / "preprocessing" / "credibility_windows.csv"

    OUT_DIR = Path(args.out_dir) if args.out_dir else BASE_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CKPT_PATH = OUT_DIR / "pi_gat_ckpt.pth"
    BEST_PATH = OUT_DIR / "pi_gat_best.pth"
    LOG_PATH = OUT_DIR / "training_log.csv"
    DEBUG_LOG_PATH = OUT_DIR / "training_log_debug.csv"

    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    print("\n=== Building the graph ===")
    data_graph, node2idx, pipe2idx, sensor2edge_idx = build_graph_from_csv(
        TOPO_CSV, weight_mode="conductance", add_self_loops=True
    )
    sensor2node_idx = data_graph.sensor2node_idx

    print("\n=== Preparing the dataset ===")
    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=ZSCORE_CSV,
        restmask_path=RESTMASK_CSV,
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WINDOW,
        random_mask_range=(0.1, 0.2),
        open_paths=OPEN_CSV,
        zscore_stats_path=ZSCORE_STATS,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None),
    )

    def _cred_level_for_dataset_idx(ds: Any, di: int) -> int:
        if getattr(ds, 'has_credibility', False):
            try:
                t = int(ds.valid_t[int(di)])
                return int(ds.cred_level_by_t.get(t, 0))
            except Exception:
                return 0
        return 2

    idx_all = list(range(len(dataset)))
    idx_high = [i for i in idx_all if _cred_level_for_dataset_idx(dataset, i) >= 2]
    idx_mid = [i for i in idx_all if _cred_level_for_dataset_idx(dataset, i) == 1]
    idx_low = [i for i in idx_all if _cred_level_for_dataset_idx(dataset, i) <= 0]

    def _shuffle_inplace(xs: list[int], seed: int) -> None:
        rng = random.Random(seed)
        rng.shuffle(xs)

    def _split_70_15_15(xs: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(xs)
        if n <= 0:
            return [], [], []
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
        train = xs[:n_train]
        val = xs[n_train:n_train + n_val]
        test = xs[n_train + n_val:n_train + n_val + n_test]
        return train, val, test

    idx_high_s = idx_high.copy()
    _shuffle_inplace(idx_high_s, SEED + 11)
    high_train, high_val, high_test = _split_70_15_15(idx_high_s)

    train_idx_high = high_train
    val_idx_high = high_val
    test_idx_high = high_test

    train_ds_high = torch.utils.data.Subset(dataset, train_idx_high)
    val_ds_high = torch.utils.data.Subset(dataset, val_idx_high)
    test_ds = torch.utils.data.Subset(dataset, test_idx_high)

    _train_g = torch.Generator()
    _train_g.manual_seed(SEED)
    _worker_init_fn = functools.partial(_worker_init_fn_impl, seed=SEED)

    train_loader_high = DataLoader(
        cast(Any, train_ds_high),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=(DEVICE == "cuda"),
        generator=_train_g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader_high = DataLoader(
        cast(Any, val_ds_high),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.val_num_workers,
        worker_init_fn=_worker_init_fn,
    )

    print(
        f"Dataset by credibility: high={len(idx_high)}, mid={len(idx_mid)}, low={len(idx_low)}"
    )
    print(
        f"Split (high credibility only): train={len(train_idx_high)}, val={len(val_idx_high)}, test={len(test_idx_high)}"
    )

    print("\n=== Building the model ===")
    assert data_graph.num_nodes is not None
    N_nodes = int(data_graph.num_nodes)

    model: Any = PI_GAT(
        win=WINDOW,
        num_nodes=N_nodes,
        hidden=64,
        heads=8,
        dropout=0.1,
        in_feat=2
    ).to(DEVICE)

    if args.fixed_leakage is not None:
        model._leakage_range = (args.fixed_leakage, args.fixed_leakage)
        model.leakage_logit.requires_grad = False
        print(f"✓ Fixed leakage rate to {args.fixed_leakage} (Frozen)")

    params = model.get_physics_params()
    print(f"Initial hydraulic parameters:")
    print(f"  C_main = {params['C_main'].item():.2f}")
    print(f"  C_branch = {params['C_branch'].item():.2f}")
    print(f"  alpha = {params['alpha'].item():.4f} (ratio of additional local losses to friction loss, 0.10-0.20)")
    print(f"  leakage = {model.get_leakage_rate().item():.4f}")

    if str(getattr(args, "effective_opening_mapping_json", "") or "").strip():
        try:
            from opening_mapping.eom import build_edge_y_knots01, load_mapping_json

            mp = load_mapping_json(Path(str(args.effective_opening_mapping_json)))
            xk01, yk01 = build_edge_y_knots01(
                sensor2edge_idx=getattr(data_graph, "sensor2edge_idx", {}),
                num_edges=int(data_graph.edge_index.size(1)),  # type: ignore[union-attr]
                mapping_json=mp,
            )
            model.set_effective_opening_mapping(yk01.to(device=DEVICE))
            print(f"✓ Effective-opening mapping enabled for the energy term: {args.effective_opening_mapping_json}")
        except Exception as e:
            print(f"⚠ Could not load effective-opening mapping. Continuing without it: {e}")

    USE_COMPILE = bool(args.use_compile)

    if USE_COMPILE and platform.system() == "Linux" and hasattr(torch, 'compile') and DEVICE == "cuda":
        try:
            model = cast(Any, torch.compile(cast(nn.Module, model), mode="reduce-overhead"))
            print("✓ Model compiled (torch.compile)")
        except Exception as e:
            print(f"⚠ Compilation failed. Running in eager mode: {e}")
    else:
        if platform.system() == "Windows":
            print("✓ Running in eager mode on Windows (compilation requires Linux and Triton in this environment)")
        else:
            print("✓ Running in eager mode")

    loss_fn = LossPI()

    optimizer_base = create_optimizer_with_param_groups(
        model, lr=LR, physics_lr_scale=PHYSICS_LR_SCALE
    )

    optimizer = PCGrad(optimizer_base)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_base, T_0=20, T_mult=2, eta_min=1e-6
    )

    set_stage_trainability(model)

    log_header = (
        "Epoch,"
        "Train_Ltot,Train_Ldata,Train_Lphys,Train_Linflow,Train_Lclosure,Train_Lleakage,Train_Lleakmass,Train_Lleakprior,"
        "Val_Ltot,Val_Ldata,Val_Lphys,Val_Linflow,Val_Lclosure,Val_Lleakage,Val_Lleakmass,Val_Lleakprior,Val_Lmass,Val_Lenergy,"
        "C_main,C_branch,Alpha,Leakage,LR,Time\n"
    )

    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if "Epoch,Stage," in first_line or "Train_Lclosure" not in first_line:
                with open(LOG_PATH, 'w', encoding='utf-8') as f:
                    f.write(log_header)
        except Exception:
            with open(LOG_PATH, 'w', encoding='utf-8') as f:
                f.write(log_header)
    else:
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(log_header)

    debug_log_header = (
        "Epoch,"
        "Lambda_mass,Lambda_energy,Lambda_pressure,Lambda_buffer,Lambda_inflow,"
        "Train_inflow_n_total,Train_inflow_n_kept,Train_inflow_keep_rate,Train_inflow_logratio_median,"
        "Train_Ltot,Train_Ltot_recomp,Train_Ldata,Train_Lmass,Train_Lenergy,Train_Lpressure,Train_Lpressdata,Train_Linflow,"
        "Train_Lclosure,Train_Lleakage,Train_Lglobalmass,Train_Lleakmass,Train_Lleakprior,Train_LC,Train_Lbeta,Train_Ldemandreg,Train_Lother,"
        "Val_inflow_n_total,Val_inflow_n_kept,Val_inflow_keep_rate,Val_inflow_logratio_median,"
        "Val_Ltot,Val_Ltot_recomp,Val_Ldata,Val_Lmass,Val_Lenergy,Val_Lpressure,Val_Lpressdata,Val_Linflow,"
        "Val_Lclosure,Val_Lleakage,Val_Lglobalmass,Val_Lleakmass,Val_Lleakprior,Val_LC,Val_Lbeta,Val_Ldemandreg,Val_Lother\n"
    )

    if not DEBUG_LOG_PATH.exists():
        with open(DEBUG_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(debug_log_header)
    else:
        try:
            with open(DEBUG_LOG_PATH, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if (
                "Epoch,Stage," in first_line
                or "Train_inflow_keep_rate" not in first_line
                or "Val_inflow_keep_rate" not in first_line
                or "Train_Lbiassmooth" in first_line
                or "Val_Lbiassmooth" in first_line
            ):
                with open(DEBUG_LOG_PATH, 'w', encoding='utf-8') as f:
                    f.write(debug_log_header)
        except Exception:
            with open(DEBUG_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write(debug_log_header)

    print("\n=== Starting training ===")
    epoch = 0
    val_metrics: dict[str, Any] = {"L_total": float("inf")}
    orig_model: Any = cast(Any, model)._orig_mod if hasattr(model, "_orig_mod") else model
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_model_saved = False

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        lambda_mass, lambda_energy, lambda_pressure, lambda_buffer, lambda_inflow = get_lambda_schedule(epoch, MAX_EPOCHS)

        print(
            f"\n[Epoch {epoch}/{MAX_EPOCHS}] λmass={lambda_mass:.3f}, λenergy={lambda_energy:.3f}, "
            f"λpressure={lambda_pressure:.4f}, λinflow={lambda_inflow:.3f}"
        )

        active_train_loader = train_loader_high

        train_metrics = train_one_epoch(
            model, active_train_loader, loss_fn, optimizer, DEVICE, epoch,
            clip_grad=1.0, accumulation_steps=1,
            lambda_mass=lambda_mass, lambda_energy=lambda_energy,
            lambda_pressure=lambda_pressure, lambda_buffer=lambda_buffer,
            lambda_inflow=lambda_inflow
        )

        active_val_loader = val_loader_high
        val_metrics = validate(
            model, active_val_loader, loss_fn, DEVICE,
            lambda_mass, lambda_energy, lambda_pressure, lambda_buffer, lambda_inflow, epoch=epoch
        )

        scheduler.step()

        orig_model: Any = cast(Any, model)._orig_mod if hasattr(model, "_orig_mod") else model
        params = orig_model.get_physics_params()

        train_Lphys = train_metrics['L_mass'] + train_metrics['L_energy']
        val_Lphys = val_metrics['L_mass'] + val_metrics['L_energy']

        train_Linflow = float(train_metrics.get('L_inflow_obs', 0.0))
        val_Linflow = float(val_metrics.get('L_inflow_obs', 0.0))

        train_Lclosure = float(train_metrics.get('L_total_closure', 0.0))
        val_Lclosure = float(val_metrics.get('L_total_closure', 0.0))

        train_Lleak = float(train_metrics.get('L_leakage', 0.0))
        train_Lleakmass = float(train_metrics.get('L_leak_mass', 0.0))
        train_Lleakprior = float(train_metrics.get('L_leak_prior', 0.0))
        val_Lleak = float(val_metrics.get('L_leakage', 0.0))
        val_Lleakmass = float(val_metrics.get('L_leak_mass', 0.0))
        val_Lleakprior = float(val_metrics.get('L_leak_prior', 0.0))

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train: L_tot={train_metrics['L_total']:.4f}, L_data={train_metrics['L_data']:.4f}, L_phys={train_Lphys:.4f}, L_inflow={train_Linflow:.4f}")
        print(f"  Val:   L_tot={val_metrics['L_total']:.4f}, L_data={val_metrics['L_data']:.4f}, L_phys={val_Lphys:.4f}, L_inflow={val_Linflow:.4f}")
        print(f"  Hydraulic parameters: C_main={params['C_main'].item():.2f}, C_branch={params['C_branch'].item():.2f}, "
              f"α={params['alpha'].item():.4f}, leak={orig_model.get_leakage_rate().item():.4f}")

        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch},"
                f"{train_metrics['L_total']:.4f},{train_metrics['L_data']:.4f},{train_Lphys:.4f},{train_Linflow:.4f},{train_Lclosure:.4f},{train_Lleak:.4f},{train_Lleakmass:.4f},{train_Lleakprior:.4f},"
                f"{val_metrics['L_total']:.4f},{val_metrics['L_data']:.4f},{val_Lphys:.4f},{val_Linflow:.4f},{val_Lclosure:.4f},{val_Lleak:.4f},{val_Lleakmass:.4f},{val_Lleakprior:.4f},{val_metrics['L_mass']:.4f},{val_metrics['L_energy']:.4f},"
                f"{params['C_main'].item():.2f},{params['C_branch'].item():.2f},"
                f"{params['alpha'].item():.4f},{orig_model.get_leakage_rate().item():.4f},"
                f"{current_lr:.6f},{epoch_time:.1f}s\n"
            )

        def _f(metrics, key):
            v = metrics.get(key, 0.0)
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        def _recompute_total(metrics):
            L_data = _f(metrics, 'L_data')
            L_mass = _f(metrics, 'L_mass')
            L_energy = _f(metrics, 'L_energy')
            L_pressure = _f(metrics, 'L_pressure')
            L_press_data = _f(metrics, 'L_press_data')
            L_inflow = _f(metrics, 'L_inflow_obs')
            L_closure = _f(metrics, 'L_total_closure')
            L_leak = _f(metrics, 'L_leakage')
            L_gmass = _f(metrics, 'L_global_mass')
            L_leak_mass = _f(metrics, 'L_leak_mass')
            L_leak_prior = _f(metrics, 'L_leak_prior')
            L_C = _f(metrics, 'L_C_constraint')
            L_beta = _f(metrics, 'L_beta_reg')
            L_demand = _f(metrics, 'L_demand_reg')

            L_total = (
                L_data
                + lambda_mass * L_mass
                + lambda_energy * L_energy
                + lambda_pressure * L_pressure
                + lambda_inflow * L_inflow
                + 1.0 * L_closure
                + L_press_data
                + L_leak
                + 1.0 * L_gmass
                + 1.0 * L_leak_mass
                + 1.0 * L_leak_prior
                + L_C
                + L_beta
                + L_demand
            )

            L_other = L_total - (L_data + lambda_mass * L_mass + lambda_energy * L_energy + lambda_inflow * L_inflow)
            return (
                L_total, L_other,
                L_data, L_mass, L_energy, L_pressure, L_press_data, L_inflow,
                L_closure, L_leak, L_gmass, L_leak_mass, L_leak_prior,
                L_C, L_beta, L_demand
            )

        (
            train_tot_recomp, train_other,
            train_Ldata_, train_Lmass_, train_Lenergy_, train_Lpressure_, train_Lpressdata_, train_Linflow_,
            train_Lclosure_, train_Lleak_, train_Lgmass_, train_Lleakmass_, train_Lleakprior_,
            train_LC_, train_Lbeta_, train_Ldemand_,
        ) = _recompute_total(train_metrics)
        (
            val_tot_recomp, val_other,
            val_Ldata_, val_Lmass_, val_Lenergy_, val_Lpressure_, val_Lpressdata_, val_Linflow_,
            val_Lclosure_, val_Lleak_, val_Lgmass_, val_Lleakmass_, val_Lleakprior_,
            val_LC_, val_Lbeta_, val_Ldemand_,
        ) = _recompute_total(val_metrics)

        train_tot_logged = float(train_metrics['L_total'])
        val_tot_logged = float(val_metrics['L_total'])
        if abs(train_tot_logged - train_tot_recomp) > 1e-3:
            print(f"  [Warning] Train L_total differs from the recomputed value: logged={train_tot_logged:.6f}, recompute={train_tot_recomp:.6f}")
        if abs(val_tot_logged - val_tot_recomp) > 1e-3:
            print(f"  [Warning] Val   L_total differs from the recomputed value: logged={val_tot_logged:.6f}, recompute={val_tot_recomp:.6f}")

        if train_tot_logged > max(5.0, 5.0 * val_tot_logged):
            print(
                "  [Training loss spike breakdown] "
                f"L_total={train_tot_logged:.4f}, L_data={train_Ldata_:.4f}, "
                f"λm*L_mass={lambda_mass*train_Lmass_:.4f}, λe*L_energy={lambda_energy*train_Lenergy_:.4f}, "
                f"λin*L_inflow={lambda_inflow*train_Linflow_:.4f}, other={train_other:.4f} | "
                f"LC={train_LC_:.4f}, Lbeta={train_Lbeta_:.4f}, "
                f"Lclosure={train_Lclosure_:.4f}, Lleak={train_Lleak_:.4f}, Lgmass={train_Lgmass_:.4f}, L_leak_mass={train_Lleakmass_:.4f}, L_leak_prior={train_Lleakprior_:.4f}, "
                f"Ldmd={train_Ldemand_:.4f}, LpData={train_Lpressdata_:.4f}"
            )

        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch},"
                f"{lambda_mass:.6f},{lambda_energy:.6f},{lambda_pressure:.6f},{lambda_buffer:.6f},{lambda_inflow:.6f},"
                f"{_f(train_metrics,'inflow_n_total'):.6f},{_f(train_metrics,'inflow_n_kept'):.6f},{_f(train_metrics,'inflow_keep_rate'):.6f},{_f(train_metrics,'inflow_logratio_median'):.6f},"
                f"{train_tot_logged:.6f},{train_tot_recomp:.6f},{train_Ldata_:.6f},{train_Lmass_:.6f},{train_Lenergy_:.6f},{train_Lpressure_:.6f},{train_Lpressdata_:.6f},{train_Linflow_:.6f},"
                f"{train_Lclosure_:.6f},{train_Lleak_:.6f},{train_Lgmass_:.6f},{train_Lleakmass_:.6f},{train_Lleakprior_:.6f},{train_LC_:.6f},{train_Lbeta_:.6f},{train_Ldemand_:.6f},{train_other:.6f},"
                f"{_f(val_metrics,'inflow_n_total'):.6f},{_f(val_metrics,'inflow_n_kept'):.6f},{_f(val_metrics,'inflow_keep_rate'):.6f},{_f(val_metrics,'inflow_logratio_median'):.6f},"
                f"{val_tot_logged:.6f},{val_tot_recomp:.6f},{val_Ldata_:.6f},{val_Lmass_:.6f},{val_Lenergy_:.6f},{val_Lpressure_:.6f},{val_Lpressdata_:.6f},{val_Linflow_:.6f},"
                f"{val_Lclosure_:.6f},{val_Lleak_:.6f},{val_Lgmass_:.6f},{val_Lleakmass_:.6f},{val_Lleakprior_:.6f},{val_LC_:.6f},{val_Lbeta_:.6f},{val_Ldemand_:.6f},{val_other:.6f}\n"
            )

        if val_metrics['L_total'] < best_val:
            best_val = val_metrics['L_total']
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, bad_epochs, CKPT_PATH)

            torch.save(orig_model.state_dict(), BEST_PATH)
            best_model_saved = True
            print(f"  ✓ Best model checkpoint saved to {BEST_PATH.name} (Epoch {epoch}, Val L_tot={best_val:.4f})")

            print(f"  ✓ New best model (Val L_tot={best_val:.4f})")
        else:
            bad_epochs += 1
            print(f"  Bad epochs: {bad_epochs}/{PATIENCE}")

        if epoch >= EARLY_STOP_MIN_EPOCH and bad_epochs >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Training ended.")
            break

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            sample_batch = next(iter(active_val_loader))
            sample_batch.lambda_mass = lambda_mass
            sample_batch.lambda_energy = lambda_energy
            sample_batch = sample_batch.to(DEVICE)
            with torch.no_grad():
                sample_out = model(sample_batch)
            diagnose_physics(model, sample_batch, sample_out, epoch)

    if not best_model_saved:
        torch.save(orig_model.state_dict(), BEST_PATH)
        print(f"\n⚠ Warning: No best-model checkpoint found. Current model saved to {BEST_PATH.name}")

    print("\n=== Training complete ===")
    print(f"Best checkpoint: Epoch {best_epoch}, Val L_tot={best_val:.4f} -> {BEST_PATH.name}")

if __name__ == "__main__":
    main()
