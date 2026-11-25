from __future__ import annotations
from typing import Dict, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


def hw_headloss(L, Q_m3ps, d, C, eps=1e-9):
    C = torch.clamp(C, min=10.0)
    d = torch.clamp(d, min=0.01)
    L = torch.clamp(L, min=0.1)
    Q_abs = torch.clamp(torch.abs(Q_m3ps), min=eps)

    h_loss = 10.67 * L * (Q_abs ** 1.852) / ((C ** 1.852) * (d ** 4.871) + eps)
    return h_loss

def local_headloss(Q_m3ps, opening, K_coeff, eps=1e-9):
    """Retain the legacy zero-return helper for compatibility.

    The model computes local loss in PI_GAT.energy_residual as K times
    velocity head. This helper does not implement that calculation."""
    Q_abs = torch.clamp(torch.abs(Q_m3ps), min=eps)
    open_val = torch.clamp(opening, min=0.0, max=1.0)
    return K_coeff * (Q_abs ** 2) * 0.0 + open_val * 0.0

class PI_GAT(nn.Module):
    """Physics-informed graph attention network for an irrigation pipeline.

    Physical parameters use bounded differentiable parameterisations."""

    def __init__(
        self,
        win: int,
        num_nodes: int,
        hidden: int = 64,
        heads: int = 8,
        dropout: float = 0.1,
        in_feat: int = 2
    ):
        super().__init__()
        in_dim = win * in_feat

        self.lin_in = nn.Linear(in_dim, hidden, bias=True)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=False)
        self.bn1 = BatchNorm(hidden)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=False)
        self.bn2 = BatchNorm(hidden)

        self.node_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.pressure_scale = nn.Parameter(torch.tensor(60.0))
        self.pressure_base = nn.Parameter(torch.tensor(30.0))

        self.resistance_beta_logit = nn.Parameter(torch.tensor(0.0))

        self._valve_K_max = 1.0e8
        self.register_buffer(
            "_valve_open_pts",
            torch.tensor([0.00, 0.20, 0.40, 0.60, 0.80, 1.00], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_valve_K_pts",
            torch.tensor([self._valve_K_max, 10000.0, 125.0, 17.0, 4.5, 1.8], dtype=torch.float32),
            persistent=False,
        )

        self.leakage_logit = nn.Parameter(torch.tensor(0.0))

        # Use a separate scale offset for each open-bin range.
        self.num_nodes = int(num_nodes)
        self.bias_num_bins = 3
        self.bias_mu_raw = nn.Parameter(torch.zeros(self.bias_num_bins))  # bins 1-3
        self.bias_delta_raw = nn.Parameter(torch.zeros(num_nodes, self.bias_num_bins))

        # Coverage weights for the bias gauge.
        self.register_buffer(
            "bias_gauge_weight",
            torch.zeros(num_nodes, self.bias_num_bins, dtype=torch.float32),
            persistent=True,
        )

        self.latent_demand_logit = nn.Parameter(torch.zeros(num_nodes) - 5.0)

        self.dropout = dropout
        self.hidden = hidden

        self.C_range = (140.0, 120.0)
        self.alpha_range = (0.10, 0.20)
        self._bias_mu_range = (-0.30, 0.30)
        self._bias_delta_range = (-0.05, 0.05)
        self._leakage_range = (0.05, 0.15)

        # Effective opening is used in the energy loss.
        self._eff_open_enabled = False
        self.register_buffer("_eff_open_y_knots", torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_effective_opening_mapping(self, y_knots01: torch.Tensor) -> None:
        """Set per-edge effective openings at the valve-curve knots.

        The tensor is shaped [edges, knots] with values in [0, 1]. The
        mapping applies only to the energy residual and its opening gate."""
        if not isinstance(y_knots01, torch.Tensor):
            self._eff_open_enabled = False
            dev = getattr(self._valve_open_pts, 'device', None)
            if dev is None:
                self._eff_open_y_knots = torch.empty(0)
            else:
                self._eff_open_y_knots = torch.empty(0, device=dev)
            return
        if y_knots01.ndim != 2:
            self._eff_open_enabled = False
            dev = getattr(self._valve_open_pts, 'device', None)
            if dev is None:
                self._eff_open_y_knots = torch.empty(0)
            else:
                self._eff_open_y_knots = torch.empty(0, device=dev)
            return
        valve_open_pts_numel: int = int(self._valve_open_pts.numel())  # type: ignore
        if int(y_knots01.shape[1]) != valve_open_pts_numel:  # type: ignore
            self._eff_open_enabled = False
            dev = getattr(self._valve_open_pts, 'device', None)
            if dev is None:
                self._eff_open_y_knots = torch.empty(0)
            else:
                self._eff_open_y_knots = torch.empty(0, device=dev)
            return

        dev = getattr(self._valve_open_pts, 'device', None)  # type: ignore
        if dev is None:
            y_step1 = y_knots01.to(dtype=torch.float32)  # type: ignore
            y_step2: torch.Tensor = cast(torch.Tensor, y_step1)  # type: ignore
            y = torch.clamp(y_step2, 0.0, 1.0)  # type: ignore
        else:
            y_step1 = y_knots01.to(device=dev, dtype=torch.float32)  # type: ignore
            y_step2: torch.Tensor = cast(torch.Tensor, y_step1)  # type: ignore
            y = torch.clamp(y_step2, 0.0, 1.0)  # type: ignore
        self._eff_open_y_knots = y  # type: ignore
        self._eff_open_enabled = True

    def _apply_effective_opening(self, opening01: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        """Map measured openings to effective openings for the energy residual."""
        if not self._eff_open_enabled:
            return opening01
        if not isinstance(opening01, torch.Tensor) or not isinstance(edge_ids, torch.Tensor):
            return opening01
        if self._eff_open_y_knots.numel() == 0:
            return opening01

        valve_open_pts_numel_val: int = int(self._valve_open_pts.numel())  # type: ignore
        K = valve_open_pts_numel_val
        if int(self._eff_open_y_knots.shape[1]) != K:  # type: ignore
            return opening01

        dev = getattr(self._eff_open_y_knots, 'device', None)
        if dev is None:
            edge_to_step = edge_ids.to(dtype=torch.long)  # type: ignore
            edge_ids_cast = cast(torch.Tensor, edge_to_step)  # type: ignore
            edge_ids_out = edge_ids_cast.view(-1)  # type: ignore
        else:
            edge_to_step = edge_ids.to(device=dev, dtype=torch.long)  # type: ignore
            edge_ids_cast = cast(torch.Tensor, edge_to_step)  # type: ignore
            edge_ids_out = edge_ids_cast.view(-1)  # type: ignore
        edge_ids = edge_ids_out
        if edge_ids.numel() != opening01.numel():
            return opening01

        y_knots = self._eff_open_y_knots.index_select(0, edge_ids)  # [E, K]
        x_knots_dtype = y_knots.dtype  # type: ignore
        x_knots_dev = y_knots.device  # type: ignore
        x_knots_temp = self._valve_open_pts.to(dtype=x_knots_dtype, device=x_knots_dev)  # type: ignore
        x_knots: torch.Tensor = cast(torch.Tensor, x_knots_temp)  # type: ignore

        x_dtype_open = y_knots.dtype  # type: ignore
        x_dev_open = y_knots.device  # type: ignore
        x_to_step = opening01.to(device=x_dev_open, dtype=x_dtype_open)  # type: ignore
        x_to_cast = cast(torch.Tensor, x_to_step)  # type: ignore
        x_view_step = x_to_cast.view(-1)  # type: ignore
        x = torch.clamp(x_view_step, 0.0, 1.0)  # type: ignore
        idx = torch.bucketize(x, x_knots) - 1  # type: ignore
        idx = torch.clamp(idx, 0, K - 2)

        x0 = x_knots[idx]
        x1 = x_knots[idx + 1]
        y0 = torch.gather(y_knots, 1, idx.view(-1, 1)).view(-1)
        y1 = torch.gather(y_knots, 1, (idx + 1).view(-1, 1)).view(-1)

        t = (x - x0) / (x1 - x0 + 1e-12)
        y = y0 + t * (y1 - y0)
        return torch.clamp(y, 0.0, 1.0)

    @torch.no_grad()
    def set_bias_gauge_weight(self, weight: torch.Tensor) -> None:
        """Set coverage weights of shape [nodes, bias bins] for weighted centring."""
        if not isinstance(weight, torch.Tensor):
            return
        if weight.ndim != 2:
            return
        if int(weight.shape[0]) != int(self.num_nodes) or int(weight.shape[1]) != int(self.bias_num_bins):
            return
        dev = getattr(self.bias_gauge_weight, 'device', None)
        target_dtype = self.bias_gauge_weight.dtype  # type: ignore
        if dev is None:
            weight_converted = weight.to(dtype=target_dtype)  # type: ignore
            self.bias_gauge_weight.copy_(weight_converted)  # type: ignore
        else:
            weight_converted = weight.to(device=dev, dtype=target_dtype)  # type: ignore
            self.bias_gauge_weight.copy_(weight_converted)  # type: ignore

    def valve_K_from_opening(self, opening01: torch.Tensor) -> torch.Tensor:
        """Evaluate the dimensionless valve-loss coefficient for openings in [0, 1].

        Interpolate log(K) between fixed anchors, using a finite K_max at closure."""
        x = torch.clamp(opening01, min=0.00, max=1.00)  # type: ignore
        x_dtype_valve_local = x.dtype  # type: ignore
        x_device_valve_local = x.device  # type: ignore
        xp_temp = self._valve_open_pts  # type: ignore
        xp_to_result = xp_temp.to(dtype=x_dtype_valve_local, device=x_device_valve_local)  # type: ignore
        xp = cast(torch.Tensor, xp_to_result)  # type: ignore
        yp_log_result = torch.log(self._valve_K_pts.to(dtype=x_dtype_valve_local, device=x_device_valve_local))  # type: ignore
        yp = cast(torch.Tensor, yp_log_result)  # type: ignore

        # idx in [0, len(xp)-2]
        idx = torch.bucketize(x, xp) - 1  # type: ignore
        xp_numel_result: int = int(xp.numel())  # type: ignore
        idx = torch.clamp(idx, 0, xp_numel_result - 2)  # type: ignore
        x0_idx: torch.Tensor = cast(torch.Tensor, xp[idx])  # type: ignore
        idx_add1 = cast(torch.Tensor, idx + 1)  # type: ignore
        x1_idx: torch.Tensor = cast(torch.Tensor, xp[idx_add1])  # type: ignore
        y0 = cast(torch.Tensor, yp[idx])  # type: ignore
        y1_idx = cast(torch.Tensor, idx + 1)  # type: ignore
        y1 = cast(torch.Tensor, yp[y1_idx])  # type: ignore
        t = (x - x0_idx) / (x1_idx - x0_idx + 1e-12)  # type: ignore
        y = y0 + t * (y1 - y0)  # type: ignore
        return torch.exp(y)

    @staticmethod
    def _total_inflow_keep_mask_from_logratio(
        pred_raw: torch.Tensor,
        obs: torch.Tensor,
        *,
        abs_obs_min: float = 0.02,
        z_thresh: float = 6.0,
        min_keep_frac: float = 0.8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Screen inlet observations by absolute flow and robust log-ratio deviation.

        Reject small absolute observations, then compare log absolute prediction
        with log absolute observation. Keep the eligible samples if robust filtering
        would retain fewer than the requested fraction."""
        if obs.numel() == 0:
            return torch.zeros_like(obs, dtype=torch.bool)

        keep = obs.abs() >= float(abs_obs_min)
        if int(keep.sum().item()) == 0:
            return keep

        pred_v = pred_raw[keep]
        obs_v = obs[keep]
        if obs_v.numel() < 5:
            return keep

        y = torch.log(torch.clamp(pred_v.abs(), min=eps)) - torch.log(torch.clamp(obs_v.abs(), min=eps))
        med = y.median()
        mad = (y - med).abs().median()
        scale = torch.clamp(1.4826 * mad, min=eps)
        z = (y - med).abs() / scale
        keep_local = z <= float(z_thresh)

        min_keep = max(1, int(float(min_keep_frac) * float(obs_v.numel())))
        if int(keep_local.sum().item()) < min_keep:
            return keep

        keep2 = keep.clone()
        keep2[keep] = keep_local
        return keep2

    def get_leakage_rate(self):  # type: ignore
        """Return the network leakage fraction bounded to the configured interval."""
        lo, hi = self._leakage_range
        return lo + (hi - lo) * torch.sigmoid(self.leakage_logit)  # type: ignore

    def get_bias_params(self):
        """Return common and node-specific scale offsets by opening bin.

        The common offset has shape [bins], and the node-specific offsets have
        shape [nodes, bins]. Coverage-weighted centring resolves the freedom to
        add a constant to the common offset and subtract it from every node offset."""
        mu_max = self._bias_mu_range[1]
        delta_max = self._bias_delta_range[1]

        mu = mu_max * torch.tanh(self.bias_mu_raw)  # [B]
        delta = delta_max * torch.tanh(self.bias_delta_raw)  # [N,B]

        w = getattr(self, 'bias_gauge_weight', None)
        if not isinstance(w, torch.Tensor) or w.numel() == 0:
            return mu, delta

        w = torch.clamp(w.to(device=delta.device, dtype=delta.dtype), min=0.0)
        sum_w = w.sum(dim=0, keepdim=True)  # [1,B]
        if float(sum_w.max().detach().cpu().item()) <= 0.0:
            return mu, delta

        mean = (w * delta).sum(dim=0, keepdim=True) / torch.clamp(sum_w, min=1e-12)  # [1,B]
        delta_centered = delta - mean
        return mu, delta_centered

    def get_bias_delta_gauge_mean(self) -> torch.Tensor:
        """Return the coverage-weighted mean of the node offsets for diagnostics."""
        delta_max = self._bias_delta_range[1]
        delta = delta_max * torch.tanh(self.bias_delta_raw)  # [N,B]
        w = getattr(self, 'bias_gauge_weight', None)
        if not isinstance(w, torch.Tensor) or w.numel() == 0:
            return torch.zeros((self.bias_num_bins,), device=delta.device, dtype=delta.dtype)
        w = torch.clamp(w.to(device=delta.device, dtype=delta.dtype), min=0.0)
        sum_w = w.sum(dim=0)  # [B]
        mean = (w * delta).sum(dim=0) / torch.clamp(sum_w, min=1e-12)
        mean = torch.where(sum_w > 0, mean, torch.zeros_like(mean))
        return mean

    def encode(self, x, edge_index):
        """Encode graph node features with two attention layers."""
        h = self.lin_in(x)
        h = F.elu(h)
        h = self.gat1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        return h

    def decode_node(self, h):
        """Predict standardised nodal flow."""
        return self.node_head(h).squeeze(-1)

    def decode_edge(self, h, edge_index, edge_attr):
        """Predict standardised pipe flow."""
        src, dst = edge_index
        hs = h[src]
        hd = h[dst]
        w = edge_attr[:, 3:4]
        ed_in = torch.cat([hs, hd, w], dim=-1)

        out = self.edge_mlp(ed_in)  # shape: [E, 1]
        flow_z = out.squeeze(-1)
        return flow_z

    def decode_pressure(self, h, node_elev):
        """Return total head and pressure head predicted at each node."""
        p_raw = self.pressure_head(h).squeeze(-1)

        scale = torch.clamp(self.pressure_scale, min=20.0, max=100.0)
        base = torch.clamp(self.pressure_base, min=10.0, max=50.0)

        elev_min = node_elev.min()
        elev_max = node_elev.max()
        elev_range = elev_max - elev_min + 1e-6
        elev_inv_normalized = (elev_max - node_elev) / elev_range

        p_baseline = base + elev_inv_normalized * scale * 0.5
        p_adjustment = torch.tanh(p_raw) * scale * 0.3
        p_head = p_baseline + p_adjustment
        p_head = torch.clamp(p_head, min=5.0, max=100.0)
        H_total = node_elev + p_head

        return H_total, p_head

    def get_physics_params(self):
        """Return the differentiably bounded hydraulic parameters."""
        beta = torch.sigmoid(self.resistance_beta_logit)

        # beta=0 -> C=140; beta=1 -> C=120
        C_start, C_end = self.C_range
        C_val = C_start + (C_end - C_start) * beta

        # beta=0 -> alpha=1.05; beta=1 -> alpha=1.15
        a_start, a_end = self.alpha_range
        alpha_val = a_start + (a_end - a_start) * beta

        return {
            "C_main": C_val,
            "C_branch": C_val + 20.0,
            "alpha": alpha_val,
        }

    def get_raw_params(self):
        """Return raw model parameters for diagnostics."""
        return {
            "resistance_beta_logit": self.resistance_beta_logit.item(),
            "leakage_logit": self.leakage_logit.item(),
        }

    def get_meter_scale(self, node_type, edge_opening=None):
        """Compute the observation scale in y_obs = y_phys * scale.

        Node types 0, 1 and 2 indicate an intermediate node, a source and a terminal."""
        if node_type is None:
            if edge_opening is None:
                dev = getattr(self.bias_mu_raw, 'device', None)
                if dev is None:
                    return torch.ones(self.num_nodes)
                return torch.ones(self.num_nodes, device=dev)
            if isinstance(edge_opening, torch.Tensor):
                return torch.ones_like(edge_opening)
            dev = getattr(self.bias_mu_raw, 'device', None)
            if dev is None:
                return torch.ones(self.num_nodes)
            return torch.ones(self.num_nodes, device=dev)

        if edge_opening is None:
            return torch.ones_like(node_type, dtype=torch.float, device=node_type.device)

        if isinstance(edge_opening, torch.Tensor) and edge_opening.shape == node_type.shape:
            return self.get_meter_scale_from_opening(node_type, edge_opening)

        return torch.ones_like(node_type, dtype=torch.float, device=node_type.device)

    def get_meter_scale_from_opening(self, node_type, node_opening):
        """Return the terminal meter scale for the corresponding opening bin."""
        if node_type is None:
            return torch.ones_like(node_opening)

        device = node_type.device
        scale = torch.ones_like(node_opening, device=device)

        is_sink = (node_type == 2)
        if not is_sink.any():
            return scale

        opening_pct = torch.clamp(node_opening, 0.0, 1.0) * 100.0

        # bins:  0-5%: bin0, (Bias=0), 6-33%: bin1, 34-66%: bin2, 67-100%: bin3
        bin_idx = torch.zeros_like(opening_pct, dtype=torch.long)
        bin_idx = bin_idx + (opening_pct > 5.0).long() + (opening_pct > 33.3333).long() + (opening_pct > 66.6667).long()

        mu, delta = self.get_bias_params()

        N_total = node_type.shape[0]
        phys_idx = torch.remainder(torch.arange(N_total, device=device), self.num_nodes)

        apply_mask = is_sink & (bin_idx > 0)
        if apply_mask.any():
            b = (bin_idx[apply_mask] - 1).clamp(min=0, max=self.bias_num_bins - 1)
            mu_sel = mu[b]
            delta_sel = delta[phys_idx[apply_mask], b]
            bias_frac = mu_sel + delta_sel
            scale[apply_mask] = 1.0 + bias_frac

        return scale

    def get_latent_demand(self, node_type, node_mask=None):
        """Compute the non-negative auxiliary demand where an observation is unavailable."""
        raw_demand = F.softplus(self.latent_demand_logit)


        if node_type is not None:
            if raw_demand.shape[0] != node_type.shape[0]:
                 num_graphs = node_type.shape[0] // raw_demand.shape[0]
                 raw_demand = raw_demand.repeat(num_graphs)  # type: ignore

            is_sink = (node_type == 2)
            demand = torch.where(is_sink, raw_demand, torch.zeros_like(raw_demand))

            if node_mask is not None:
                demand = torch.where(~node_mask, demand, torch.zeros_like(demand))

            return demand
        return raw_demand

    def mass_residual(self, edge_index, q_edge, num_nodes, node_type=None, y_node_obs=None, node_mask=None):
        src, dst = edge_index
        device = q_edge.device

        sum_in = torch.zeros(num_nodes, device=device).index_add_(0, dst, q_edge)
        sum_out = torch.zeros(num_nodes, device=device).index_add_(0, src, q_edge)
        net_flow_accum = sum_in - sum_out

        latent_demand = self.get_latent_demand(node_type, node_mask)

        # Use observed demand where a node label is present.
        total_demand = latent_demand.clone()

        if y_node_obs is not None and node_mask is not None:
            total_demand = torch.where(node_mask, y_node_obs, latent_demand)


        r_mass = net_flow_accum - total_demand

        if node_type is not None:
            is_source = (node_type == 1)
            r_mass = torch.where(is_source, torch.zeros_like(r_mass), r_mass)

        return r_mass, sum_in, sum_out, net_flow_accum, latent_demand

    def enforce_mass_conservation(
        self,
        edge_index,
        q_pred,
        demand,
        num_nodes,
        edge_opening=None,
        iterations=3,
        hard_bc_edge_idx=None,
        hard_bc_edge_value=None,
    ):
        """Project pipe flows towards the nodal balance by opening-weighted sweeps.

        Closed pipes receive no redistribution and a valid inlet measurement can
        be imposed as a fixed boundary after each sweep."""
        src, dst = edge_index
        q_corrected = q_pred.clone()
        device = q_pred.device

        fixed_mask = None
        fixed_values = None
        if hard_bc_edge_idx is not None and hard_bc_edge_value is not None:
            try:
                idx = hard_bc_edge_idx.to(device).view(-1).long()
                val = hard_bc_edge_value.to(device).view(-1)
                if idx.numel() > 0:
                    valid = (idx >= 0) & (idx < q_corrected.numel())
                    idx = idx[valid]
                    val = val[valid]
                    if idx.numel() > 0:
                        fixed_mask = torch.zeros_like(q_corrected)
                        fixed_values = torch.zeros_like(q_corrected)
                        fixed_mask[idx] = 1.0
                        fixed_values[idx] = val
                        q_corrected = q_corrected * (1.0 - fixed_mask) + fixed_values
            except Exception:
                fixed_mask = None
                fixed_values = None

        if edge_opening is not None:
            conductance = torch.clamp(edge_opening, min=0.0, max=1.0)
        else:
            conductance = torch.ones_like(q_pred)

        for _ in range(iterations):
            sum_in = torch.zeros(num_nodes, device=device).index_add_(0, dst, q_corrected)
            sum_out = torch.zeros(num_nodes, device=device).index_add_(0, src, q_corrected)
            residual = (sum_in - sum_out) - demand

            grad_edges = residual[dst] - residual[src]

            update = 0.2 * grad_edges * conductance

            q_corrected = q_corrected - update

            if fixed_mask is not None and fixed_values is not None:
                q_corrected = q_corrected * (1.0 - fixed_mask) + fixed_values

        return q_corrected

    def energy_residual(
        self,
        edge_index,
        edge_attr,
        q_edge_m3ps,
        H_node,
        edge_opening=None,
        node_type=None,
    ):
        src, dst = edge_index
        L = edge_attr[:, 0]
        d = edge_attr[:, 1]

        params = self.get_physics_params()
        C_main = params["C_main"]
        C_branch = params["C_branch"]
        alpha = params["alpha"]

        is_main = d > 0.35
        C_edge = torch.where(is_main, C_main, C_branch)

        if edge_opening is None:
            edge_opening = torch.ones_like(L)

        h_friction = hw_headloss(L, q_edge_m3ps, d, C_edge)

        area = 3.14159 * (d / 2.0) ** 2
        velocity = q_edge_m3ps / (area + 1e-9)
        velocity_head = (velocity ** 2) / (2 * 9.81)

        # h_local = K(opening) * v^2/(2g)
        safe_opening = torch.clamp(edge_opening, min=0.0, max=1.0)
        K_open = self.valve_K_from_opening(safe_opening)
        h_local = K_open * velocity_head

        is_exit_edge = torch.zeros_like(L, dtype=torch.bool)
        if node_type is not None:
            dst_types = node_type[dst]
            is_exit_edge = (dst_types == 2)

        h_exit = torch.where(is_exit_edge, velocity_head, torch.zeros_like(velocity_head))

        # Apply the sign of flow to the head loss.
        eps = 1e-9
        q_abs = torch.abs(q_edge_m3ps) + eps
        s = q_edge_m3ps / q_abs

        h_loss_pos = (1.0 + alpha) * h_friction + h_local + h_exit
        h_loss_signed = s * h_loss_pos

        delta_H = H_node[src] - H_node[dst]

        valid_physics = torch.sigmoid((edge_opening - 0.20) * 50)

        r_energy = (delta_H - h_loss_signed) * valid_physics

        return r_energy, h_friction, h_local, h_exit, delta_H

    def forward(self, data):
        """Encode the graph, decode hydraulic quantities and evaluate physical residuals."""
        x = data.x
        ei = data.edge_index
        ea = data.edge_attr
        N = x.size(0)
        device = x.device

        h = self.encode(x, ei)

        node_hat_z = self.decode_node(h)
        edge_hat_z = self.decode_edge(h, ei, ea)
        H_node, p_head = self.decode_pressure(h, data.node_elev)

        node_type = getattr(data, 'node_type', None)
        node_opening = torch.zeros(N, device=device)
        if node_type is not None and hasattr(data, 'edge_opening'):
            is_self_full = (ea[:, 5] > 0.5)
            mask_e_full = ~is_self_full
            ei_full = ei[:, mask_e_full]
            open_full = data.edge_opening[mask_e_full]

            try:
                node_opening.scatter_reduce_(0, ei_full[1], open_full, reduce='amax', include_self=True)
            except Exception:
                dst_idx = ei_full[1]
                for k in range(open_full.numel()):
                    d_idx = int(dst_idx[k].item())
                    node_opening[d_idx] = torch.maximum(node_opening[d_idx], open_full[k])

        bias_bin_idx = None
        bias_apply_mask = None
        if node_type is not None:
            opening_pct = torch.clamp(node_opening, 0.0, 1.0) * 100.0
            bias_bin_idx = torch.zeros_like(opening_pct, dtype=torch.long)
            bias_bin_idx = bias_bin_idx + (opening_pct > 5.0).long() + (opening_pct > 33.3333).long() + (opening_pct > 66.6667).long()
            bias_apply_mask = (node_type == 2) & (bias_bin_idx > 0)

        meter_scale = self.get_meter_scale_from_opening(node_type, node_opening) if node_type is not None else torch.ones(N, device=device)

        if hasattr(data, "edge_mean") and hasattr(data, "edge_std"):
            edge_hat_m3ph = edge_hat_z * data.edge_std + data.edge_mean
        else:
            edge_hat_m3ph = edge_hat_z
        edge_hat_m3ps = edge_hat_m3ph / 3600.0  # [Initial Prediction]

        node_mask = getattr(data, 'node_label_mask', None)
        latent_demand = self.get_latent_demand(node_type, node_mask)

        y_node_phys = None
        total_demand = latent_demand.clone() # Start with latent

        if hasattr(data, "y_node") and hasattr(data, "node_mean") and hasattr(data, "node_std"):
            y_node_phys = data.y_node * data.node_std + data.node_mean
            y_node_phys = y_node_phys / 3600.0

            y_node_phys_true = y_node_phys / torch.clamp(meter_scale, min=1e-6)

            if node_mask is not None:
                # Blend terminal demand using graph-level credibility weights.
                w_node = None
                if hasattr(data, 'credibility_weight') and hasattr(data, 'batch'):
                    try:
                        w_graph = data.credibility_weight.to(device=device, dtype=torch.float32).view(-1)
                        if w_graph.numel() > 0 and isinstance(data.batch, torch.Tensor) and data.batch.numel() == N:
                            w_node = w_graph[data.batch].clamp(0.0, 1.0)
                    except Exception:
                        w_node = None

                if w_node is None:
                    total_demand = torch.where(node_mask, y_node_phys_true, latent_demand)
                else:
                    mixed = w_node * y_node_phys_true + (1.0 - w_node) * latent_demand
                    total_demand = torch.where(node_mask, mixed, latent_demand)

        sink_open_mask = None
        if node_type is not None and hasattr(data, 'edge_opening'):
            is_self_full = (ea[:, 5] > 0.5)
            mask_e_full = ~is_self_full
            ei_full = ei[:, mask_e_full]
            open_full = data.edge_opening[mask_e_full]

            max_in_open = torch.zeros(N, device=device)
            try:
                max_in_open.scatter_reduce_(0, ei_full[1], open_full, reduce='amax', include_self=True)
            except Exception:
                dst_idx = ei_full[1]
                for k in range(open_full.numel()):
                    d_idx = int(dst_idx[k].item())
                    max_in_open[d_idx] = torch.maximum(max_in_open[d_idx], open_full[k])

            sink_open_mask = (max_in_open > 0.05)

            is_sink = (node_type == 2)
            total_demand = torch.where(is_sink & (~sink_open_mask), torch.zeros_like(total_demand), total_demand)

        # Project pipe discharges towards continuity.
        edge_open_val = None
        if hasattr(data, 'edge_opening'):
            edge_open_val = data.edge_opening

        hard_bc_idx = None
        hard_bc_val = None
        if (
            (hasattr(data, 'total_inflow_hard_mask') or hasattr(data, 'total_inflow_mask'))
            and hasattr(data, 'total_inflow_m3ps')
            and hasattr(data, 'total_inflow_edge_idx')
        ):
            try:
                if hasattr(data, 'batch') and isinstance(data.batch, torch.Tensor) and data.batch.numel() > 0:
                    num_graphs = int(data.batch.max().item()) + 1
                else:
                    num_graphs = 1

                if hasattr(data, 'batch') and isinstance(data.batch, torch.Tensor) and data.batch.numel() > 0:
                    edge_graph = data.batch[ei[0]]
                else:
                    edge_graph = torch.zeros(ei.size(1), device=device, dtype=torch.long)

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

                inflow_edge_idx = data.total_inflow_edge_idx
                if isinstance(inflow_edge_idx, int):
                    inflow_edge_idx = torch.full((num_graphs,), inflow_edge_idx, device=device, dtype=torch.long)
                elif isinstance(inflow_edge_idx, torch.Tensor):
                    inflow_edge_idx = inflow_edge_idx.to(device).view(-1).long()
                    if inflow_edge_idx.numel() == 1 and num_graphs > 1:
                        inflow_edge_idx = inflow_edge_idx.repeat(num_graphs)
                else:
                    inflow_edge_idx = torch.full((num_graphs,), -1, device=device, dtype=torch.long)

                mask = getattr(data, 'total_inflow_hard_mask', getattr(data, 'total_inflow_mask', None))
                if isinstance(mask, torch.Tensor):
                    valid = mask.to(device).view(-1)
                else:
                    valid = torch.zeros((num_graphs,), device=device, dtype=torch.bool)

                if valid.numel() != num_graphs:
                    valid = torch.zeros((num_graphs,), device=device, dtype=torch.bool)

                valid = valid & (inflow_edge_idx >= 0) & (inflow_edge_idx < edge_counts)
                if valid.any():
                    obs_all = data.total_inflow_m3ps.to(device).view(-1)
                    valid = valid & (torch.abs(obs_all) >= 0.02)

                    if valid.any():
                        global_edge_idx = edge_starts + inflow_edge_idx
                        hard_bc_idx = global_edge_idx[valid]
                        hard_bc_val = obs_all[valid]
            except Exception:
                hard_bc_idx = None
                hard_bc_val = None

        edge_hat_m3ps_raw = edge_hat_m3ps
        edge_hat_m3ps = self.enforce_mass_conservation(
            ei,
            edge_hat_m3ps,
            total_demand,
            N,
            edge_opening=edge_open_val,
            iterations=3,
            hard_bc_edge_idx=hard_bc_idx,
            hard_bc_edge_value=hard_bc_val,
        )

        edge_hat_m3ph = edge_hat_m3ps * 3600.0
        if hasattr(data, "edge_std") and hasattr(data, "edge_mean"):
            edge_hat_z = (edge_hat_m3ph - data.edge_mean) / (data.edge_std + 1e-9)

        node_hat_m3ph = torch.zeros(N, device=device)
        node_hat_m3ps = torch.zeros(N, device=device)
        if hasattr(data, "node_std") and hasattr(data, "node_mean"):
            node_hat_m3ph = node_hat_z * data.node_std + data.node_mean
            node_hat_m3ps = node_hat_m3ph / 3600.0

        node_hat_m3ps_biased = node_hat_m3ps * meter_scale

        node_hat_z_biased = torch.zeros_like(node_hat_z)
        if hasattr(data, "node_std") and hasattr(data, "node_mean"):
             # m3/s -> m3/h -> z-score
             node_hat_m3ph_biased = node_hat_m3ps_biased * 3600.0
             node_hat_z_biased = (node_hat_m3ph_biased - data.node_mean) / (data.node_std + 1e-9)

        r_mass = torch.zeros(N, device=device)
        r_energy = torch.zeros(ei.size(1), device=device)
        valid_mass_mask = torch.zeros(N, device=device, dtype=torch.bool)
        demand = torch.zeros(N, device=device)
        net_flow = torch.zeros(N, device=device)
        sum_in = torch.zeros(N, device=device)
        sum_out = torch.zeros(N, device=device)

        h_loss_pos = torch.tensor(0.0, device=device)
        h_loss_signed = torch.tensor(0.0, device=device)
        delta_H = torch.tensor(0.0, device=device)
        h_exit = torch.tensor(0.0, device=device)

        need_physics = (
            hasattr(data, "lambda_mass") and
            hasattr(data, "lambda_energy") and
            (data.lambda_mass > 0 or data.lambda_energy > 0)
        )

        params = self.get_physics_params()

        if need_physics:
            is_self = (ea[:, 5] > 0.5)
            mask_e = ~is_self
            ei_phys = ei[:, mask_e]
            q_phys = edge_hat_m3ps[mask_e]
            ea_phys = ea[mask_e]

            open_phys = None
            if hasattr(data, "edge_opening"):
                open_phys_raw = data.edge_opening[mask_e]
                edge_ids_full = torch.arange(ea.size(0), device=device, dtype=torch.long)
                E_base = int(self._eff_open_y_knots.shape[0])
                if E_base > 0:
                    edge_ids_full = torch.remainder(edge_ids_full, E_base)
                edge_ids = edge_ids_full[mask_e]
                open_phys = self._apply_effective_opening(open_phys_raw, edge_ids)

            y_node_phys = None
            if hasattr(data, "y_node") and hasattr(data, "node_mean") and hasattr(data, "node_std"):
                y_node_phys = data.y_node * data.node_std + data.node_mean
                y_node_phys = y_node_phys / 3600.0  # m³/h -> m³/s

                y_node_phys = y_node_phys / torch.clamp(meter_scale, min=1e-6)

            r_mass, sum_in, sum_out, net_flow, _ = self.mass_residual(
                ei_phys, q_phys, N,
                node_type=node_type,
                y_node_obs=y_node_phys,
                node_mask=node_mask
            )

            if node_type is not None:
                mask_mid = (node_type == 0)

                mask_sink_with_label = (node_type == 2) & node_mask
                if sink_open_mask is not None:
                    mask_sink_with_label = mask_sink_with_label & sink_open_mask

                valid_mass_mask = mask_mid | mask_sink_with_label
            else:
                valid_mass_mask = torch.ones(N, device=device, dtype=torch.bool)

            open_phys = None
            if hasattr(data, "edge_opening"):
                open_phys_raw = data.edge_opening[mask_e]
                edge_ids_full = torch.arange(ea.size(0), device=device, dtype=torch.long)
                E_base = int(self._eff_open_y_knots.shape[0])
                if E_base > 0:
                    edge_ids_full = torch.remainder(edge_ids_full, E_base)
                edge_ids = edge_ids_full[mask_e]
                open_phys = self._apply_effective_opening(open_phys_raw, edge_ids)

            r_energy_phys, h_f, h_m, h_exit_phys, dH = self.energy_residual(
                ei_phys, ea_phys, q_phys, H_node,
                edge_opening=open_phys,
                node_type=node_type,
            )

            r_energy_full = torch.zeros(ei.size(1), device=device)
            r_energy_full[mask_e] = r_energy_phys
            r_energy = r_energy_full
            delta_H = dH

            h_loss_pos = h_f + h_m + h_exit_phys
            h_loss_signed = h_loss_pos * (q_phys / (torch.abs(q_phys) + 1e-9))

        is_source = (node_type == 1) if node_type is not None else False
        is_sink = (node_type == 2) if node_type is not None else False

        total_system_in = sum_out[is_source].sum() if isinstance(is_source, torch.Tensor) and is_source.any() else torch.tensor(0.0, device=device)
        total_system_out = sum_in[is_sink].sum() if isinstance(is_sink, torch.Tensor) and is_sink.any() else torch.tensor(0.0, device=device)

        r_global_mass = total_system_in - total_system_out

        return {
            "h": h,
            "node_hat": node_hat_z,
            "node_hat_biased": node_hat_z_biased,
            "edge_hat": edge_hat_z,
            "H_node": H_node,
            "p_head": p_head,
            "r_mass": r_mass,
            "r_energy": r_energy,
            "valid_mass_mask": valid_mass_mask,
            "edge_hat_m3ps_raw": edge_hat_m3ps_raw,
            "edge_hat_m3ps": edge_hat_m3ps,
            "edge_hat_m3ph": edge_hat_m3ph,
            "demand": demand,
            "net_flow": net_flow,
            "sum_in": sum_in,
            "sum_out": sum_out,
            "h_loss_pos": h_loss_pos,
            "h_loss_signed": h_loss_signed,
            "delta_H": delta_H,
            "h_exit": h_exit,
            "meter_scale": meter_scale,
            "leakage_rate": cast(torch.Tensor, self.get_leakage_rate()),  # type: ignore
            "bias_mu": cast(torch.Tensor, self.get_bias_params()[0]),  # type: ignore
            "bias_delta": cast(torch.Tensor, self.get_bias_params()[1]),  # type: ignore
            "bias_delta_gauge_mean": self.get_bias_delta_gauge_mean(),
            "bias_gauge_weight_sum": cast(torch.Tensor, self.bias_gauge_weight.sum(dim=0)) if isinstance(getattr(self, 'bias_gauge_weight', None), torch.Tensor) else torch.zeros((self.bias_num_bins,), device=device),  # type: ignore
            "node_opening": node_opening,
            "bias_bin_idx": bias_bin_idx if bias_bin_idx is not None else torch.zeros(N, device=device, dtype=torch.long),
            "bias_apply_mask": bias_apply_mask if bias_apply_mask is not None else torch.zeros(N, device=device, dtype=torch.bool),
            "r_global_mass": r_global_mass,
            "total_system_in": total_system_in,
            "total_system_out": total_system_out,
            "C_main": params["C_main"],
            "C_branch": params["C_branch"],
            "alpha": params["alpha"],
            "latent_demand": latent_demand,
            "total_demand": total_demand,
            "sink_open_mask": sink_open_mask if sink_open_mask is not None else torch.ones(N, device=device, dtype=torch.bool)
        }
