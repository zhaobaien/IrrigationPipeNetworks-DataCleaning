from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class GAT(nn.Module):
    """Data-driven graph attention network without hydraulic constraints."""
    
    def __init__(
        self, 
        win: int, 
        num_nodes: int,
        num_edges: int,
        hidden: int = 64, 
        heads: int = 8, 
        dropout: float = 0.1, 
        in_feat: int = 2
    ):
        super().__init__()
        in_dim = win * in_feat

        # Graph encoder.
        self.lin_in = nn.Linear(in_dim, hidden, bias=True)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=False)
        self.bn1 = BatchNorm(hidden)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=False)
        self.bn2 = BatchNorm(hidden)

        # Decoder for standardised node flows.
        self.node_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # Decoder for standardised edge flows.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # Return only the standardised flow.
        )

        # Predict node pressure with an unconstrained multilayer perceptron.
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        self.dropout = dropout
        self.hidden = hidden

    def encode(self, x, edge_index):
        """Encode the graph observations."""
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
        """Predict standardised node flow."""
        return self.node_head(h).squeeze(-1)

    def decode_edge(self, h, edge_index, edge_attr):
        """Predict standardised edge flow."""
        src, dst = edge_index
        hs = h[src]
        hd = h[dst]
        w = edge_attr[:, 3:4]
        ed_in = torch.cat([hs, hd, w], dim=-1)
        
        out = self.edge_mlp(ed_in) # shape: [E, 1]
        return out.squeeze(-1)

    def decode_pressure(self, h, node_elev):
        """Predict pressure head with a multilayer perceptron without elevation scaling."""
        p_raw = self.pressure_head(h).squeeze(-1)
        # Use the multilayer perceptron output directly without elevation scaling or hydraulic constraints.
        H_total = node_elev + p_raw
        return H_total, p_raw

    def forward(self, data):
        """Run the forward pass."""
        x = data.x
        ei = data.edge_index
        ea = data.edge_attr
        N = x.size(0)
        device = x.device
        
        # Encode graph observations.
        h = self.encode(x, ei)
        
        # Decode flow and pressure.
        node_hat_z = self.decode_node(h)
        edge_hat_z = self.decode_edge(h, ei, ea)
        H_node, p_head = self.decode_pressure(h, data.node_elev)
        
        return {
            "h": h,
            "node_hat": node_hat_z,          # Standardised node flow predictions.
            "edge_hat": edge_hat_z,          # Standardised edge flow predictions.
            "H_node": H_node,
            "p_head": p_head,
        } 
