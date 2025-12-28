"""
E(n)-Equivariant Graph Neural Network Encoder for TSP

This encoder produces node embeddings that are invariant to rotation and translation,
making it suitable for geometric problems like TSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant graph neural network layer."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 1,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = True,
        normalize: bool = False,
        coords_agg: str = 'mean',
        tanh: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh

        # Edge MLP: transforms edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate MLP: updates coordinates (optional, usually not used for encoder)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Attention weights
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: Node features (n, hidden_dim)
            x: Node coordinates (n, coord_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)

        Returns:
            Updated node features and coordinates
        """
        row, col = edge_index

        # Compute relative positions and distances
        coord_diff = x[row] - x[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)

        # Prepare edge input
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1], 1, device=h.device)

        edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=1)

        # Edge message
        m_ij = self.edge_mlp(edge_input)

        # Attention weights
        if self.attention:
            att = self.att_mlp(m_ij)
            m_ij = m_ij * att

        # Aggregate messages
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, m_ij)

        # Update node features
        h_input = torch.cat([h, agg], dim=1)
        h_out = self.node_mlp(h_input)

        if self.residual:
            h_out = h + h_out

        # Coordinate update (optional, usually identity for encoder)
        coord_update = self.coord_mlp(m_ij)
        if self.tanh:
            coord_update = torch.tanh(coord_update)

        # Aggregate coordinate updates
        if self.normalize:
            coord_diff = coord_diff / (torch.sqrt(radial) + 1e-8)

        coord_agg = torch.zeros_like(x)
        coord_agg.index_add_(0, row, coord_diff * coord_update)

        if self.coords_agg == 'mean':
            # Count neighbors for each node
            neighbor_count = torch.zeros(x.shape[0], device=x.device)
            neighbor_count.index_add_(0, row, torch.ones(row.shape[0], device=x.device))
            neighbor_count = neighbor_count.clamp(min=1).unsqueeze(1)
            coord_agg = coord_agg / neighbor_count

        x_out = x + coord_agg

        return h_out, x_out


class EGNNEncoder(nn.Module):
    """
    E(n)-Equivariant encoder for TSP graphs.

    Produces E(n)-invariant node embeddings from node coordinates.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,  # Matched to competitors (GLOP=48, UDC=64)
        output_dim: int = 64,
        num_layers: int = 8,  # Fewer than GLOP/UDC (12) due to heavier EGNN layers
        attention: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                attention=attention,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def build_full_graph(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Build fully connected graph edge index."""
        # Create all pairs (excluding self-loops)
        row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes - 1)
        col = torch.cat([
            torch.cat([torch.arange(i, device=device), torch.arange(i + 1, num_nodes, device=device)])
            for i in range(num_nodes)
        ])
        return torch.stack([row, col], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node coordinates (batch_size, n, 2) or (n, 2)
            edge_index: Edge indices (optional, builds full graph if None)
            batch: Batch indices for batched graphs

        Returns:
            Node embeddings (batch_size, n, output_dim) or (n, output_dim)
        """
        # Handle batched input
        if x.dim() == 3:
            batch_size, num_nodes, coord_dim = x.shape
            # Flatten to (batch_size * n, coord_dim)
            x_flat = x.reshape(-1, coord_dim)

            # Build batch of edge indices
            if edge_index is None:
                edge_indices = []
                for b in range(batch_size):
                    offset = b * num_nodes
                    ei = self.build_full_graph(num_nodes, x.device) + offset
                    edge_indices.append(ei)
                edge_index = torch.cat(edge_indices, dim=1)

            # Process flattened graph
            h = self.node_embedding(x_flat)

            for layer in self.layers:
                h, x_flat = layer(h, x_flat, edge_index)

            h = self.output_proj(h)

            # Reshape back to batched form
            h = h.reshape(batch_size, num_nodes, -1)

        else:
            # Single graph
            if edge_index is None:
                edge_index = self.build_full_graph(x.shape[0], x.device)

            h = self.node_embedding(x)

            for layer in self.layers:
                h, x = layer(h, x, edge_index)

            h = self.output_proj(h)

        return h


class DistanceEncoder(nn.Module):
    """
    Simple distance-based encoder (non-equivariant baseline).

    Uses pairwise distances as input features.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = 1 if i == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node coordinates (batch_size, n, 2)

        Returns:
            Node embeddings (batch_size, n, output_dim)
        """
        # Compute pairwise distances
        dist = torch.cdist(x, x, p=2)  # (batch, n, n)

        # Use distance matrix as features
        h = dist.unsqueeze(-1)  # (batch, n, n, 1)

        for layer in self.layers:
            h = layer(h)

        # Aggregate over neighbors
        h = h.mean(dim=2)  # (batch, n, hidden)

        h = self.output_proj(h)

        return h
