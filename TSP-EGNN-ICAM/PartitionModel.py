"""
GEPNet Partition Model for TSP.

E(2)-Equivariant partition network that replaces UDC's AGNN.

Key differences from CVRP:
- No depot node (all nodes are equal)
- No capacity constraints
- State query: first + last + visited (no depot)
- Node features: Constant ones (E(2)-invariant)
  NOTE: Raw (x, y) coordinates are NOT used as features - they break equivariance!
  Coordinates are passed separately for EGNN distance computation.
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class EGNNLayer(nn.Module):
    """
    E(2)-Equivariant GNN Layer.

    Same as CVRP version - shared architecture.
    """

    def __init__(self, hidden_dim, act_fn='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_fn = getattr(F, act_fn)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, coords, e, edge_index):
        row, col = edge_index[0], edge_index[1]
        n_nodes = h.shape[0]

        coord_diff = coords[col] - coords[row]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        msg_input = torch.cat([h[row], h[col], dist, e], dim=-1)
        msg = self.msg_mlp(msg_input)

        coord_weights = self.coord_mlp(msg)
        coord_weights = torch.tanh(coord_weights)
        direction = coord_diff / (dist + 1e-8)
        coord_update = coord_weights * direction

        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_update)
        coords_new = coords + 0.1 * coord_agg

        msg_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        msg_agg.index_add_(0, row, msg)

        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, msg_agg], dim=-1)))
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, msg], dim=-1)))

        return h_new, coords_new, e_new


class EGNN(nn.Module):
    """
    E(2)-Equivariant GNN backbone for TSP partition.

    Node features: Constant ones (E(2)-invariant)
    Edge features: Negative distances (E(2)-invariant)
    Coordinates: Used for EGNN message passing (distances are invariant)

    NOTE: Raw (x, y) are NOT used as node features - they break equivariance!
    """

    def __init__(self, depth=12, node_feats=1, edge_feats=2, units=48):
        super().__init__()
        self.depth = depth
        self.units = units

        self.node_embed = nn.Linear(node_feats, units)
        self.edge_embed = nn.Linear(edge_feats, units)

        self.layers = nn.ModuleList([
            EGNNLayer(units) for _ in range(depth)
        ])

    def forward(self, x, edge_index, edge_attr, coords):
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        pos = coords.clone()

        for layer in self.layers:
            h, pos, e = layer(h, pos, e, edge_index)

        return h, e


class EmbNet(nn.Module):
    """UDC's original GNN for edge embeddings (non-equivariant)."""

    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def forward(self, x, edge_index, edge_attr):
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(edge_attr)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


class ParNet(nn.Module):
    """
    State-aware partition head for TSP.

    Query = first + last + avg(visited)
    No depot for TSP (unlike CVRP).
    """

    def __init__(self, k_sparse, depth=3, units=48, act_fn='silu'):
        super().__init__()
        self.units = units
        self.k_sparse = k_sparse
        self.depth = depth
        self.act_fn = getattr(F, act_fn)

        self.lins = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.lin_f = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # first
        self.lin_l = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # last
        self.lin_v = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # visited

    def forward(self, node_emb, edge_emb, solution, visited):
        """
        Args:
            node_emb: Node embeddings (n_nodes, units)
            edge_emb: Edge embeddings (n_edges, units)
            solution: [first_node, last_node] per sample (sample_size, 2)
            visited: Binary mask (sample_size, n_nodes)

        Returns:
            probs: Selection probabilities (sample_size, n_nodes, k_sparse)
        """
        sample_size = solution.size(0)

        # First node embedding
        first_idx = solution[:, 0]
        x_first = node_emb[first_idx].unsqueeze(1)

        # Last node embedding
        last_idx = solution[:, 1]
        x_last = node_emb[last_idx].unsqueeze(1)

        # Visited nodes average embedding
        x_visited = node_emb.clone()
        x = edge_emb.clone()

        for i in range(self.depth):
            x = self.lins[i](x)
            x_first = self.lin_f[i](x_first)
            x_last = self.lin_l[i](x_last)
            x_visited = self.lin_v[i](x_visited)

            if i < self.depth - 1:
                x = self.act_fn(x)
                x_first = self.act_fn(x_first)
                x_last = self.act_fn(x_last)
                x_visited = self.act_fn(x_visited)
            else:
                # Query = first + last + avg(visited)
                visited_mask = visited[:, :, None].float()
                x_visited_expanded = x_visited[None, :, :].expand(sample_size, -1, -1)
                q_visited = (x_visited_expanded * visited_mask).sum(1) / (visited_mask.sum(1) + 1e-8)

                q = q_visited + x_first.squeeze(1) + x_last.squeeze(1)

                scores = torch.mm(q, x.T)
                scores = scores.reshape(sample_size, -1, self.k_sparse)
                probs = torch.softmax(scores, dim=-1)

        return probs


class PartitionModel(nn.Module):
    """
    GEPNet Partition Model for TSP.

    E(2)-equivariant partition with state-aware autoregressive selection.
    """

    def __init__(self, units, feats, k_sparse, edge_feats=2, depth=12, use_egnn=True):
        super().__init__()
        self.use_egnn = use_egnn
        self.k_sparse = k_sparse

        if use_egnn:
            self.emb_net = EGNN(depth=depth, node_feats=feats, edge_feats=edge_feats, units=units)
        else:
            self.emb_net = EmbNet(depth=depth, feats=feats, edge_feats=edge_feats, units=units)

        self.par_net = ParNet(k_sparse=k_sparse, units=units)
        self.x_emb = None
        self.emb = None
        self.coords = None

    def pre(self, pyg):
        """Pre-compute embeddings."""
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr

        if self.use_egnn:
            coords = pyg.pos if hasattr(pyg, 'pos') else pyg.x
            self.coords = coords
            self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr, coords)
        else:
            self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr)

    def forward(self, solution=None, visited=None):
        """
        Compute partition heuristics.

        For TSP: solution contains first and last nodes of current path.
        """
        # solution_cat: [first_node, last_node]
        solution_cat = torch.cat([
            solution[:, 0:1],
            solution[:, -1:]
        ], dim=-1)

        heu = self.par_net(self.x_emb, self.emb, solution_cat, visited)
        return self.x_emb, heu

    @staticmethod
    def reshape(pyg, vector):
        """Turn heuristic vector into adjacency matrix."""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(vector.size(0), n_nodes, n_nodes), device=device)
        idx = torch.repeat_interleave(torch.arange(vector.size(0), device=device), repeats=pyg.edge_index[0].shape[0])
        idx0 = pyg.edge_index[0].repeat(vector.size(0))
        idx1 = pyg.edge_index[1].repeat(vector.size(0))
        matrix[idx, idx0, idx1] = vector.view(-1)
        return matrix
