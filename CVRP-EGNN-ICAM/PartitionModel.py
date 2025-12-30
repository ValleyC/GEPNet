"""
GEPNet Partition Model for CVRP.

E(2)-Equivariant partition network that replaces UDC's AGNN.

Key innovations:
1. EGNN backbone produces invariant embeddings
2. No coordinate_transformation needed
3. State-aware partition considers depot, first, last, visited nodes
4. Node features: (demand, r) - E(2)-invariant features only
   NOTE: theta (polar angle) is NOT used - it breaks equivariance!
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class EGNNLayer(nn.Module):
    """
    E(2)-Equivariant GNN Layer.

    Message passing that respects geometric symmetry:
    - Messages computed from invariant quantities (distances)
    - Coordinate updates are equivariant
    """

    def __init__(self, hidden_dim, act_fn='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_fn = getattr(F, act_fn)

        # Message MLP: [h_i, h_j, ||x_i-x_j||, e_ij] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate weight MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, coords, e, edge_index):
        """
        Args:
            h: Node features (n_nodes, hidden_dim)
            coords: Node coordinates (n_nodes, 2)
            e: Edge features (n_edges, hidden_dim)
            edge_index: (2, n_edges)

        Returns:
            h_new, coords_new, e_new
        """
        row, col = edge_index[0], edge_index[1]
        n_nodes = h.shape[0]

        # Compute distances (E(2)-invariant)
        coord_diff = coords[col] - coords[row]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        # Messages
        msg_input = torch.cat([h[row], h[col], dist, e], dim=-1)
        msg = self.msg_mlp(msg_input)

        # Coordinate updates (equivariant)
        coord_weights = self.coord_mlp(msg)
        coord_weights = torch.tanh(coord_weights)
        direction = coord_diff / (dist + 1e-8)
        coord_update = coord_weights * direction

        # Aggregate coordinate updates
        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_update)
        coords_new = coords + 0.1 * coord_agg

        # Aggregate messages for node update
        msg_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        msg_agg.index_add_(0, row, msg)

        # Update nodes
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, msg_agg], dim=-1)))

        # Update edges
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, msg], dim=-1)))

        return h_new, coords_new, e_new


class EGNN(nn.Module):
    """
    E(2)-Equivariant GNN backbone for CVRP partition.

    Input node features: (demand, r) - E(2)-invariant features
    NOTE: theta is NOT used as it breaks equivariance under rotation!
    """

    def __init__(self, depth=12, node_feats=2, edge_feats=2, units=48):
        super().__init__()
        self.depth = depth
        self.units = units

        self.node_embed = nn.Linear(node_feats, units)
        self.edge_embed = nn.Linear(edge_feats, units)

        self.layers = nn.ModuleList([
            EGNNLayer(units) for _ in range(depth)
        ])

    def forward(self, x, edge_index, edge_attr, coords):
        """
        Args:
            x: Node features (n_nodes, node_feats) - (demand, r)
            edge_index: (2, n_edges)
            edge_attr: Edge features (n_edges, edge_feats)
            coords: Node coordinates (n_nodes, 2) for EGNN updates

        Returns:
            node_emb, edge_emb
        """
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        pos = coords.clone()

        for layer in self.layers:
            h, pos, e = layer(h, pos, e, edge_index)

        return h, e


# UDC's original EmbNet (for ablation)
class EmbNet(nn.Module):
    """UDC's original GNN for edge embeddings (non-equivariant)."""

    def __init__(self, depth=12, feats=3, edge_feats=2, units=48, act_fn='silu', agg_fn='mean'):
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
    State-aware partition head for CVRP.

    Query = depot + first + last + avg(visited)
    This captures the current solution state for autoregressive partition.
    """

    def __init__(self, k_sparse, depth=3, units=48, act_fn='silu'):
        super().__init__()
        self.units = units
        self.k_sparse = k_sparse
        self.depth = depth
        self.act_fn = getattr(F, act_fn)

        # Projections for each state component
        self.lins = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])
        self.lin_d = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # depot
        self.lin_f = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # first
        self.lin_l = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # last
        self.lin_v = nn.ModuleList([nn.Linear(units, units) for _ in range(depth)])  # visited

    def forward(self, node_emb, edge_emb, solution, visited):
        """
        Args:
            node_emb: Node embeddings (n_nodes, units)
            edge_emb: Edge embeddings (n_edges, units)
            solution: Current solution [first_node, last_node] per sample (sample_size, 2)
            visited: Binary mask (sample_size, n_nodes)

        Returns:
            probs: Selection probabilities (sample_size, n_nodes, k_sparse)
        """
        sample_size = solution.size(0)

        # Depot embedding (node 0)
        x_depot = node_emb[0:1, :].expand(sample_size, -1).unsqueeze(1)  # (sample_size, 1, units)

        # First node embedding
        first_idx = solution[:, 0]
        x_first = node_emb[first_idx].unsqueeze(1)  # (sample_size, 1, units)

        # Last node embedding
        last_idx = solution[:, 1]
        x_last = node_emb[last_idx].unsqueeze(1)  # (sample_size, 1, units)

        # Visited nodes average embedding
        x_visited = node_emb.clone()  # (n_nodes, units)

        x = edge_emb.clone()

        for i in range(self.depth):
            x = self.lins[i](x)
            x_depot = self.lin_d[i](x_depot)
            x_first = self.lin_f[i](x_first)
            x_last = self.lin_l[i](x_last)
            x_visited = self.lin_v[i](x_visited)

            if i < self.depth - 1:
                x = self.act_fn(x)
                x_depot = self.act_fn(x_depot)
                x_first = self.act_fn(x_first)
                x_last = self.act_fn(x_last)
                x_visited = self.act_fn(x_visited)
            else:
                # Final layer: compute query and attention
                # q = depot + first + last + avg(visited)
                visited_mask = visited[:, :, None].float()  # (sample_size, n_nodes, 1)
                x_visited_expanded = x_visited[None, :, :].expand(sample_size, -1, -1)  # (sample_size, n_nodes, units)
                q_visited = (x_visited_expanded * visited_mask).sum(1) / (visited_mask.sum(1) + 1e-8)  # (sample_size, units)

                q = q_visited + x_first.squeeze(1) + x_last.squeeze(1) + x_depot.squeeze(1)  # (sample_size, units)

                # Attention scores
                scores = torch.mm(q, x.T)  # (sample_size, n_edges)
                scores = scores.reshape(sample_size, -1, self.k_sparse)  # (sample_size, n_nodes, k_sparse)
                probs = torch.softmax(scores, dim=-1)

        return probs


class PartitionModel(nn.Module):
    """
    GEPNet Partition Model for CVRP.

    E(2)-equivariant partition with state-aware autoregressive selection.

    Key differences from UDC:
    1. Uses EGNN backbone (equivariant)
    2. No coordinate_transformation needed
    3. Invariant node features by design
    """

    def __init__(self, units, feats, k_sparse, edge_feats=2, depth=12, use_egnn=True):
        """
        Args:
            units: Hidden dimension
            feats: Node feature dim (2 for CVRP: demand, r) - NO theta!
            k_sparse: Number of sparse neighbors
            edge_feats: Edge feature dim
            depth: Number of GNN layers
            use_egnn: If True, use EGNN; else use UDC's EmbNet
        """
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

        # Store coordinates for EGNN
        self.coords = None

    def pre(self, pyg):
        """
        Pre-compute embeddings for the instance.

        Args:
            pyg: PyG Data with x, edge_index, edge_attr, and optionally pos
        """
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr

        if self.use_egnn:
            # EGNN needs coordinates
            coords = pyg.pos if hasattr(pyg, 'pos') else pyg.x[:, :2]
            self.coords = coords
            self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr, coords)
        else:
            self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr)

    def forward(self, solution=None, selected=None, visited=None):
        """
        Compute partition heuristics given current solution state.

        Args:
            solution: Current partial solution (for getting first node)
            selected: Currently selected node (last node)
            visited: Binary mask of visited nodes

        Returns:
            node_emb, heu_probs
        """
        # Combine solution info: [first_node, last_node]
        solution_cat = torch.cat((solution[:, 0:1], selected), dim=-1)

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
