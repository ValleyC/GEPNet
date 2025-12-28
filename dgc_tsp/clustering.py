"""
E(2)-Equivariant Graph Clustering for TSP

This module learns to partition TSP graphs using EGNN-based clustering.
The clustering is E(2)-invariant (rotation/translation invariant).

Key features:
1. EGNN encoder produces invariant node embeddings
2. Soft cluster assignment via attention to learnable centroids
3. Tour-aware self-supervised loss to minimize inter-cluster tour edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np

from .encoder import EGNNEncoder


class EGNNClusteringNetwork(nn.Module):
    """
    E(2)-Equivariant Clustering Network for TSP.

    Uses EGNN to produce invariant node embeddings, then assigns
    nodes to clusters in a way that minimizes tour edge cuts.
    """

    def __init__(
        self,
        num_clusters: int = 10,
        hidden_dim: int = 128,
        num_egnn_layers: int = 4,
        temperature: float = 0.5,
        lambda_balance: float = 0.1,
        lambda_tour: float = 1.0,
        lambda_contrastive: float = 0.1,
    ):
        """
        Args:
            num_clusters: Number of clusters (k)
            hidden_dim: Hidden dimension for EGNN and MLP
            num_egnn_layers: Number of EGNN layers
            temperature: Temperature for cluster assignment softmax
            lambda_balance: Weight for balance loss
            lambda_tour: Weight for tour alignment loss
            lambda_contrastive: Weight for contrastive loss
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.lambda_balance = lambda_balance
        self.lambda_tour = lambda_tour
        self.lambda_contrastive = lambda_contrastive

        # EGNN encoder: produces E(2)-invariant node embeddings
        self.encoder = EGNNEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_egnn_layers,
        )

        # Learnable cluster centroids in embedding space
        self.cluster_centroids = nn.Parameter(
            torch.randn(num_clusters, hidden_dim) * 0.1
        )

        # MLP for cluster assignment
        self.assignment_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates to E(2)-invariant node embeddings.

        Args:
            coords: Node coordinates (batch, n, 2)

        Returns:
            Node embeddings (batch, n, hidden_dim)
        """
        return self.encoder(coords)

    def get_assignments(
        self,
        h: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Get cluster assignments from node embeddings.

        Args:
            h: Node embeddings (batch, n, hidden_dim)
            hard: If True, return one-hot assignments

        Returns:
            Soft or hard cluster assignments (batch, n, k)
        """
        batch_size, n, d = h.shape

        # Transform embeddings
        h_transformed = self.assignment_mlp(h)  # (batch, n, d)

        # Compute similarity to cluster centroids
        # centroids: (k, d) -> (1, k, d) -> (batch, k, d)
        centroids = self.cluster_centroids.unsqueeze(0).expand(batch_size, -1, -1)

        # Dot product similarity: (batch, n, d) @ (batch, d, k) -> (batch, n, k)
        scores = torch.bmm(h_transformed, centroids.transpose(1, 2))
        scores = scores / (d ** 0.5)  # Scale by sqrt(d)
        scores = scores / self.temperature

        # Soft assignment via softmax
        assignments = F.softmax(scores, dim=-1)

        if hard:
            # Straight-through estimator
            hard_assignments = F.one_hot(assignments.argmax(dim=-1), self.num_clusters).float()
            assignments = hard_assignments - assignments.detach() + assignments

        return assignments

    def forward(
        self,
        coords: torch.Tensor,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            coords: Node coordinates (batch, n, 2)
            hard: Whether to use hard cluster assignments

        Returns:
            Dictionary with embeddings, assignments, projections
        """
        # Encode to invariant embeddings
        h = self.encode(coords)

        # Get cluster assignments
        assignments = self.get_assignments(h, hard=hard)

        # Project for contrastive learning
        projections = self.projection_head(h)
        projections = F.normalize(projections, p=2, dim=-1)

        return {
            'embeddings': h,
            'assignments': assignments,
            'projections': projections,
        }

    def tour_alignment_loss(
        self,
        assignments: torch.Tensor,
        tour: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute tour alignment loss (vectorized, no loops).

        Encourages adjacent nodes in tour to be in the same cluster.

        Args:
            assignments: Soft cluster assignments (batch, n, k)
            tour: Tour indices (batch, n)

        Returns:
            Loss scalar
        """
        batch_size, n, k = assignments.shape
        device = assignments.device

        # Get assignments for each position in tour
        # tour[b, i] gives the node index at position i
        # We want to compare assignment of tour[b, i] with tour[b, i+1]

        # Gather assignments according to tour order
        # assignments: (batch, n, k), tour: (batch, n)
        tour_expanded = tour.unsqueeze(-1).expand(-1, -1, k)  # (batch, n, k)
        ordered_assignments = torch.gather(assignments, 1, tour_expanded)  # (batch, n, k)

        # Compare adjacent positions in tour
        # Current position assignments
        curr_assign = ordered_assignments[:, :-1, :]  # (batch, n-1, k)
        # Next position assignments
        next_assign = ordered_assignments[:, 1:, :]  # (batch, n-1, k)

        # Also compare last with first (tour is cyclic)
        last_assign = ordered_assignments[:, -1:, :]  # (batch, 1, k)
        first_assign = ordered_assignments[:, :1, :]  # (batch, 1, k)

        # Concatenate
        curr_assign = torch.cat([curr_assign, last_assign], dim=1)  # (batch, n, k)
        next_assign = torch.cat([next_assign, first_assign], dim=1)  # (batch, n, k)

        # Compute probability that adjacent nodes are in same cluster
        # sum_k(p_i[k] * p_j[k]) = probability of same cluster
        same_cluster_prob = (curr_assign * next_assign).sum(dim=-1)  # (batch, n)

        # Loss: maximize same_cluster_prob -> minimize (1 - same_cluster_prob)
        loss = (1 - same_cluster_prob).mean()

        return loss

    def balance_loss(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Compute balance loss for equal-sized clusters.

        Args:
            assignments: Soft cluster assignments (batch, n, k)

        Returns:
            Loss scalar
        """
        # Average assignment per cluster: (batch, k)
        cluster_sizes = assignments.mean(dim=1)

        # Target: uniform distribution
        target = torch.ones_like(cluster_sizes) / self.num_clusters

        # KL divergence from uniform
        loss = F.kl_div(
            torch.log(cluster_sizes + 1e-8),
            target,
            reduction='batchmean'
        )
        return loss

    def contrastive_loss(
        self,
        projections: torch.Tensor,
        assignments: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Contrastive loss: nodes in same cluster should have similar embeddings.

        Args:
            projections: L2-normalized projections (batch, n, d)
            assignments: Soft cluster assignments (batch, n, k)
            temperature: Temperature for contrastive loss

        Returns:
            Loss scalar
        """
        batch_size, n, d = projections.shape

        # Compute similarity matrix
        sim_matrix = torch.bmm(projections, projections.transpose(1, 2))  # (batch, n, n)
        sim_matrix = sim_matrix / temperature

        # Compute cluster membership similarity
        cluster_sim = torch.bmm(assignments, assignments.transpose(1, 2))  # (batch, n, n)

        # Mask diagonal (don't compare node with itself)
        mask = torch.eye(n, device=projections.device).unsqueeze(0)
        sim_matrix = sim_matrix - mask * 1e9

        # InfoNCE-style loss
        exp_sim = torch.exp(sim_matrix)
        positive_sim = (exp_sim * cluster_sim).sum(dim=-1)
        total_sim = exp_sim.sum(dim=-1)

        loss = -torch.log(positive_sim / (total_sim + 1e-8) + 1e-8)
        return loss.mean()

    def compute_loss(
        self,
        coords: torch.Tensor,
        tour: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            coords: Node coordinates (batch, n, 2)
            tour: Optional tour indices (batch, n) for tour alignment loss

        Returns:
            Dictionary of losses
        """
        output = self.forward(coords)
        assignments = output['assignments']
        projections = output['projections']

        losses = {}

        # Balance loss (always computed)
        losses['balance'] = self.balance_loss(assignments)

        # Contrastive loss
        losses['contrastive'] = self.contrastive_loss(projections, assignments)

        # Tour alignment loss (if tour provided)
        if tour is not None:
            losses['tour'] = self.tour_alignment_loss(assignments, tour)
        else:
            losses['tour'] = torch.tensor(0.0, device=coords.device)

        # Total loss
        losses['total'] = (
            self.lambda_balance * losses['balance'] +
            self.lambda_contrastive * losses['contrastive'] +
            self.lambda_tour * losses['tour']
        )

        return losses, output

    def predict_clusters(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Predict hard cluster assignments.

        Args:
            coords: Node coordinates (batch, n, 2) or (n, 2)

        Returns:
            Cluster assignments (batch, n) or (n,)
        """
        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        with torch.no_grad():
            output = self.forward(coords, hard=True)
            assignments = output['assignments'].argmax(dim=-1)

        if single:
            assignments = assignments.squeeze(0)

        return assignments

    def get_cluster_nodes(self, assignments: torch.Tensor) -> List[List[int]]:
        """
        Get list of node indices for each cluster.

        Args:
            assignments: Hard assignments (n,) for single graph

        Returns:
            List of lists containing node indices for each cluster
        """
        clusters = []
        for k in range(self.num_clusters):
            nodes = (assignments == k).nonzero(as_tuple=True)[0].tolist()
            clusters.append(nodes)
        return clusters

    def compute_tour_edge_cuts(
        self,
        assignments: torch.Tensor,
        tour: torch.Tensor,
    ) -> int:
        """
        Count number of tour edges that cross cluster boundaries.

        Args:
            assignments: Hard assignments (n,)
            tour: Tour indices (n,)

        Returns:
            Number of inter-cluster edges
        """
        n = len(tour)
        cuts = 0
        for i in range(n):
            j = (i + 1) % n
            node_i = tour[i].item()
            node_j = tour[j].item()
            if assignments[node_i] != assignments[node_j]:
                cuts += 1
        return cuts


class KMeansBaseline(nn.Module):
    """K-Means clustering baseline for comparison."""

    def __init__(self, num_clusters: int = 10, num_iters: int = 20):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iters = num_iters

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Perform K-Means clustering on coordinates.

        Args:
            coords: Node coordinates (batch, n, 2) or (n, 2)

        Returns:
            Cluster assignments
        """
        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        batch_size, n, d = coords.shape
        device = coords.device
        assignments = torch.zeros(batch_size, n, dtype=torch.long, device=device)

        for b in range(batch_size):
            points = coords[b]

            # Initialize centroids randomly
            indices = torch.randperm(n, device=device)[:self.num_clusters]
            centroids = points[indices].clone()

            for _ in range(self.num_iters):
                # Assign points to nearest centroid
                dists = torch.cdist(points, centroids)
                cluster_ids = dists.argmin(dim=1)

                # Update centroids
                for k in range(self.num_clusters):
                    mask = (cluster_ids == k)
                    if mask.sum() > 0:
                        centroids[k] = points[mask].mean(dim=0)

                assignments[b] = cluster_ids

        if single:
            assignments = assignments.squeeze(0)

        return assignments
