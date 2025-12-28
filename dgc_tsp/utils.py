"""
Utility functions for DGC-TSP.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist


def compute_tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    """
    Compute the length of a TSP tour.

    Args:
        coords: Node coordinates (n, 2)
        tour: Tour as node indices (n,)

    Returns:
        Tour length
    """
    n = len(tour)
    length = 0.0
    for i in range(n):
        j = (i + 1) % n
        length += np.linalg.norm(coords[tour[i]] - coords[tour[j]])
    return length


def tour_to_adjacency(tour: np.ndarray, n: int) -> np.ndarray:
    """
    Convert tour sequence to adjacency matrix.

    Args:
        tour: Tour as node indices (n,)
        n: Number of nodes

    Returns:
        Adjacency matrix (n, n)
    """
    adj = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        adj[tour[i], tour[j]] = 1
        adj[tour[j], tour[i]] = 1
    return adj


def adjacency_to_tour(adj: np.ndarray) -> np.ndarray:
    """
    Extract tour from adjacency matrix.

    Args:
        adj: Adjacency matrix (n, n)

    Returns:
        Tour as node indices (n,)
    """
    n = adj.shape[0]

    # Find starting node (any node with degree >= 1)
    degrees = adj.sum(axis=1)
    start = np.argmax(degrees > 0)

    tour = [start]
    visited = {start}
    current = start

    while len(tour) < n:
        neighbors = np.where(adj[current] > 0.5)[0]
        next_node = None

        for neighbor in neighbors:
            if neighbor not in visited:
                next_node = neighbor
                break

        if next_node is None:
            # Find any unvisited node
            unvisited = [i for i in range(n) if i not in visited]
            if unvisited:
                next_node = unvisited[0]
            else:
                break

        tour.append(next_node)
        visited.add(next_node)
        current = next_node

    return np.array(tour)


def nearest_neighbor_tour(coords: np.ndarray, start: int = 0) -> np.ndarray:
    """
    Compute nearest neighbor tour.

    Args:
        coords: Node coordinates (n, 2)
        start: Starting node

    Returns:
        Tour as node indices (n,)
    """
    n = len(coords)
    visited = {start}
    tour = [start]
    current = start

    while len(tour) < n:
        distances = cdist([coords[current]], coords)[0]
        distances[list(visited)] = np.inf
        next_node = np.argmin(distances)
        tour.append(next_node)
        visited.add(next_node)
        current = next_node

    return np.array(tour)


def two_opt(coords: np.ndarray, tour: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    """
    2-opt local search for TSP.

    Args:
        coords: Node coordinates (n, 2)
        tour: Initial tour (n,)
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    n = len(tour)
    tour = tour.copy()
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue

                # Compute change in tour length
                i1, i2 = tour[i], tour[i + 1]
                j1, j2 = tour[j], tour[(j + 1) % n]

                d_before = (
                    np.linalg.norm(coords[i1] - coords[i2]) +
                    np.linalg.norm(coords[j1] - coords[j2])
                )
                d_after = (
                    np.linalg.norm(coords[i1] - coords[j1]) +
                    np.linalg.norm(coords[i2] - coords[j2])
                )

                if d_after < d_before - 1e-10:
                    # Reverse segment
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True

    return tour


def compute_gap(pred_length: float, gt_length: float) -> float:
    """Compute optimality gap in percentage."""
    return (pred_length - gt_length) / gt_length * 100


def generate_tsp_instance(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random TSP instance.

    Args:
        n: Number of nodes
        seed: Random seed

    Returns:
        Node coordinates (n, 2) in [0, 1]^2
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, 2).astype(np.float32)


def solve_tsp_concorde(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve TSP using Concorde (if available).

    Args:
        coords: Node coordinates (n, 2)

    Returns:
        Optimal tour and length

    Note:
        Falls back to nearest neighbor + 2-opt if Concorde not available.
    """
    try:
        from concorde.tsp import TSPSolver
        solver = TSPSolver.from_data(
            coords[:, 0] * 10000,
            coords[:, 1] * 10000,
            norm="EUC_2D"
        )
        solution = solver.solve()
        tour = np.array(solution.tour)
        length = compute_tour_length(coords, tour)
        return tour, length
    except ImportError:
        # Fallback to heuristic
        tour = nearest_neighbor_tour(coords)
        tour = two_opt(coords, tour)
        length = compute_tour_length(coords, tour)
        return tour, length


def solve_tsp_lkh(coords: np.ndarray, runs: int = 1) -> Tuple[np.ndarray, float]:
    """
    Solve TSP using LKH-3 (if available).

    Args:
        coords: Node coordinates (n, 2)
        runs: Number of LKH runs

    Returns:
        Tour and length

    Note:
        Falls back to 2-opt if LKH not available.
    """
    try:
        import lkh
        problem = lkh.LKHProblem.from_coordinates(coords * 10000)
        solution = lkh.solve(problem, runs=runs)
        tour = np.array(solution[0]) - 1  # LKH uses 1-indexed
        length = compute_tour_length(coords, tour)
        return tour, length
    except (ImportError, Exception):
        # Fallback
        tour = nearest_neighbor_tour(coords)
        tour = two_opt(coords, tour)
        length = compute_tour_length(coords, tour)
        return tour, length


class TSPDataset(torch.utils.data.Dataset):
    """
    Dataset for TSP instances.

    Supports two formats:
    1. EDISCO format: "x1 y1 x2 y2 ... output idx1 idx2 ..." (1-indexed tours)
    2. Simple format: "x1 y1 x2 y2 ... idx1 idx2 ..." (0-indexed tours)

    Auto-detects format based on presence of " output " delimiter.
    """

    def __init__(
        self,
        data_file: str,
        compute_adjacency: bool = True,
        format: str = 'auto',
    ):
        """
        Args:
            data_file: Path to data file
            compute_adjacency: Whether to compute adjacency matrices
            format: 'auto', 'edisco', or 'simple'
        """
        self.compute_adjacency = compute_adjacency
        self.instances = []

        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Auto-detect format based on first line
                if format == 'auto':
                    format = 'edisco' if ' output ' in line else 'simple'

                if format == 'edisco' or ' output ' in line:
                    # EDISCO format: "x1 y1 x2 y2 ... output idx1 idx2 ..."
                    coords_str, tour_str = line.split(' output ')
                    coords_parts = coords_str.strip().split()
                    tour_parts = tour_str.strip().split()

                    n = len(coords_parts) // 2
                    coords = np.array([float(coords_parts[i]) for i in range(2 * n)]).reshape(n, 2)
                    tour = np.array([int(t) - 1 for t in tour_parts])  # Convert to 0-indexed
                else:
                    # Simple format: "x1 y1 x2 y2 ... idx1 idx2 ..."
                    parts = line.split()
                    n = len(parts) // 3  # n coords (x,y) + n tour indices
                    coords = np.array([float(parts[i]) for i in range(2 * n)]).reshape(n, 2)
                    tour = np.array([int(parts[2 * n + i]) for i in range(n)])

                self.instances.append({
                    'coords': coords.astype(np.float32),
                    'tour': tour,
                })

        print(f'Loaded "{data_file}" with {len(self.instances)} instances (format: {format})')

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        instance = self.instances[idx]
        coords = torch.from_numpy(instance['coords'])
        tour = torch.from_numpy(instance['tour'])

        item = {
            'coords': coords,
            'tour': tour,
        }

        if self.compute_adjacency:
            n = len(coords)
            adj = tour_to_adjacency(instance['tour'], n)
            item['adj'] = torch.from_numpy(adj).float()

        return item


def collate_tsp_batch(batch: List[dict]) -> dict:
    """
    Collate function for TSP batches.

    Handles variable-size graphs by padding.
    """
    # Find max size
    max_n = max(item['coords'].shape[0] for item in batch)

    coords_batch = []
    tour_batch = []
    adj_batch = []
    mask_batch = []

    for item in batch:
        n = item['coords'].shape[0]

        # Pad coordinates
        coords = F.pad(item['coords'], (0, 0, 0, max_n - n))
        coords_batch.append(coords)

        # Pad tour
        tour = F.pad(item['tour'], (0, max_n - n), value=-1)
        tour_batch.append(tour)

        # Pad adjacency
        if 'adj' in item:
            adj = F.pad(item['adj'], (0, max_n - n, 0, max_n - n))
            adj_batch.append(adj)

        # Create mask
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n] = True
        mask_batch.append(mask)

    result = {
        'coords': torch.stack(coords_batch),
        'tour': torch.stack(tour_batch),
        'mask': torch.stack(mask_batch),
    }

    if adj_batch:
        result['adj'] = torch.stack(adj_batch)

    return result
