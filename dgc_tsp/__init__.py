"""
DGC-TSP: E(2)-Equivariant Graph Clustering for TSP

Stage 1: Learn to partition TSP instances using EGNN-based clustering.
The clustering is E(2)-invariant (rotation/translation invariant).
"""

from .encoder import EGNNEncoder
from .clustering import EGNNClusteringNetwork

__all__ = [
    'EGNNEncoder',
    'EGNNClusteringNetwork',
]
