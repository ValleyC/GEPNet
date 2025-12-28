# E²P: E(2)-Equivariant Partition for Large-Scale TSP

This repository implements **E(2)-equivariant graph clustering** for partitioning large-scale TSP instances. The clustering is rotation and translation invariant by design, eliminating the need for data augmentation.

## Motivation

Traditional divide-and-conquer approaches for large-scale TSP (like GLOP, UDC) use:
- Fixed partitioning (sliding window) or
- Standard GNN-based partitioning (requires data augmentation for rotation invariance)

Our approach uses **EGNN-based clustering** that:
1. Is **E(2)-invariant** by design (no augmentation needed)
2. **Minimizes inter-cluster tour edges** via tour-aware loss
3. Creates **balanced clusters** for parallel processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Stage 1: E(2)-Equivariant Clustering              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Node coordinates (n × 2)                            │
│              ↓                                              │
│  ┌─────────────────────────┐                                │
│  │   EGNN Encoder Layers   │  ← E(2)-equivariant            │
│  │   (message passing)     │     message passing            │
│  └───────────┬─────────────┘                                │
│              ↓                                              │
│  Node embeddings h ∈ R^(n × d)  ← E(2)-INVARIANT features   │
│              ↓                                              │
│  ┌─────────────────────────┐                                │
│  │  Cluster Assignment MLP │  → Soft assignments (n × k)    │
│  │  + Learnable Centroids  │                                │
│  └───────────┬─────────────┘                                │
│              ↓                                              │
│  Output: Cluster labels     ← Invariant to rotation         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│           Stage 2: Sub-problem Solving (Future)             │
├─────────────────────────────────────────────────────────────┤
│  • Solve each cluster with EDISCO/LKH                       │
│  • Connect clusters optimally                               │
│  • Optional: 2-opt refinement                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Innovation

### E(2)-Invariant Clustering

Unlike existing methods that use standard GNN + data augmentation:

| Method | Network | Augmentation | Tour-Aware |
|--------|---------|--------------|------------|
| GLOP | Attention | Required (8x) | No |
| UDC | Standard GNN | Required | No |
| **Ours** | **EGNN** | **Not needed** | **Yes** |

### Tour-Aware Loss

```
L = λ_balance * L_balance + λ_contrastive * L_contrastive + λ_tour * L_tour
```

- `L_balance`: KL divergence from uniform cluster sizes
- `L_contrastive`: Nodes in same cluster have similar embeddings
- `L_tour`: Adjacent nodes in tour should be in same cluster

## Installation

```bash
pip install torch numpy scipy tqdm
```

## Usage

### Training Stage 1 (Clustering)

```bash
# Using existing EDISCO data format
python train.py \
    --data_path data/tsp100/tsp100_lkh_train.txt \
    --val_data_path data/tsp100/tsp100_lkh_valid.txt \
    --num_clusters 10 \
    --epochs 100 \
    --batch_size 32
```

### Evaluation Metrics

The model is evaluated by counting **tour edge cuts** - the number of tour edges that cross cluster boundaries. Lower is better.

```
Val Tour Edge Cuts: EGNN=12.3, K-Means=18.7
```

## File Structure

```
DGC-TSP/
├── dgc_tsp/
│   ├── __init__.py
│   ├── encoder.py      # EGNN encoder
│   ├── clustering.py   # EGNNClusteringNetwork
│   └── utils.py        # Dataset, utilities
├── train.py            # Training script
├── scripts/
│   └── generate_data.py
└── related_works/      # Reference papers
```

## Comparison with Related Work

| Paper | Venue | Partition | Equivariant | Tour-Aware |
|-------|-------|-----------|-------------|------------|
| GLOP | AAAI 2024 | Sliding window | No | No |
| UDC | NeurIPS 2024 | Learned GNN | No | No |
| H-TSP | AAAI 2023 | RL selection | No | Implicit |
| **Ours** | - | **Learned EGNN** | **Yes** | **Yes** |

## References

- [UDC: Unified Neural Divide-and-Conquer](https://arxiv.org/abs/2407.00312) (NeurIPS 2024)
- [GLOP: Learning Global Partition and Local Construction](https://arxiv.org/abs/2312.08224) (AAAI 2024)
- [H-TSP: Hierarchically Solving Large-Scale TSP](https://arxiv.org/abs/2304.09395) (AAAI 2023)
- [EDISCO: Equivariant Diffusion for Combinatorial Optimization](https://github.com/ValleyC/EDISCO)

## License

MIT License
