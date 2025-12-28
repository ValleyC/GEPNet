"""
Training script for E(2)-Equivariant Graph Clustering (Stage 1).

This trains only the clustering network to partition TSP instances
in a tour-aware manner using EGNN-based representations.

Usage:
    python train.py --data_path data/tsp100_train.txt --num_clusters 10 --epochs 100
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dgc_tsp import EGNNClusteringNetwork
from dgc_tsp.clustering import KMeansBaseline
from dgc_tsp.utils import TSPDataset, collate_tsp_batch


def parse_args():
    parser = argparse.ArgumentParser(description='Train EGNN Clustering (Stage 1)')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data file')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--num_egnn_layers', type=int, default=4,
                        help='Number of EGNN layers')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for cluster assignment')

    # Loss weights
    parser.add_argument('--lambda_balance', type=float, default=0.1,
                        help='Weight for balance loss')
    parser.add_argument('--lambda_tour', type=float, default=1.0,
                        help='Weight for tour alignment loss')
    parser.add_argument('--lambda_contrastive', type=float, default=0.1,
                        help='Weight for contrastive loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')

    # Logging
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N batches')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tour_loss = 0.0
    total_balance_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        coords = batch['coords'].to(device)
        tour = batch['tour'].to(device)

        # Forward pass
        optimizer.zero_grad()
        losses, output = model.compute_loss(coords, tour=tour)

        # Backward pass
        loss = losses['total']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_tour_loss += losses['tour'].item()
        total_balance_loss += losses['balance'].item()
        total_contrastive_loss += losses['contrastive'].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_tour = total_tour_loss / num_batches
            avg_balance = total_balance_loss / num_batches
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} (Tour: {avg_tour:.4f}, Bal: {avg_balance:.4f})")

    return {
        'loss': total_loss / num_batches,
        'tour_loss': total_tour_loss / num_batches,
        'balance_loss': total_balance_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device, kmeans_baseline=None):
    """Validate model by counting tour edge cuts."""
    model.eval()

    total_cuts_egnn = 0
    total_cuts_kmeans = 0
    total_edges = 0
    num_samples = 0

    for batch in dataloader:
        coords = batch['coords'].to(device)
        tour = batch['tour'].to(device)

        batch_size = coords.shape[0]

        for b in range(batch_size):
            # Get EGNN cluster assignments
            assignments = model.predict_clusters(coords[b])
            cuts = model.compute_tour_edge_cuts(assignments, tour[b])
            total_cuts_egnn += cuts

            # Get K-Means baseline assignments
            if kmeans_baseline is not None:
                km_assignments = kmeans_baseline(coords[b])
                km_cuts = model.compute_tour_edge_cuts(km_assignments, tour[b])
                total_cuts_kmeans += km_cuts

            total_edges += len(tour[b])
            num_samples += 1

    avg_cuts_egnn = total_cuts_egnn / num_samples if num_samples > 0 else 0
    avg_cuts_kmeans = total_cuts_kmeans / num_samples if num_samples > 0 else 0
    cut_ratio = total_cuts_egnn / total_edges if total_edges > 0 else 0

    return {
        'avg_cuts_egnn': avg_cuts_egnn,
        'avg_cuts_kmeans': avg_cuts_kmeans,
        'cut_ratio': cut_ratio,
        'num_samples': num_samples,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup experiment name
    if args.exp_name is None:
        args.exp_name = f"egnn_cluster_k{args.num_clusters}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create save directory
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Load data
    print(f"Loading training data from: {args.data_path}")
    train_dataset = TSPDataset(args.data_path, compute_adjacency=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tsp_batch,
    )
    print(f"Training samples: {len(train_dataset)}")

    val_loader = None
    if args.val_data_path:
        print(f"Loading validation data from: {args.val_data_path}")
        val_dataset = TSPDataset(args.val_data_path, compute_adjacency=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_tsp_batch,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = EGNNClusteringNetwork(
        num_clusters=args.num_clusters,
        hidden_dim=args.hidden_dim,
        num_egnn_layers=args.num_egnn_layers,
        temperature=args.temperature,
        lambda_balance=args.lambda_balance,
        lambda_tour=args.lambda_tour,
        lambda_contrastive=args.lambda_contrastive,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # K-Means baseline for comparison
    kmeans_baseline = KMeansBaseline(num_clusters=args.num_clusters).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5,
        )
    else:
        scheduler = None

    # Training loop
    best_cuts = float('inf')
    best_epoch = 0

    print("\n" + "="*60)
    print("Starting Stage 1: EGNN Clustering Training")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Warmup learning rate
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args)

        # Update scheduler
        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Tour: {train_metrics['tour_loss']:.4f}, "
              f"Balance: {train_metrics['balance_loss']:.4f}, "
              f"Contrastive: {train_metrics['contrastive_loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if val_loader is not None and epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device, kmeans_baseline)
            print(f"  Val Tour Edge Cuts: EGNN={val_metrics['avg_cuts_egnn']:.1f}, "
                  f"K-Means={val_metrics['avg_cuts_kmeans']:.1f}")
            print(f"  Val Cut Ratio: {val_metrics['cut_ratio']:.2%}")

            # Save best model (fewer cuts = better)
            if val_metrics['avg_cuts_egnn'] < best_cuts:
                best_cuts = val_metrics['avg_cuts_egnn']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_cuts': val_metrics['avg_cuts_egnn'],
                    'args': args,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  -> New best model! Avg cuts: {best_cuts:.1f}")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt'))

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best average tour edge cuts: {best_cuts:.1f} at epoch {best_epoch}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
    }, os.path.join(save_dir, 'final_model.pt'))
    print(f"Final model saved to: {save_dir}/final_model.pt")


if __name__ == '__main__':
    main()
