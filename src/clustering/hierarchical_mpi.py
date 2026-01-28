"""
Distributed Hierarchical Clustering with MPI
Divide-and-conquer approach using local clustering + global merging.

UPDATED: Added --use-pca option for better clustering quality!

KEY STRATEGY:
1. Partition data across MPI ranks
2. Each rank performs local hierarchical clustering
3. Rank 0 gathers local cluster summaries
4. Rank 0 performs second-level hierarchical clustering
5. Broadcast final centroids and reassign all points
"""

import numpy as np
import sys
import time
import json
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py not installed")
    sys.exit(1)

import config


def local_hierarchical_clustering(X_local: np.ndarray, n_clusters: int, 
                                  method: str = 'ward', sample_size: int = None):
    """Perform hierarchical clustering on local data chunk."""
    n_local = len(X_local)
    
    if sample_size is not None and n_local > sample_size:
        indices = np.random.choice(n_local, sample_size, replace=False)
        X_sample = X_local[indices]
    else:
        X_sample = X_local
        indices = np.arange(n_local)
    
    Z = linkage(X_sample, method=method)
    labels_sample = fcluster(Z, n_clusters, criterion='maxclust')
    
    if sample_size is not None and n_local > sample_size:
        centroids = np.array([X_sample[labels_sample == i].mean(axis=0) 
                             for i in range(1, n_clusters + 1)])
        distances = np.linalg.norm(X_local[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1) + 1
    else:
        labels = labels_sample
    
    summaries = []
    for cluster_id in range(1, n_clusters + 1):
        mask = labels == cluster_id
        X_cluster = X_local[mask]
        
        if len(X_cluster) > 0:
            centroid = X_cluster.mean(axis=0)
            size = len(X_cluster)
            
            if size > 1:
                distances = np.linalg.norm(X_cluster - centroid, axis=1)
                radius = distances.mean()
            else:
                radius = 0.0
            
            summaries.append({
                'centroid': centroid,
                'size': size,
                'radius': radius
            })
    
    return labels, summaries


def global_hierarchical_clustering(all_summaries: list, n_global_clusters: int, method: str = 'ward'):
    """Perform hierarchical clustering on cluster summaries."""
    centroids = np.array([s['centroid'] for s in all_summaries])
    sizes = np.array([s['size'] for s in all_summaries])
    
    Z = linkage(centroids, method=method)
    global_labels = fcluster(Z, n_global_clusters, criterion='maxclust')
    
    global_centroids = []
    for gid in range(1, n_global_clusters + 1):
        mask = global_labels == gid
        if mask.sum() > 0:
            weighted_sum = np.sum(centroids[mask] * sizes[mask, np.newaxis], axis=0)
            total_weight = sizes[mask].sum()
            global_centroids.append(weighted_sum / total_weight)
    
    return global_labels, np.array(global_centroids)


def hierarchical_mpi(X_local, k_global, k_local=None, method='ward', sample_size=None,
                     comm=None, rank=None, size=None):
    """Distributed hierarchical clustering."""
    if comm is None:
        comm = MPI.COMM_WORLD
    if rank is None:
        rank = comm.Get_rank()
    if size is None:
        size = comm.Get_size()
    
    if k_local is None:
        k_local = max(k_global, 5)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED HIERARCHICAL CLUSTERING WITH MPI")
        print(f"{'='*80}")
        print(f"Number of processes: {size}")
        print(f"Global clusters: {k_global}")
        print(f"Local clusters per rank: {k_local}")
        print(f"Linkage method: {method}")
    
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Phase 1: Local Clustering
    if rank == 0:
        print("\n--- Phase 1: Local Hierarchical Clustering ---")
    
    local_start = MPI.Wtime()
    local_labels, local_summaries = local_hierarchical_clustering(
        X_local, k_local, method=method, sample_size=sample_size
    )
    local_elapsed = MPI.Wtime() - local_start
    
    print(f"Rank {rank}: Created {len(local_summaries)} local clusters in {local_elapsed:.2f}s")
    
    # Phase 2: Gather Summaries
    if rank == 0:
        print("\n--- Phase 2: Gathering Cluster Summaries ---")
    
    all_summaries = comm.gather(local_summaries, root=0)
    
    # Phase 3: Global Clustering
    if rank == 0:
        print("\n--- Phase 3: Global Hierarchical Clustering ---")
        
        all_summaries_flat = [s for rank_summaries in all_summaries for s in rank_summaries]
        print(f"Total local clusters: {len(all_summaries_flat)}")
        
        global_start = MPI.Wtime()
        global_labels, global_centroids = global_hierarchical_clustering(
            all_summaries_flat, k_global, method=method
        )
        global_elapsed = MPI.Wtime() - global_start
        
        print(f"Created {len(global_centroids)} global clusters in {global_elapsed:.2f}s")
    else:
        global_centroids = None
    
    # Phase 4: Broadcast
    if rank == 0:
        print("\n--- Phase 4: Broadcasting Global Centroids ---")
        centroid_shape = global_centroids.shape
    else:
        centroid_shape = None
    
    centroid_shape = comm.bcast(centroid_shape, root=0)
    
    if rank != 0:
        global_centroids = np.zeros(centroid_shape)
    
    comm.Bcast(global_centroids, root=0)
    
    # Phase 5: Final Assignment
    if rank == 0:
        print("\n--- Phase 5: Assigning Points to Global Clusters ---")
    
    distances = np.linalg.norm(X_local[:, np.newaxis, :] - global_centroids[np.newaxis, :, :], axis=2)
    final_labels = np.argmin(distances, axis=1)
    
    comm.Barrier()
    total_time = MPI.Wtime() - start_time
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED HIERARCHICAL CLUSTERING COMPLETE")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"{'='*80}")
    
    return global_centroids, final_labels, total_time


def load_and_partition_data(comm, rank, size, sample_fraction=None, use_pca=True):
    """Load data and partition across MPI ranks."""
    if rank == 0:
        print("\n--- Loading and Partitioning Data ---")
        
        # Choose data source based on use_pca flag
        if use_pca:
            pca_path = config.DATA_DIR / "processed_X_pca.npy"
            if pca_path.exists():
                X_full = np.load(pca_path)
                print(f"✓ [PCA MODE] Loaded: {pca_path}")
                print(f"  Shape: {X_full.shape}")
                print(f"  Expected silhouette: ~0.25-0.30")
            else:
                print(f"⚠ WARNING: PCA data not found at {pca_path}")
                print(f"  Falling back to original features...")
                X_full = np.load(config.PROCESSED_DATA_PATH)
        else:
            X_full = np.load(config.PROCESSED_DATA_PATH)
            print(f"✓ [ORIGINAL MODE] Loaded: {config.PROCESSED_DATA_PATH}")
            print(f"  Shape: {X_full.shape}")
        
        if hasattr(X_full, 'toarray'):
            X_full = X_full.toarray()
        
        n_total, n_features = X_full.shape
        
        # Sample if requested
        if sample_fraction is not None and sample_fraction < 1.0:
            sample_size = int(n_total * sample_fraction)
            indices = np.random.choice(n_total, sample_size, replace=False)
            X_full = X_full[indices]
            n_total = sample_size
            print(f"Sampled to {n_total:,} rows ({sample_fraction*100:.0f}%)")
        
        # Partition
        base_size = n_total // size
        remainder = n_total % size
        
        counts = np.array([base_size + (1 if i < remainder else 0) for i in range(size)])
        displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
        
        print(f"\nPartition sizes:")
        for i, count in enumerate(counts):
            print(f"  Rank {i}: {count:,} samples")
    else:
        X_full = None
        n_features = None
        counts = None
        displacements = None
    
    # Broadcast
    n_features = comm.bcast(n_features, root=0)
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)
    
    X_local = np.zeros((counts[rank], n_features))
    
    if rank == 0:
        print("\nScattering data...")
        sendbuf = X_full.astype(np.float64)
    else:
        sendbuf = None
    
    counts_bytes = (counts * n_features).astype(int)
    displacements_bytes = (displacements * n_features).astype(int)
    
    comm.Scatterv([sendbuf, counts_bytes, displacements_bytes, MPI.DOUBLE],
                  X_local, root=0)
    
    if rank == 0:
        print("Data scattered successfully!")
    
    return X_local


def save_results(centroids, labels_local, elapsed_time, k_global, nprocs, rank, comm):
    """Save clustering results."""
    import pandas as pd
    
    output_dir = config.CLUSTERS_DIR / "hierarchical_mpi"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        centroids_path = output_dir / f'centroids_k{k_global}_np{nprocs}.npy'
        np.save(centroids_path, centroids)
        print(f"Saved centroids to {centroids_path}")
        
        metrics = {
            'k': k_global,
            'n_processes': nprocs,
            'elapsed_time': elapsed_time
        }
        
        metrics_path = output_dir / f'metrics_k{k_global}_np{nprocs}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Append to scaling log
        scaling_log_path = config.SCALING_DIR / 'hierarchical_mpi_scaling.csv'
        
        if scaling_log_path.exists():
            scaling_df = pd.read_csv(scaling_log_path)
        else:
            scaling_df = pd.DataFrame()
        
        new_row = pd.DataFrame([metrics])
        scaling_df = pd.concat([scaling_df, new_row], ignore_index=True)
        scaling_df.to_csv(scaling_log_path, index=False)
        
        print(f"Appended to scaling log: {scaling_log_path}")
    
    # Gather all labels
    all_labels = comm.gather(labels_local, root=0)
    
    if rank == 0:
        all_labels = np.concatenate(all_labels)
        labels_path = output_dir / f'labels_k{k_global}_np{nprocs}.npy'
        np.save(labels_path, all_labels)
        print(f"Saved all labels to {labels_path}")


def main():
    """Main entry point."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*80)
        print("MPI HIERARCHICAL CLUSTERING")
        print("="*80)
        print(f"MPI size: {size} processes")
    
    import argparse
    parser = argparse.ArgumentParser(description='Distributed Hierarchical Clustering with MPI')
    parser.add_argument('--k', type=int, default=10, help='Number of global clusters')
    parser.add_argument('--k-local', type=int, default=None, help='Local clusters per rank')
    parser.add_argument('--method', type=str, default='ward', 
                       choices=['ward', 'complete', 'average', 'single'])
    parser.add_argument('--sample', type=float, default=None, help='Sample fraction (0-1)')
    parser.add_argument('--local-sample', type=int, default=None, help='Local sample size')
    # NEW: PCA option
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA-transformed data (default: True)')
    parser.add_argument('--no-pca', action='store_false', dest='use_pca',
                       help='Use original feature-selected data')
    
    args = parser.parse_args()
    
    if rank == 0:
        if args.use_pca:
            print("\n*** PCA MODE: Using PCA-transformed data ***")
        else:
            print("\n*** ORIGINAL MODE: Using feature-selected data ***")
    
    # Load and partition data with PCA option
    X_local = load_and_partition_data(
        comm, rank, size, 
        sample_fraction=args.sample,
        use_pca=args.use_pca  # NEW!
    )
    
    # Run distributed hierarchical clustering
    centroids, labels_local, elapsed_time = hierarchical_mpi(
        X_local=X_local,
        k_global=args.k,
        k_local=args.k_local,
        method=args.method,
        sample_size=args.local_sample,
        comm=comm,
        rank=rank,
        size=size
    )
    
    # Save results
    save_results(centroids, labels_local, elapsed_time, args.k, size, rank, comm)
    
    if rank == 0:
        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)


if __name__ == "__main__":
    main()