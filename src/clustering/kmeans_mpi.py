"""
Distributed K-Means Clustering with MPI
True MapReduce-style implementation using mpi4py.

UPDATED: Added --use-pca option for better clustering quality!
         Default is now PCA mode for better silhouette scores.

KEY CHANGES:
- load_and_partition_data now has use_pca parameter
- Default uses PCA data (processed_X_pca.npy) for better clustering
- Use --no-pca to use original feature-selected data
"""

import numpy as np
import sys
import time
import json
from pathlib import Path

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py not installed. Install with: pip install mpi4py")
    sys.exit(1)

import config


def initialize_centroids(X: np.ndarray, k: int, method: str = 'kmeans++', 
                        random_state: int = None):
    """
    Initialize centroids using k-means++ or random selection.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    if method == 'random':
        indices = np.random.choice(n_samples, k, replace=False)
        return X[indices].copy()
    
    elif method == 'kmeans++':
        centroids = np.zeros((k, n_features))
        centroids[0] = X[np.random.randint(n_samples)]
        
        for i in range(1, k):
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 
                               for c in centroids[:i]], axis=0)
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def assign_labels(X_local: np.ndarray, centroids: np.ndarray):
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(X_local[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def compute_local_statistics(X_local: np.ndarray, labels: np.ndarray, k: int):
    """Compute local sum and count per cluster for MapReduce aggregation."""
    n_features = X_local.shape[1]
    local_sums = np.zeros((k, n_features))
    local_counts = np.zeros(k, dtype=np.int64)
    
    for cluster_id in range(k):
        mask = labels == cluster_id
        local_counts[cluster_id] = mask.sum()
        if local_counts[cluster_id] > 0:
            local_sums[cluster_id] = X_local[mask].sum(axis=0)
    
    return local_sums, local_counts


def kmeans_mpi(X_local, k, max_iter=100, tol=1e-4, init_method='kmeans++',
               random_state=None, comm=None, rank=None, size=None):
    """
    Distributed K-Means using MPI with MapReduce-style aggregation.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if rank is None:
        rank = comm.Get_rank()
    if size is None:
        size = comm.Get_size()
    
    n_local, n_features = X_local.shape
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED K-MEANS CLUSTERING")
        print(f"{'='*80}")
        print(f"K = {k} clusters")
        print(f"Processes = {size}")
        print(f"Max iterations = {max_iter}")
        print(f"Convergence tolerance = {tol}")
        print(f"Initialization = {init_method}")
    
    # Timing
    comm.Barrier()
    start_time = MPI.Wtime()
    total_comp_time = 0.0
    total_comm_time = 0.0
    
    # Gather samples for initialization
    sample_size_per_rank = min(1000, n_local)
    local_sample = X_local[np.random.choice(n_local, sample_size_per_rank, replace=False)]
    all_samples = comm.gather(local_sample, root=0)
    
    # Initialize centroids on rank 0
    if rank == 0:
        X_init = np.vstack(all_samples)
        print(f"Gathered {len(X_init):,} samples for initialization")
        centroids = initialize_centroids(X_init, k, method=init_method, 
                                        random_state=random_state)
        print(f"Initialized {k} centroids using {init_method}")
    else:
        centroids = np.zeros((k, n_features))
    
    # Broadcast initial centroids
    comm_start = MPI.Wtime()
    comm.Bcast(centroids, root=0)
    total_comm_time += MPI.Wtime() - comm_start
    
    # Main K-Means loop
    converged = False
    iteration = 0
    
    if rank == 0:
        print("\n--- Iterative Refinement ---")
    
    for iteration in range(max_iter):
        iter_start = MPI.Wtime()
        
        # MAP PHASE
        comp_start = MPI.Wtime()
        labels_local = assign_labels(X_local, centroids)
        local_sums, local_counts = compute_local_statistics(X_local, labels_local, k)
        total_comp_time += MPI.Wtime() - comp_start
        
        # REDUCE PHASE
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        
        comm_start = MPI.Wtime()
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        total_comm_time += MPI.Wtime() - comm_start
        
        # UPDATE PHASE
        comp_start = MPI.Wtime()
        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(k):
            if global_counts[cluster_id] > 0:
                new_centroids[cluster_id] = global_sums[cluster_id] / global_counts[cluster_id]
            else:
                new_centroids[cluster_id] = centroids[cluster_id]
        
        centroid_shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        total_comp_time += MPI.Wtime() - comp_start
        
        iter_time = MPI.Wtime() - iter_start
        
        if rank == 0:
            if iteration < 5 or iteration % 10 == 9 or centroid_shift < tol:
                print(f"  Iteration {iteration+1}: shift={centroid_shift:.6f}, time={iter_time:.3f}s")
        
        if centroid_shift < tol:
            converged = True
            if rank == 0:
                print(f"  Converged at iteration {iteration+1}!")
            break
    
    comm.Barrier()
    elapsed_time = MPI.Wtime() - start_time
    
    if rank == 0:
        print(f"\n--- Summary ---")
        print(f"Total iterations: {iteration + 1}")
        print(f"Converged: {converged}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Computation time: {total_comp_time:.2f}s ({100*total_comp_time/elapsed_time:.1f}%)")
        print(f"Communication time: {total_comm_time:.2f}s ({100*total_comm_time/elapsed_time:.1f}%)")
        
        print(f"\nCluster sizes:")
        for i, count in enumerate(global_counts):
            print(f"  Cluster {i}: {count:,} points")
    
    return centroids, labels_local, iteration + 1, elapsed_time, total_comp_time, total_comm_time


def load_and_partition_data(comm, rank, size, use_full_data=False, use_pca=True):
    """
    Load data and partition across MPI ranks.
    
    UPDATED: Added use_pca option for better clustering quality!
    
    Args:
        comm: MPI communicator
        rank: MPI rank
        size: Number of processes
        use_full_data: If True, use entire dataset without sampling
        use_pca: If True, use PCA-transformed data (better silhouette ~0.25)
                 If False, use feature-selected data (silhouette ~0.08 but interpretable)
        
    Returns:
        X_local: Local chunk of data for this rank
    """
    if rank == 0:
        print("\n--- Loading and Partitioning Data ---")
        
        # Choose data source based on use_pca flag
        if use_pca:
            pca_path = config.DATA_DIR / "processed_X_pca.npy"
            if pca_path.exists():
                X_full = np.load(pca_path)
                print(f"✓ [PCA MODE] Loaded PCA-transformed data: {X_full.shape}")
                print(f"  Expected silhouette: ~0.25 (3.3x better than original)")
            else:
                print(f"⚠ WARNING: PCA data not found at {pca_path}")
                print(f"  Run step2a_dim_reduction.slurm first!")
                print(f"  Falling back to feature-selected data...")
                X_full = np.load(config.PROCESSED_DATA_PATH)
                use_pca = False  # Update flag for logging
        else:
            X_full = np.load(config.PROCESSED_DATA_PATH)
            print(f"✓ [ORIGINAL MODE] Loaded feature-selected data: {X_full.shape}")
            print(f"  Note: Lower silhouette (~0.08) but directly interpretable features")
        
        # Convert sparse to dense if needed
        if hasattr(X_full, 'toarray'):
            X_full = X_full.toarray()
        
        n_total, n_features = X_full.shape
        print(f"Total samples: {n_total:,}")
        print(f"Features/Components: {n_features}")
        
        # Load feature names if using original data
        if not use_pca:
            feature_names_path = config.DATA_DIR / "feature_names.txt"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
                print(f"Using features: {feature_names}")
        else:
            print(f"Using {n_features} PCA components (95% variance preserved)")

        # Subsampling logic
        if use_full_data:
            print(f"Using FULL dataset: {n_total:,} rows")
        else:
            MAX_SAMPLES_FOR_SCALING = 500_000
            if n_total > MAX_SAMPLES_FOR_SCALING:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_total, size=MAX_SAMPLES_FOR_SCALING, replace=False)
                X_full = X_full[idx]
                n_total, n_features = X_full.shape
                print(f"Subsampled to {n_total:,} rows for MPI scaling experiment")
        
        # Compute partition sizes
        base_size = n_total // size
        remainder = n_total % size
        
        counts = np.array([base_size + (1 if i < remainder else 0) for i in range(size)])
        displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
        
        print(f"\nPartition sizes per rank:")
        for i, count in enumerate(counts):
            print(f"  Rank {i}: {count:,} samples ({100*count/n_total:.1f}%)")
    else:
        X_full = None
        n_features = None
        counts = None
        displacements = None
    
    # Broadcast metadata
    n_features = comm.bcast(n_features, root=0)
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)
    
    # Allocate local buffer
    X_local = np.zeros((counts[rank], n_features))
    
    # Scatter data
    if rank == 0:
        print("\nScattering data to all ranks...")
        sendbuf = X_full.astype(np.float64)
    else:
        sendbuf = None
    
    # Use Scatterv for variable-size chunks
    # Convert counts and displacements to bytes
    counts_bytes = (counts * n_features).astype(int)
    displacements_bytes = (displacements * n_features).astype(int)
    
    comm.Scatterv([sendbuf, counts_bytes, displacements_bytes, MPI.DOUBLE],
                  X_local, root=0)
    
    if rank == 0:
        print(f"Data scattered successfully!")
    
    return X_local


def save_results(centroids, labels_local, n_iter, elapsed_time,
                comp_time, comm_time, k, nprocs, rank, comm):
    """Save clustering results and scaling metrics."""
    import pandas as pd
    
    output_dir = config.CLUSTERS_DIR / "kmeans_mpi"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        # Save centroids
        centroids_path = output_dir / f'centroids_k{k}_np{nprocs}.npy'
        np.save(centroids_path, centroids)
        print(f"Saved centroids to {centroids_path}")
        
        # Update scaling log
        metrics = {
            'n_processes': nprocs,
            'k': k,
            'n_iterations': n_iter,
            'elapsed_time': elapsed_time,
            'computation_time': comp_time,
            'communication_time': comm_time,
            'comp_pct': 100 * comp_time / elapsed_time,
            'comm_pct': 100 * comm_time / elapsed_time
        }
        
        scaling_log_path = config.SCALING_DIR / 'kmeans_mpi_scaling.csv'
        scaling_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if scaling_log_path.exists():
            scaling_df = pd.read_csv(scaling_log_path)
        else:
            scaling_df = pd.DataFrame()
        
        new_row = pd.DataFrame([metrics])
        scaling_df = pd.concat([scaling_df, new_row], ignore_index=True)
        scaling_df.to_csv(scaling_log_path, index=False)
        
        print(f"Appended to scaling log: {scaling_log_path}")
    
    # Gather all labels to rank 0
    all_labels = comm.gather(labels_local, root=0)
    
    if rank == 0:
        all_labels = np.concatenate(all_labels)
        labels_path = output_dir / f'labels_k{k}_np{nprocs}.npy'
        np.save(labels_path, all_labels)
        print(f"Saved all labels to {labels_path}")


def main():
    """
    Main entry point for distributed K-Means.
    
    Usage:
        # With PCA (default, better silhouette):
        mpirun -n 4 python kmeans_mpi.py
        
        # Without PCA (interpretable features):
        mpirun -n 4 python kmeans_mpi.py --no-pca
        
        # Full data mode:
        mpirun -n 8 python kmeans_mpi.py --full-data
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*80)
        print("MPI K-MEANS CLUSTERING")
        print("="*80)
        print(f"MPI size: {size} processes")
    
    import argparse
    parser = argparse.ArgumentParser(description='Distributed K-Means with MPI')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters')
    parser.add_argument('--max-iter', type=int, default=config.MPI_MAX_ITERATIONS,
                       help='Maximum iterations')
    parser.add_argument('--tol', type=float, default=config.MPI_CONVERGENCE_TOL,
                       help='Convergence tolerance')
    parser.add_argument('--init', type=str, default='kmeans++',
                       choices=['kmeans++', 'random'], help='Initialization method')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                       help='Random seed')
    parser.add_argument('--full-data', action='store_true',
                       help='Use full dataset instead of sampling')
    # NEW: PCA option
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA-transformed data for better clustering (default: True)')
    parser.add_argument('--no-pca', action='store_false', dest='use_pca',
                       help='Use original feature-selected data (interpretable but lower silhouette)')
    
    args = parser.parse_args()
    
    if rank == 0:
        if args.use_pca:
            print("\n*** PCA MODE: Using PCA-transformed data ***")
            print("    (Better silhouette scores ~0.25)")
        else:
            print("\n*** ORIGINAL MODE: Using feature-selected data ***")
            print("    (Directly interpretable, silhouette ~0.08)")
        
        if args.full_data:
            print("*** FULL DATA MODE: Using entire dataset ***")
        else:
            print("*** SAMPLE MODE: Using max 500k samples ***")
    
    # Load and partition data with PCA option
    X_local = load_and_partition_data(
        comm, rank, size, 
        use_full_data=args.full_data,
        use_pca=args.use_pca  # NEW!
    )
    
    if rank == 0:
        print(f"\nLocal data shape on rank {rank}: {X_local.shape}")
    
    # Run distributed K-Means
    centroids, labels_local, n_iter, elapsed_time, comp_time, comm_time = kmeans_mpi(
        X_local=X_local,
        k=args.k,
        max_iter=args.max_iter,
        tol=args.tol,
        init_method=args.init,
        random_state=args.seed,
        comm=comm,
        rank=rank,
        size=size
    )
    
    # Save results
    save_results(centroids, labels_local, n_iter, elapsed_time,
                comp_time, comm_time, args.k, size, rank, comm)
    
    if rank == 0:
        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)


if __name__ == "__main__":
    main()