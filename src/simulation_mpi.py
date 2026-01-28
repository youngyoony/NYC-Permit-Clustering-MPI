#!/usr/bin/env python3
"""
Simulation MPI: Distributed K-Means Clustering on Synthetic Data
AMS 598 - Big Data Analysis

This script runs distributed K-Means on synthetic data using MPI
and compares results with single-node baseline.

Usage:
    mpirun -np 4 python simulation_mpi.py
    mpirun -np 8 python simulation_mpi.py
    mpirun -np 16 python simulation_mpi.py
"""

import numpy as np
import time
import json
import os
from mpi4py import MPI
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MPI Setup
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# Configuration - MUST MATCH BASELINE!
# ============================================================
CONFIG = {
    'n_clusters': 10,      # FIXED: Match baseline (was 5)
    'max_iter': 100,
    'tol': 1e-4,
    'random_state': 42
}

def load_data():
    """Load synthetic data (rank 0 only, then scatter)."""
    if rank == 0:
        if not os.path.exists('simulation_X.npy'):
            print("ERROR: Run simulation_baseline.py first to generate data!")
            comm.Abort(1)
        
        X = np.load('simulation_X.npy')
        y_true = np.load('simulation_y_true.npy')
        baseline_labels = np.load('simulation_baseline_labels.npy')
        baseline_centroids = np.load('simulation_baseline_centroids.npy')
        
        print("=" * 60)
        print(f"MPI K-MEANS ON SIMULATION DATA")
        print(f"Processes: {size}")
        print("=" * 60)
        print(f"  Total samples: {X.shape[0]:,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Clusters: {CONFIG['n_clusters']}")
        print(f"  Samples per process: ~{X.shape[0] // size:,}")
    else:
        X = None
        y_true = None
        baseline_labels = None
        baseline_centroids = None
    
    return X, y_true, baseline_labels, baseline_centroids

def distribute_data(X):
    """Scatter data to all processes."""
    if rank == 0:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Calculate chunk sizes
        chunk_size = n_samples // size
        remainder = n_samples % size
        
        # Create chunks (handle uneven division)
        chunks = []
        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(X[start:end].copy())
            start = end
        
        send_info = (n_features, CONFIG['n_clusters'])
    else:
        chunks = None
        send_info = None
    
    # Broadcast metadata
    n_features, n_clusters = comm.bcast(send_info, root=0)
    
    # Scatter data chunks
    local_X = comm.scatter(chunks, root=0)
    
    return local_X, n_features, n_clusters

def kmeans_plusplus_init(X, n_clusters, n_features):
    """Distributed K-Means++ initialization."""
    if rank == 0:
        # Simple K-Means++ on rank 0 (could be improved for very large data)
        np.random.seed(CONFIG['random_state'])
        
        # First centroid: random point
        idx = np.random.randint(X.shape[0])
        centroids = [X[idx]]
        
        for _ in range(1, n_clusters):
            # Compute distances to nearest centroid
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            # Probability proportional to distance squared
            probs = dists / dists.sum()
            # Sample next centroid
            idx = np.random.choice(len(X), p=probs)
            centroids.append(X[idx])
        
        centroids = np.array(centroids)
    else:
        centroids = np.empty((n_clusters, n_features))
    
    # Broadcast initial centroids to all processes
    comm.Bcast(centroids, root=0)
    
    return centroids

def assign_clusters(local_X, centroids):
    """Assign each local point to nearest centroid."""
    # Compute distances to all centroids
    distances = np.zeros((local_X.shape[0], len(centroids)))
    for i, c in enumerate(centroids):
        distances[:, i] = np.sum((local_X - c) ** 2, axis=1)
    
    # Assign to nearest
    local_labels = np.argmin(distances, axis=1)
    
    return local_labels

def update_centroids(local_X, local_labels, n_clusters, n_features):
    """Update centroids using MPI_Allreduce."""
    # Local partial sums and counts
    local_sums = np.zeros((n_clusters, n_features))
    local_counts = np.zeros(n_clusters)
    
    for k in range(n_clusters):
        mask = local_labels == k
        local_counts[k] = np.sum(mask)
        if local_counts[k] > 0:
            local_sums[k] = np.sum(local_X[mask], axis=0)
    
    # Global aggregation
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)
    
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
    
    # Compute new centroids
    new_centroids = np.zeros((n_clusters, n_features))
    for k in range(n_clusters):
        if global_counts[k] > 0:
            new_centroids[k] = global_sums[k] / global_counts[k]
    
    return new_centroids

def kmeans_mpi(local_X, n_clusters, n_features, max_iter=100, tol=1e-4):
    """Distributed K-Means main loop."""
    
    # Gather some data on rank 0 for initialization
    all_X = comm.gather(local_X, root=0)
    if rank == 0:
        all_X = np.vstack(all_X)
    else:
        all_X = None
    
    # Initialize centroids
    centroids = kmeans_plusplus_init(all_X if rank == 0 else None, n_clusters, n_features)
    
    # Timing
    comm.Barrier()
    start_time = MPI.Wtime()
    comm_time = 0.0
    
    for iteration in range(max_iter):
        # Assign clusters (local computation)
        local_labels = assign_clusters(local_X, centroids)
        
        # Update centroids (requires communication)
        comm_start = MPI.Wtime()
        new_centroids = update_centroids(local_X, local_labels, n_clusters, n_features)
        comm_time += MPI.Wtime() - comm_start
        
        # Check convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        
        if centroid_shift < tol:
            if rank == 0:
                print(f"  Converged at iteration {iteration + 1}")
            break
    
    comm.Barrier()
    total_time = MPI.Wtime() - start_time
    
    return local_labels, centroids, iteration + 1, total_time, comm_time

def gather_results(local_labels):
    """Gather all labels to rank 0."""
    all_labels = comm.gather(local_labels, root=0)
    if rank == 0:
        all_labels = np.concatenate(all_labels)
    return all_labels

def compute_metrics(X, labels, y_true, baseline_labels):
    """Compute clustering metrics (rank 0 only)."""
    if rank != 0:
        return None
    
    # Silhouette (sample for speed)
    if len(X) > 50000:
        sample_idx = np.random.choice(len(X), 50000, replace=False)
        sil_score = silhouette_score(X[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(X, labels)
    
    # Agreement with ground truth
    ari_true = adjusted_rand_score(y_true, labels)
    nmi_true = normalized_mutual_info_score(y_true, labels)
    
    # Agreement with baseline
    ari_baseline = adjusted_rand_score(baseline_labels, labels)
    nmi_baseline = normalized_mutual_info_score(baseline_labels, labels)
    
    # Direct label agreement (note: labels may be permuted)
    # So we use ARI instead for meaningful comparison
    agreement = ari_baseline * 100  # Convert to percentage-like scale
    
    return {
        'silhouette': sil_score,
        'ari_vs_truth': ari_true,
        'nmi_vs_truth': nmi_true,
        'ari_vs_baseline': ari_baseline,
        'nmi_vs_baseline': nmi_baseline,
        'agreement_pct': agreement
    }

def main():
    # Load data
    X, y_true, baseline_labels, baseline_centroids = load_data()
    
    # Distribute data
    local_X, n_features, n_clusters = distribute_data(X)
    
    if rank == 0:
        print(f"\n  Local data shape on rank 0: {local_X.shape}")
    
    # Run distributed K-Means
    if rank == 0:
        print("\nRunning distributed K-Means...")
    
    local_labels, centroids, iterations, total_time, comm_time = kmeans_mpi(
        local_X, n_clusters, n_features,
        max_iter=CONFIG['max_iter'],
        tol=CONFIG['tol']
    )
    
    # Gather results
    all_labels = gather_results(local_labels)
    
    # Compute metrics on rank 0
    if rank == 0:
        metrics = compute_metrics(X, all_labels, y_true, baseline_labels)
        
        # Load baseline results for comparison
        with open('simulation_baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        baseline_time = baseline_results['runtime_seconds']
        speedup = baseline_time / total_time
        efficiency = speedup / size * 100
        comp_time = total_time - comm_time
        comm_pct = comm_time / total_time * 100
        
        # Print results
        print("\n" + "=" * 60)
        print("MPI RESULTS")
        print("=" * 60)
        print(f"\n  TIMING:")
        print(f"    Total time: {total_time:.2f} seconds")
        print(f"    Computation: {comp_time:.2f} seconds ({100-comm_pct:.1f}%)")
        print(f"    Communication: {comm_time:.2f} seconds ({comm_pct:.1f}%)")
        print(f"    Iterations: {iterations}")
        
        print(f"\n  SCALING:")
        print(f"    Baseline time: {baseline_time:.2f} seconds")
        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Efficiency: {efficiency:.1f}%")
        
        print(f"\n  QUALITY:")
        print(f"    Silhouette: {metrics['silhouette']:.4f}")
        print(f"    ARI vs ground truth: {metrics['ari_vs_truth']:.4f}")
        print(f"    NMI vs ground truth: {metrics['nmi_vs_truth']:.4f}")
        
        print(f"\n  VALIDATION (vs baseline):")
        print(f"    ARI: {metrics['ari_vs_baseline']:.4f}")
        print(f"    NMI: {metrics['nmi_vs_baseline']:.4f}")
        
        # Save results
        results = {
            'n_processes': size,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_clusters': n_clusters,
            'total_time': total_time,
            'comp_time': comp_time,
            'comm_time': comm_time,
            'comm_pct': comm_pct,
            'iterations': iterations,
            'baseline_time': baseline_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'silhouette': metrics['silhouette'],
            'ari_vs_truth': metrics['ari_vs_truth'],
            'nmi_vs_truth': metrics['nmi_vs_truth'],
            'ari_vs_baseline': metrics['ari_vs_baseline'],
            'nmi_vs_baseline': metrics['nmi_vs_baseline'],
            'agreement_pct': metrics['agreement_pct']
        }
        
        filename = f'simulation_mpi_np{size}_results.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {filename}")
        
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()
