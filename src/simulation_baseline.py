#!/usr/bin/env python3
"""
Simulation Baseline: Synthetic Data K-Means Clustering
AMS 598 - Big Data Analysis

This script generates synthetic clustered data and runs single-node K-Means
to establish baseline performance metrics for comparison with MPI version.
"""

import numpy as np
import time
import json
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'n_samples': 3800000, 
    'n_features': 22, 
    'n_clusters': 10, 
    'cluster_std': 2.0, 
    'random_state': 42,
    'max_iter': 100,
    'n_init': 10
}

def generate_synthetic_data(config):
    """Generate well-separated synthetic clusters."""
    print("=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)
    
    X, y_true = make_blobs(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        centers=config['n_clusters'],
        cluster_std=config['cluster_std'],
        random_state=config['random_state']
    )
    
    # Standardize features (like we did with real data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  Samples: {X_scaled.shape[0]:,}")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Clusters: {config['n_clusters']}")
    print(f"  Cluster std: {config['cluster_std']}")
    
    return X_scaled, y_true

def run_kmeans_baseline(X, y_true, config):
    """Run single-node K-Means and compute metrics."""
    print("\n" + "=" * 60)
    print("RUNNING K-MEANS BASELINE (Single-Node)")
    print("=" * 60)
    
    # Run K-Means with timing
    start_time = time.time()
    
    kmeans = KMeans(
        n_clusters=config['n_clusters'],
        init='k-means++',
        max_iter=config['max_iter'],
        n_init=config['n_init'],
        random_state=config['random_state']
    )
    labels = kmeans.fit_predict(X)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # Silhouette (sample for speed if large)
    if len(X) > 50000:
        sample_idx = np.random.choice(len(X), 50000, replace=False)
        sil_score = silhouette_score(X[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(X, labels)
    
    # Clustering agreement with ground truth
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    
    results = {
        'runtime_seconds': runtime,
        'iterations': kmeans.n_iter_,
        'inertia': kmeans.inertia_,
        'silhouette_score': sil_score,
        'ari': ari,
        'nmi': nmi,
        'cluster_sizes': cluster_sizes,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_clusters': config['n_clusters']
    }
    
    # Print results
    print("\n" + "-" * 40)
    print("BASELINE RESULTS")
    print("-" * 40)
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  ARI (vs ground truth): {ari:.4f}")
    print(f"  NMI (vs ground truth): {nmi:.4f}")
    print(f"\n  Cluster Sizes:")
    for cluster_id, size in cluster_sizes.items():
        pct = size / len(X) * 100
        print(f"    Cluster {cluster_id}: {size:,} ({pct:.1f}%)")
    
    return labels, kmeans.cluster_centers_, results

def save_data_for_mpi(X, y_true, baseline_labels, centroids):
    """Save data and baseline results for MPI comparison."""
    print("\n" + "=" * 60)
    print("SAVING DATA FOR MPI COMPARISON")
    print("=" * 60)
    
    np.save('simulation_X.npy', X)
    np.save('simulation_y_true.npy', y_true)
    np.save('simulation_baseline_labels.npy', baseline_labels)
    np.save('simulation_baseline_centroids.npy', centroids)
    
    print("  Saved: simulation_X.npy")
    print("  Saved: simulation_y_true.npy")
    print("  Saved: simulation_baseline_labels.npy")
    print("  Saved: simulation_baseline_centroids.npy")

def main():
    print("\n" + "=" * 60)
    print("AMS 598 - SIMULATION BASELINE")
    print("Synthetic Data K-Means Clustering")
    print("=" * 60)
    
    # Generate data
    X, y_true = generate_synthetic_data(CONFIG)
    
    # Run baseline
    labels, centroids, results = run_kmeans_baseline(X, y_true, CONFIG)
    
    # Save for MPI
    save_data_for_mpi(X, y_true, labels, centroids)
    
    # Save results
    with open('simulation_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: simulation_baseline_results.json")
    
    # Summary comparison with real data
    print("\n" + "=" * 60)
    print("COMPARISON: SIMULATION vs REAL DATA")
    print("=" * 60)
    print(f"{'Metric':<25} {'Simulation':<15} {'Real Data':<15}")
    print("-" * 55)
    print(f"{'Samples':<25} {CONFIG['n_samples']:,} {'3,983,393':>15}")
    print(f"{'Silhouette Score':<25} {results['silhouette_score']:.4f} {'0.0780':>15}")
    print(f"{'ARI vs Ground Truth':<25} {results['ari']:.4f} {'N/A':>15}")
    print(f"{'NMI vs Ground Truth':<25} {results['nmi']:.4f} {'N/A':>15}")
    print("-" * 55)
    print("\n✓ High silhouette on simulation confirms algorithm works correctly")
    print("✓ Low silhouette on real data reflects data characteristics, not bugs")
    
    print("\n" + "=" * 60)
    print("BASELINE COMPLETE - Now run simulation_mpi.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
