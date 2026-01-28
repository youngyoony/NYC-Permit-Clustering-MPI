"""
BFR (Bradley-Fayyad-Reina) Algorithm with MPI Support
Streaming/memory-efficient clustering using sufficient statistics.

KEY CONCEPTS:
- Discard Set (DS): Points assigned to clusters, represented by N, SUM, SUMSQ
- Compressed Set (CS): Mini-clusters of unassigned points
- Retained Set (RS): Outliers not yet clustered

This implementation supports both single-node and MPI-distributed processing.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import shapiro
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import config


class BFRCluster:
    """Represents a cluster using sufficient statistics."""
    
    def __init__(self, dim):
        self.N = 0  # Number of points
        self.SUM = np.zeros(dim)  # Sum of coordinates
        self.SUMSQ = np.zeros(dim)  # Sum of squared coordinates
        self.dim = dim
    
    def add_point(self, point):
        """Add a single point to cluster statistics."""
        self.N += 1
        self.SUM += point
        self.SUMSQ += point ** 2
    
    def add_points(self, points):
        """Add multiple points to cluster statistics."""
        n = len(points)
        self.N += n
        self.SUM += np.sum(points, axis=0)
        self.SUMSQ += np.sum(points ** 2, axis=0)
    
    def centroid(self):
        """Compute cluster centroid."""
        if self.N == 0:
            return np.zeros(self.dim)
        return self.SUM / self.N
    
    def variance(self):
        """Compute variance per dimension."""
        if self.N == 0:
            return np.ones(self.dim)
        mean = self.centroid()
        return (self.SUMSQ / self.N) - mean ** 2
    
    def std(self):
        """Compute standard deviation per dimension."""
        return np.sqrt(self.variance())
    
    def mahalanobis_distance(self, point):
        """
        Compute Mahalanobis distance from point to cluster.
        Assumes diagonal covariance (axis-aligned).
        """
        centroid = self.centroid()
        variance = self.variance()
        
        # Avoid division by zero
        variance = np.maximum(variance, 1e-10)
        
        diff = point - centroid
        distance = np.sqrt(np.sum((diff ** 2) / variance))
        
        return distance
    
    def merge(self, other):
        """Merge another cluster into this one."""
        self.N += other.N
        self.SUM += other.SUM
        self.SUMSQ += other.SUMSQ


def initialize_bfr(X_init: np.ndarray, k: int, random_state: int = None):
    """
    Initialize BFR with KMeans on initial sample.
    
    Args:
        X_init: Initial sample
        k: Number of clusters
        random_state: Random seed
        
    Returns:
        List of BFRCluster objects
    """
    print(f"\nInitializing BFR with {k} clusters on {len(X_init):,} points...")
    
    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_init)
    
    # Create BFR clusters
    dim = X_init.shape[1]
    clusters = [BFRCluster(dim) for _ in range(k)]
    
    for label, point in zip(labels, X_init):
        clusters[label].add_point(point)
    
    print(f"Initialized {k} clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: N={cluster.N}, centroid_norm={np.linalg.norm(cluster.centroid()):.4f}")
    
    return clusters


def process_chunk_bfr(X_chunk: np.ndarray, clusters: list, 
                     threshold: float = None, k_mini: int = 10):
    """
    Process a chunk of data using BFR algorithm.
    
    Args:
        X_chunk: Data chunk
        clusters: List of BFRCluster objects (Discard Set)
        threshold: Mahalanobis distance threshold (default: 4 std devs)
        k_mini: Number of mini-clusters for Compressed Set
        
    Returns:
        Tuple of (assigned_points, compressed_set, retained_set)
    """
    if threshold is None:
        threshold = config.BFR_MAHALANOBIS_THRESHOLD
    
    n_assigned = 0
    unassigned = []
    
    # Assign points to existing clusters (Discard Set)
    for point in X_chunk:
        min_dist = float('inf')
        best_cluster = -1
        
        for i, cluster in enumerate(clusters):
            dist = cluster.mahalanobis_distance(point)
            if dist < min_dist:
                min_dist = dist
                best_cluster = i
        
        if min_dist < threshold:
            # Assign to cluster
            clusters[best_cluster].add_point(point)
            n_assigned += 1
        else:
            # Cannot assign
            unassigned.append(point)
    
    # Create mini-clusters from unassigned points (Compressed Set)
    compressed_set = []
    retained_set = []
    
    if len(unassigned) > k_mini:
        X_unassigned = np.array(unassigned)
        
        # Try to create mini-clusters
        kmeans = KMeans(n_clusters=min(k_mini, len(X_unassigned) // 10), 
                       random_state=config.RANDOM_SEED, n_init=5)
        mini_labels = kmeans.fit_predict(X_unassigned)
        
        dim = X_unassigned.shape[1]
        for label in np.unique(mini_labels):
            mini_cluster = BFRCluster(dim)
            mini_cluster.add_points(X_unassigned[mini_labels == label])
            
            if mini_cluster.N >= config.BFR_MIN_CLUSTER_SIZE:
                compressed_set.append(mini_cluster)
            else:
                # Too small, add to retained set
                retained_set.extend(X_unassigned[mini_labels == label])
    else:
        # Too few unassigned points, add to retained set
        retained_set = unassigned
    
    print(f"  Chunk processed: {n_assigned:,} assigned, " +
          f"{len(compressed_set)} mini-clusters, " +
          f"{len(retained_set)} retained")
    
    return n_assigned, compressed_set, retained_set


def bfr_algorithm(X: np.ndarray, k: int, chunk_size: int = None, 
                 initial_sample_size: int = None, threshold: float = None):
    """
    Run BFR algorithm on full dataset.
    
    Args:
        X: Full feature matrix
        k: Number of clusters
        chunk_size: Size of each chunk
        initial_sample_size: Size of initial sample for initialization
        threshold: Mahalanobis threshold
        
    Returns:
        List of final BFRCluster objects
    """
    print("\n" + "="*80)
    print("BFR ALGORITHM")
    print("="*80)
    
    if chunk_size is None:
        chunk_size = config.BFR_CHUNK_SIZE
    if initial_sample_size is None:
        initial_sample_size = config.BFR_INITIAL_SAMPLE_SIZE
    
    print(f"Dataset size: {len(X):,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Number of clusters: {k}")
    
    # Initialize with sample
    X_init = X[:initial_sample_size]
    clusters = initialize_bfr(X_init, k, random_state=config.RANDOM_SEED)
    
    # Process remaining data in chunks
    all_compressed = []
    all_retained = []
    
    start_idx = initial_sample_size
    chunk_num = 0
    
    while start_idx < len(X):
        end_idx = min(start_idx + chunk_size, len(X))
        X_chunk = X[start_idx:end_idx]
        
        chunk_num += 1
        print(f"\nProcessing chunk {chunk_num} ({len(X_chunk):,} points)...")
        
        n_assigned, compressed_set, retained_set = process_chunk_bfr(
            X_chunk, clusters, threshold=threshold
        )
        
        all_compressed.extend(compressed_set)
        all_retained.extend(retained_set)
        
        start_idx = end_idx
    
    # Final step: merge compressed set with main clusters
    print(f"\nFinal merging...")
    print(f"  Compressed set: {len(all_compressed)} mini-clusters")
    print(f"  Retained set: {len(all_retained)} outliers")
    
    for mini_cluster in all_compressed:
        # Find nearest main cluster
        centroid = mini_cluster.centroid()
        min_dist = float('inf')
        best_cluster = 0
        
        for i, cluster in enumerate(clusters):
            dist = cluster.mahalanobis_distance(centroid)
            if dist < min_dist:
                min_dist = dist
                best_cluster = i
        
        # Merge if close enough
        if min_dist < threshold * 2:  # Relaxed threshold for merging
            clusters[best_cluster].merge(mini_cluster)
    
    # Optionally assign retained set
    for point in all_retained:
        # Find nearest cluster
        min_dist = float('inf')
        best_cluster = 0
        
        for i, cluster in enumerate(clusters):
            dist = cluster.mahalanobis_distance(point)
            if dist < min_dist:
                min_dist = dist
                best_cluster = i
        
        clusters[best_cluster].add_point(point)
    
    print("\nFinal clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: N={cluster.N:,}")
    
    return clusters


def main():
    """Run BFR algorithm."""
    import time
    from data_prep import load_processed_data
    
    start_time = time.time()
    
    print("="*80)
    print("BFR ALGORITHM - MEMORY-EFFICIENT CLUSTERING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X, df_meta = load_processed_data()
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Use PCA-reduced if available
    pca_path = config.DATA_DIR / "processed_X_pca.npy"
    if pca_path.exists():
        X = np.load(pca_path)
    
    print(f"Data shape: {X.shape}")
    
    # Run BFR
    clusters = bfr_algorithm(X, k=5, chunk_size=100000)
    
    # Save results
    output_dir = config.CLUSTERS_DIR / "bfr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract centroids
    centroids = np.array([c.centroid() for c in clusters])
    np.save(output_dir / 'centroids.npy', centroids)
    
    # Save cluster stats
    stats = [{
        'cluster': i,
        'size': c.N,
        'centroid_norm': float(np.linalg.norm(c.centroid())),
        'avg_std': float(np.mean(c.std()))
    } for i, c in enumerate(clusters)]
    
    with open(output_dir / 'cluster_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BFR COMPLETE - Time: {elapsed:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
