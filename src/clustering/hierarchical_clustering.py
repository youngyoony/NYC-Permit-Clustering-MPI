"""
Hierarchical Clustering - Single Node Implementation
Implements Ward linkage on stratified sample with dendrogram analysis.

UPDATED: Added --use-pca option for better clustering quality!
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

import config


def load_data_with_pca_option(use_pca: bool = True):
    """
    Load data with optional PCA transformation.
    """
    from data_prep import load_processed_data
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    if use_pca:
        pca_path = config.DATA_DIR / "processed_X_pca.npy"
        if pca_path.exists():
            X = np.load(pca_path)
            print(f"✓ [PCA MODE] Loaded: {pca_path}")
            print(f"  Shape: {X.shape}")
            print(f"  Expected silhouette: ~0.25-0.30")
        else:
            print(f"⚠ WARNING: PCA data not found. Run dim_reduction.py first!")
            print(f"  Falling back to original features...")
            X, _ = load_processed_data()
    else:
        X, _ = load_processed_data()
        print(f"✓ [ORIGINAL MODE] Using feature-selected data")
        print(f"  Shape: {X.shape}")
    
    # Load metadata
    df_meta = pd.read_csv(config.PROCESSED_META_PATH, index_col=0)
    
    # Convert sparse to dense
    if hasattr(X, 'toarray'):
        print("Converting sparse to dense...")
        X = X.toarray()
    
    print(f"Final data shape: {X.shape}")
    return X, df_meta


def stratified_sample(X: np.ndarray, df_meta: pd.DataFrame, 
                     samples_per_borough: int = None, total_samples: int = None):
    """Create stratified sample ensuring representation from each borough."""
    print("\n" + "="*80)
    print("CREATING STRATIFIED SAMPLE")
    print("="*80)
    
    if samples_per_borough is None:
        samples_per_borough = config.HIERARCHICAL_SAMPLE_PER_BOROUGH
    
    if 'Borough' not in df_meta.columns:
        print("WARNING: Borough column not found. Using random sampling.")
        if total_samples is None:
            total_samples = config.HIERARCHICAL_SAMPLE_SIZE
        indices = np.random.choice(len(X), min(total_samples, len(X)), replace=False)
        return X[indices], df_meta.iloc[indices], indices
    
    boroughs = df_meta['Borough'].unique()
    print(f"\nBoroughs found: {len(boroughs)}")
    
    sampled_indices = []
    for borough in boroughs:
        borough_mask = df_meta['Borough'] == borough
        borough_indices = np.where(borough_mask)[0]
        n_samples = min(samples_per_borough, len(borough_indices))
        
        if len(borough_indices) > 0:
            sampled = np.random.choice(borough_indices, n_samples, replace=False)
            sampled_indices.extend(sampled)
            print(f"  {borough}: {n_samples} samples")
    
    sampled_indices = np.array(sampled_indices)
    print(f"\nTotal samples: {len(sampled_indices)}")
    
    return X[sampled_indices], df_meta.iloc[sampled_indices], sampled_indices


def compute_linkage_matrix(X: np.ndarray, method: str = None):
    """Compute hierarchical clustering linkage matrix."""
    if method is None:
        method = config.HIERARCHICAL_LINKAGE
    
    print(f"\n--- Computing {method.upper()} linkage ---")
    print(f"Data shape: {X.shape}")
    print("Computing linkage matrix...")
    
    Z = linkage(X, method=method)
    
    print(f"Linkage matrix shape: {Z.shape}")
    return Z


def plot_dendrogram(Z: np.ndarray, max_clusters: int = 20, output_dir=None):
    """Generate and save dendrogram plot."""
    if output_dir is None:
        output_dir = config.CLUSTERS_DIR / "hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating dendrogram...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=max_clusters,
              leaf_font_size=10, show_contracted=True)
    
    ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram (Truncated)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dendrogram.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dendrogram to {output_dir / 'dendrogram.png'}")


def find_optimal_k(Z: np.ndarray, X: np.ndarray, k_range: range = None, output_dir=None):
    """Find optimal number of clusters using multiple methods."""
    if output_dir is None:
        output_dir = config.CLUSTERS_DIR / "hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if k_range is None:
        k_range = range(2, min(config.HIERARCHICAL_MAX_K, len(X) // 10) + 1)
    
    print("\n" + "="*80)
    print("FINDING OPTIMAL K")
    print("="*80)
    print(f"Testing k in range {k_range.start} to {k_range.stop - 1}")
    
    results = {
        'k': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'last_merge_distance': []
    }
    
    for k in k_range:
        print(f"\nTesting k={k}...")
        
        labels = fcluster(Z, k, criterion='maxclust')
        
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        last_merge_dist = Z[-(k-1), 2] if k > 1 else Z[-1, 2]
        
        results['k'].append(k)
        results['silhouette'].append(sil)
        results['davies_bouldin'].append(db)
        results['calinski_harabasz'].append(ch)
        results['last_merge_distance'].append(last_merge_dist)
        
        print(f"  Silhouette: {sil:.4f}")
        print(f"  Davies-Bouldin: {db:.4f}")
        print(f"  Calinski-Harabasz: {ch:.2f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'optimal_k_metrics.csv', index=False)
    print(f"\nSaved metrics to {output_dir / 'optimal_k_metrics.csv'}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(results['k'], results['silhouette'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score (higher is better)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(results['k'], results['davies_bouldin'], marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Davies-Bouldin Index')
    axes[0, 1].set_title('Davies-Bouldin Index (lower is better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results['k'], results['calinski_harabasz'], marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (higher is better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results['k'], results['last_merge_distance'], marker='o', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Merge Distance')
    axes[1, 1].set_title('Linkage Distance (elbow method)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_k_analysis.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    optimal_k = {
        'silhouette': results['k'][np.argmax(results['silhouette'])],
        'davies_bouldin': results['k'][np.argmin(results['davies_bouldin'])],
        'calinski_harabasz': results['k'][np.argmax(results['calinski_harabasz'])]
    }
    
    print("\n--- Optimal k by method ---")
    for method, k_opt in optimal_k.items():
        print(f"  {method}: k={k_opt}")
    
    return optimal_k, results_df


def analyze_clusters(X: np.ndarray, labels: np.ndarray, df_meta: pd.DataFrame = None):
    """Analyze cluster properties."""
    print("\n" + "="*80)
    print("ANALYZING CLUSTER PROPERTIES")
    print("="*80)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    print(f"Number of clusters: {n_clusters}")
    
    cluster_stats = []
    
    for label in unique_labels:
        mask = labels == label
        X_cluster = X[mask]
        
        size = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        
        distances_to_centroid = np.linalg.norm(X_cluster - centroid, axis=1)
        radius = np.mean(distances_to_centroid)
        max_radius = np.max(distances_to_centroid)
        
        if len(X_cluster) > 100:
            sample_indices = np.random.choice(len(X_cluster), 100, replace=False)
            X_sample = X_cluster[sample_indices]
        else:
            X_sample = X_cluster
        
        pairwise_dists = pdist(X_sample)
        diameter = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 0
        
        stats = {
            'cluster': label,
            'size': size,
            'radius_mean': radius,
            'radius_max': max_radius,
            'diameter': diameter
        }
        
        if df_meta is not None and 'Borough' in df_meta.columns:
            cluster_meta = df_meta.iloc[mask]
            if len(cluster_meta) > 0 and 'Borough' in cluster_meta.columns:
                dominant_borough = cluster_meta['Borough'].mode()[0]
                borough_pct = 100 * (cluster_meta['Borough'] == dominant_borough).sum() / len(cluster_meta)
                stats['dominant_borough'] = dominant_borough
                stats['borough_pct'] = borough_pct
        
        cluster_stats.append(stats)
        
        print(f"\nCluster {label}:")
        print(f"  Size: {size}")
        print(f"  Radius (mean): {radius:.4f}")
        if 'dominant_borough' in stats:
            print(f"  Dominant borough: {stats['dominant_borough']} ({stats['borough_pct']:.1f}%)")
    
    return pd.DataFrame(cluster_stats)


def main():
    """Run complete hierarchical clustering pipeline."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Hierarchical Clustering Baseline')
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA-transformed data (default: True)')
    parser.add_argument('--no-pca', action='store_false', dest='use_pca',
                       help='Use original feature-selected data')
    parser.add_argument('--k', type=int, default=None, help='Number of clusters (auto if not specified)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("HIERARCHICAL CLUSTERING (SINGLE NODE)")
    print("="*80)
    
    if args.use_pca:
        print("\n*** PCA MODE: Using PCA-transformed data ***")
    else:
        print("\n*** ORIGINAL MODE: Using feature-selected data ***")
    
    # Load data with PCA option
    X, df_meta = load_data_with_pca_option(use_pca=args.use_pca)
    
    # Create stratified sample
    X_sample, df_meta_sample, sample_indices = stratified_sample(
        X, df_meta, samples_per_borough=config.HIERARCHICAL_SAMPLE_PER_BOROUGH
    )
    
    # Compute linkage
    Z = compute_linkage_matrix(X_sample, method=config.HIERARCHICAL_LINKAGE)
    
    # Plot dendrogram
    plot_dendrogram(Z)
    
    # Find optimal k
    optimal_k, metrics_df = find_optimal_k(Z, X_sample, k_range=range(2, 21))
    
    # Use specified k or silhouette-based optimal
    k_final = args.k if args.k else optimal_k['silhouette']
    
    print(f"\n{'='*80}")
    print(f"FINAL CLUSTERING WITH k={k_final}")
    print(f"{'='*80}")
    
    labels = fcluster(Z, k_final, criterion='maxclust')
    
    # Analyze clusters
    cluster_stats = analyze_clusters(X_sample, labels, df_meta_sample)
    
    # Save results
    output_dir = config.CLUSTERS_DIR / "hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_stats.to_csv(output_dir / f'cluster_stats_k{k_final}.csv', index=False)
    np.save(output_dir / f'labels_k{k_final}.npy', labels)
    
    print(f"\nSaved results to {output_dir}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL CLUSTERING COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()