"""
Visualization Module
Generate visualizations for clustering results: t-SNE/UMAP plots,
geographic maps, cluster profiles.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

import config


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, 
                    title: str = "Cluster Visualization",
                    method: str = "t-SNE", output_path=None):
    """
    Plot 2D cluster visualization (t-SNE or UMAP).
    
    Args:
        X_2d: 2D coordinates
        labels: Cluster labels
        title: Plot title
        method: Dimensionality reduction method name
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[colors[i]], label=f'Cluster {label}',
                  alpha=0.7, s=15, edgecolors='none')
    
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(title)
    ax.legend(loc='best', markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.close()


def plot_geographic_clusters(df_meta: pd.DataFrame, labels: np.ndarray,
                            output_path=None):
    """
    Plot clusters on geographic map (lat/lon).
    
    Args:
        df_meta: Metadata with Latitude, Longitude
        labels: Cluster labels
        output_path: Path to save plot
    """
    if 'Latitude' not in df_meta.columns or 'Longitude' not in df_meta.columns:
        print("Warning: Latitude/Longitude not in metadata")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(df_meta.loc[mask, 'Longitude'], 
                  df_meta.loc[mask, 'Latitude'],
                  c=[colors[i]], label=f'Cluster {label}',
                  alpha=0.3, s=1)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Clusters - Geographic Distribution')
    ax.legend(loc='best', markerscale=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved geographic plot to {output_path}")
    
    plt.close()


def visualize_algorithm_results(algorithm: str = 'kmeans', k: int = 10,
                                nprocs: int = None):
    """
    Generate comprehensive visualizations for algorithm results.
    
    Args:
        algorithm: 'kmeans', 'kmeans_mpi', 'hierarchical', etc.
        k: Number of clusters
        nprocs: Number of processes (for MPI algorithms)
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZING: {algorithm.upper()} (k={k})")
    print(f"{'='*80}")
    
    # Determine paths
    if 'mpi' in algorithm:
        cluster_dir = config.CLUSTERS_DIR / algorithm
        suffix = f'_k{k}_np{nprocs}' if nprocs else f'_k{k}'
    else:
        cluster_dir = config.CLUSTERS_DIR / algorithm
        suffix = f'_k{k}'
    
    # Load labels
    labels_path = cluster_dir / f'labels{suffix}.npy'
    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        return
    
    labels = np.load(labels_path)
    print(f"Loaded labels: {len(labels):,} points, {len(np.unique(labels))} clusters")
    
    # Load data
    from data_prep import load_processed_data
    X, df_meta = load_processed_data()
    
    # Use PCA-reduced if available
    pca_path = config.DATA_DIR / "processed_X_pca.npy"
    if pca_path.exists():
        X = np.load(pca_path)
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Match X to labels size (labels might be from a sample)
    n_labels = len(labels)
    if len(X) > n_labels:
        print(f"Warning: X has {len(X)} samples but labels has {n_labels} samples")
        print(f"Using first {n_labels} samples from X to match labels")
        X = X[:n_labels]
        df_meta = df_meta.iloc[:n_labels]
    elif len(X) < n_labels:
        print(f"Warning: X has {len(X)} samples but labels has {n_labels} samples")
        print(f"Trimming labels to match X")
        labels = labels[:len(X)]
        n_labels = len(labels)
    
    # Sample for visualization (balanced across clusters)
    sample_size = min(10000, len(labels))  # 10k points for better visualization
    print(f"Sampling {sample_size} points for visualization...")
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.choice(len(labels), sample_size, replace=False)
    X_sample = X[indices]
    labels_sample = labels[indices]
    df_meta_sample = df_meta.iloc[indices]
    
    output_dir = config.VISUALIZATIONS_DIR / algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    print(f"Input shape for t-SNE: {X_sample.shape}")
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, 
               perplexity=min(30, sample_size-1), max_iter=500, verbose=1)
    X_tsne = tsne.fit_transform(X_sample)
    
    plot_clusters_2d(X_tsne, labels_sample, 
                    title=f'{algorithm.upper()} Clusters (t-SNE, k={k})',
                    method='t-SNE',
                    output_path=output_dir / f'tsne{suffix}.png')
    
    # 2. Geographic visualization
    print("\nGenerating geographic visualization...")
    plot_geographic_clusters(df_meta_sample, labels_sample,
                            output_path=output_dir / f'geographic{suffix}.png')
    
    print(f"\nVisualization complete! Saved to {output_dir}")


def main():
    """Generate visualizations for all completed algorithms."""
    import time
    
    start_time = time.time()
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # FIXED: Changed k=5 to k=10 to match actual clustering results
    
    # Visualize K-Means baseline
    print("\n[1/3] K-Means Baseline...")
    visualize_algorithm_results('kmeans', k=10)
    
    # Visualize K-Means MPI with np=1 (baseline comparison)
    print("\n[2/3] K-Means MPI (np=1)...")
    visualize_algorithm_results('kmeans_mpi', k=10, nprocs=1)
    
    # Visualize Hierarchical MPI
    print("\n[3/3] Hierarchical MPI (np=16)...")
    visualize_algorithm_results('hierarchical_mpi', k=10, nprocs=16)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE - Time: {elapsed:.2f}s")
    print(f"Results saved to: {config.VISUALIZATIONS_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
