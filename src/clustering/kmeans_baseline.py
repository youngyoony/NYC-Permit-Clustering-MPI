"""
K-Means Clustering - Single Node Baseline Implementation
Standard sklearn K-Means with various k selection strategies.

UPDATED: Added --use-pca option for better clustering quality!
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time
import warnings
warnings.filterwarnings('ignore')

import config


def load_data_with_pca_option(use_pca: bool = True, max_samples: int = 500000):
    """
    Load data with optional PCA transformation.
    
    Args:
        use_pca: If True, use PCA-transformed data
        max_samples: Maximum samples to load (for memory)
        
    Returns:
        X: Feature matrix
        df_meta: Metadata DataFrame
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
            print(f"  Expected silhouette: ~0.25")
        else:
            print(f"⚠ WARNING: PCA data not found. Run dim_reduction.py first!")
            print(f"  Falling back to original features...")
            X, _ = load_processed_data()
    else:
        X, _ = load_processed_data()
        print(f"✓ [ORIGINAL MODE] Using feature-selected data")
        print(f"  Shape: {X.shape}")
        print(f"  Note: Lower silhouette (~0.08) but interpretable")
    
    # Load metadata
    df_meta = pd.read_csv(config.PROCESSED_META_PATH, index_col=0)
    
    # Convert sparse to dense
    if hasattr(X, 'toarray'):
        print("Converting sparse to dense...")
        X = X.toarray()
    
    # Sample if needed
    if X.shape[0] > max_samples:
        print(f"Sampling {max_samples:,} rows...")
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[sample_idx]
        df_meta = df_meta.iloc[sample_idx].copy()
    
    print(f"Final data shape: {X.shape}")
    return X, df_meta


def find_optimal_k_elbow(X: np.ndarray, k_range=None, output_dir=None):
    """
    Find optimal k using elbow method (inertia).
    """
    if output_dir is None:
        output_dir = config.CLUSTERS_DIR / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if k_range is None:
        k_range = config.KMEANS_K_RANGE
    
    print("\n" + "="*80)
    print("ELBOW METHOD FOR OPTIMAL K")
    print("="*80)
    print(f"Testing k values: {k_range}")
    
    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'time': []
    }
    
    for k in k_range:
        print(f"\nTesting k={k}...")
        start = time.time()
        
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=config.RANDOM_SEED,
            verbose=0
        )
        
        labels = kmeans.fit_predict(X)
        elapsed = time.time() - start
        
        # Compute metrics
        sample_size = min(config.SILHOUETTE_SAMPLE_SIZE, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        sil = silhouette_score(X[sample_indices], labels[sample_indices])
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(sil)
        results['davies_bouldin'].append(db)
        results['calinski_harabasz'].append(ch)
        results['time'].append(elapsed)
        
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Silhouette: {sil:.4f}")
        print(f"  Davies-Bouldin: {db:.4f}")
        print(f"  Calinski-Harabasz: {ch:.2f}")
        print(f"  Time: {elapsed:.2f}s")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'elbow_analysis.csv', index=False)
    print(f"\nSaved metrics to {output_dir / 'elbow_analysis.csv'}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(results['k'], results['inertia'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method: Inertia')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(results['k'], results['silhouette'], marker='o', linewidth=2, 
                   markersize=8, color='green')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (higher is better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results['k'], results['davies_bouldin'], marker='o', linewidth=2,
                   markersize=8, color='orange')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results['k'], results['calinski_harabasz'], marker='o', linewidth=2,
                   markersize=8, color='red')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Calinski-Harabasz Score')
    axes[1, 1].set_title('Calinski-Harabasz Score (higher is better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elbow_analysis.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_dir / 'elbow_analysis.png'}")
    
    optimal_k = {
        'silhouette': results['k'][np.argmax(results['silhouette'])],
        'davies_bouldin': results['k'][np.argmin(results['davies_bouldin'])],
        'calinski_harabasz': results['k'][np.argmax(results['calinski_harabasz'])]
    }
    
    print("\n--- Optimal k by method ---")
    for method, k_opt in optimal_k.items():
        print(f"  {method}: k={k_opt}")
    
    return optimal_k, results_df


def run_kmeans(X: np.ndarray, k: int, init_method: str = 'k-means++'):
    """Run K-Means clustering."""
    print(f"\n--- Running K-Means with k={k}, init={init_method} ---")
    print(f"Data shape: {X.shape}")
    
    start = time.time()
    
    kmeans = KMeans(
        n_clusters=k,
        init=init_method,
        n_init=config.KMEANS_N_INIT,
        max_iter=config.KMEANS_MAX_ITER,
        tol=config.KMEANS_TOL,
        random_state=config.RANDOM_SEED,
        verbose=1
    )
    
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start
    
    # Compute metrics
    print("\nComputing quality metrics...")
    sample_size = min(config.SILHOUETTE_SAMPLE_SIZE, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    
    silhouette = silhouette_score(X[sample_indices], labels[sample_indices])
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    metrics = {
        'k': k,
        'init_method': init_method,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'n_iter': kmeans.n_iter_,
        'time': elapsed
    }
    
    print(f"\nResults:")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz: {calinski_harabasz:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Time: {elapsed:.2f}s")
    
    return kmeans, labels, metrics


def main():
    """Run complete K-Means baseline pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Means Baseline Clustering')
    parser.add_argument('--use-pca', action='store_true', default=True,
                       help='Use PCA-transformed data (default: True)')
    parser.add_argument('--no-pca', action='store_false', dest='use_pca',
                       help='Use original feature-selected data')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("K-MEANS CLUSTERING (SINGLE NODE BASELINE)")
    print("="*80)
    
    if args.use_pca:
        print("\n*** PCA MODE: Using PCA-transformed data ***")
    else:
        print("\n*** ORIGINAL MODE: Using feature-selected data ***")
    
    # Load data with PCA option
    X, df_meta = load_data_with_pca_option(use_pca=args.use_pca)
    
    # Find optimal k
    optimal_k, elbow_df = find_optimal_k_elbow(X, k_range=config.KMEANS_K_RANGE)
    
    # Run K-Means with specified k
    k = args.k
    print(f"\n{'='*80}")
    print(f"RUNNING K-MEANS WITH k={k}")
    print(f"{'='*80}")
    
    kmeans_model, labels, metrics = run_kmeans(X, k=k)
    
    # Save results
    output_dir = config.CLUSTERS_DIR / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / f'labels_k{k}.npy', labels)
    np.save(output_dir / f'centroids_k{k}.npy', kmeans_model.cluster_centers_)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'all_metrics.csv', index=False)
    
    print(f"\nSaved results to {output_dir}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"K-MEANS BASELINE COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()