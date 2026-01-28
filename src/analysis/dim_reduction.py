"""
Dimensionality Reduction Module
Implements PCA, UMAP, t-SNE, and SVD with validation metrics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

import config


def apply_pca(X: np.ndarray, variance_threshold: float = None, 
             n_components: int = None, output_dir=None):
    """
    Apply PCA with automatic or manual component selection.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        variance_threshold: Keep components explaining this fraction of variance
        n_components: Explicit number of components (overrides variance_threshold)
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (X_pca, pca_model, explained_variance_ratio)
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "dimensionality_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*80)
    
    print(f"\nInput shape: {X.shape}")
    
    # Standardize first
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components first to analyze
    print("\nFitting PCA...")
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    elif variance_threshold is not None:
        pca = PCA(n_components=variance_threshold, random_state=config.RANDOM_SEED)
    else:
        pca = PCA(n_components=config.PCA_VARIANCE_THRESHOLD, random_state=config.RANDOM_SEED)
    
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Components selected: {pca.n_components_}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance per component
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    axes[0].bar(components, pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(components, cumsum, marker='o', linewidth=2, markersize=4)
    axes[1].axhline(0.90, color='red', linestyle='--', label='90%')
    axes[1].axhline(0.95, color='orange', linestyle='--', label='95%')
    axes[1].axhline(0.99, color='green', linestyle='--', label='99%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_scree_plot.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved scree plot to {output_dir / 'pca_scree_plot.png'}")
    
    # Print component statistics
    print("\nTop 10 Components:")
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% " +
              f"(cumulative: {cumsum[i]*100:.2f}%)")
    
    return X_pca, pca, pca.explained_variance_ratio_


def apply_umap(X: np.ndarray, n_components: int = 2, 
              n_neighbors: int = None, min_dist: float = None,
              output_dir=None):
    """
    Apply UMAP for dimensionality reduction or visualization.
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions to reduce to
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        output_dir: Directory to save results
        
    Returns:
        X_umap: Reduced representation
    """
    try:
        import umap
    except ImportError:
        print("\n⚠ WARNING: umap-learn not installed. Skipping UMAP.")
        print("  Install with: pip install umap-learn")
        return None
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "dimensionality_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP)")
    print("="*80)
    
    if n_neighbors is None:
        n_neighbors = config.UMAP_N_NEIGHBORS
    if min_dist is None:
        min_dist = config.UMAP_MIN_DIST
    
    print(f"\nInput shape: {X.shape}")
    print(f"Parameters: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Sample if too large
    sample_size = min(100000, X.shape[0])
    if X.shape[0] > sample_size:
        print(f"Sampling {sample_size:,} points for UMAP...")
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    print("Fitting UMAP...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=config.RANDOM_SEED,
        verbose=True
    )
    
    X_umap = reducer.fit_transform(X_sample)
    
    print(f"Output shape: {X_umap.shape}")
    
    # If 2D, create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.3, s=1, c='steelblue')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'UMAP Projection (n={len(X_sample):,})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_projection.png', dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to {output_dir / 'umap_projection.png'}")
    
    return X_umap


def apply_tsne(X: np.ndarray, n_components: int = 2, 
              perplexity: int = None, n_iter: int = None,
              output_dir=None):
    """
    Apply t-SNE for visualization (typically 2D).
    WARNING: t-SNE is for visualization only, not clustering input.
    
    Args:
        X: Feature matrix
        n_components: Number of dimensions (typically 2 or 3)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        output_dir: Directory to save results
        
    Returns:
        X_tsne: Reduced representation
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "dimensionality_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
    print("="*80)
    
    if perplexity is None:
        perplexity = config.TSNE_PERPLEXITY
    if n_iter is None:
        n_iter = config.TSNE_N_ITER
    
    print(f"\nInput shape: {X.shape}")
    print(f"Parameters: n_components={n_components}, perplexity={perplexity}, n_iter={n_iter}")
    
    # Sample for speed (t-SNE is slow)
    sample_size = min(10000, X.shape[0])
    if X.shape[0] > sample_size:
        print(f"Sampling {sample_size:,} points for t-SNE...")
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    print("Fitting t-SNE (this may take a while)...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=config.RANDOM_SEED,
        verbose=1
    )
    
    X_tsne = tsne.fit_transform(X_sample)
    
    print(f"Output shape: {X_tsne.shape}")
    
    # If 2D, create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5, s=3, c='steelblue')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE Projection (n={len(X_sample):,})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_projection.png', dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to {output_dir / 'tsne_projection.png'}")
    
    return X_tsne


def apply_truncated_svd(X: np.ndarray, n_components: int = None, output_dir=None):
    """
    Apply Truncated SVD (LSA) for sparse matrices or text features.
    
    Args:
        X: Feature matrix (can be sparse)
        n_components: Number of components
        output_dir: Directory to save results
        
    Returns:
        Tuple of (X_svd, svd_model, explained_variance_ratio)
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "dimensionality_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TRUNCATED SINGULAR VALUE DECOMPOSITION (SVD)")
    print("="*80)
    
    print(f"\nInput shape: {X.shape}")
    
    if n_components is None:
        n_components = min(100, X.shape[1] - 1)
    
    print(f"Computing {n_components} components...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    X_svd = svd.fit_transform(X)
    
    print(f"Output shape: {X_svd.shape}")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Plot explained variance
    fig, ax = plt.subplots(figsize=(12, 6))
    cumsum = np.cumsum(svd.explained_variance_ratio_)
    components = np.arange(1, len(cumsum) + 1)
    
    ax.plot(components, cumsum, marker='o', linewidth=2, markersize=3)
    ax.axhline(0.90, color='red', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Truncated SVD: Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'svd_variance.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_dir / 'svd_variance.png'}")
    
    return X_svd, svd, svd.explained_variance_ratio_


def validate_reduction(X_original: np.ndarray, X_reduced: np.ndarray,
                      labels: np.ndarray = None, method_name: str = "Reduction",
                      sample_size: int = 10000):
    """
    Validate dimensionality reduction using multiple metrics.
    
    Args:
        X_original: Original high-dimensional data
        X_reduced: Reduced representation
        labels: Cluster labels (if available)
        method_name: Name of reduction method
        sample_size: Sample size for silhouette computation
    """
    print(f"\n--- Validating {method_name} ---")
    
    # Sample for efficiency
    n = min(sample_size, len(X_reduced))
    indices = np.random.choice(len(X_reduced), n, replace=False)
    
    # 1. Silhouette score comparison (if labels provided)
    if labels is not None:
        from sklearn.metrics import silhouette_score
        
        sil_original = silhouette_score(X_original[indices], labels[indices])
        sil_reduced = silhouette_score(X_reduced[indices], labels[indices])
        
        print(f"Silhouette Score:")
        print(f"  Original space: {sil_original:.4f}")
        print(f"  Reduced space:  {sil_reduced:.4f}")
        print(f"  Change: {sil_reduced - sil_original:+.4f}")
    
    # 2. Reconstruction error (for linear methods)
    if method_name in ['PCA', 'SVD']:
        # For PCA/SVD, we can compute reconstruction error
        print(f"Reconstruction:")
        print(f"  Original dims: {X_original.shape[1]}")
        print(f"  Reduced dims:  {X_reduced.shape[1]}")
        print(f"  Compression:   {100 * X_reduced.shape[1] / X_original.shape[1]:.1f}%")
    
    # 3. Variance retention
    print(f"Statistics:")
    print(f"  Original variance: {np.var(X_original):.4f}")
    print(f"  Reduced variance:  {np.var(X_reduced):.4f}")


def main():
    """
    Run complete dimensionality reduction pipeline.
    """
    import time
    from data_prep import load_processed_data
    
    start_time = time.time()
    
    print("="*80)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading processed data...")
    X, df_meta = load_processed_data()
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        print("Converting sparse matrix to dense...")
        X = X.toarray()
    
    print(f"Loaded data shape: {X.shape}")
    
    # 1. PCA
    X_pca, pca_model, var_ratio = apply_pca(X, variance_threshold=0.95)
    
    # Save PCA-reduced data
    output_path = config.DATA_DIR / "processed_X_pca.npy"
    np.save(output_path, X_pca)
    print(f"\nSaved PCA-reduced data to {output_path}")
    
    # 2. UMAP (sample for speed)
    sample_size = min(50000, X.shape[0])
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[indices]
    
    X_umap = apply_umap(X_sample, n_components=2)
    
    # 3. t-SNE (smaller sample)
    tsne_sample_size = min(10000, X.shape[0])
    tsne_indices = np.random.choice(X.shape[0], tsne_sample_size, replace=False)
    X_tsne_sample = X[tsne_indices]
    
    X_tsne = apply_tsne(X_tsne_sample, n_components=2)
    
    # 4. SVD (if data is sparse)
    # X_svd, svd_model, svd_var = apply_truncated_svd(X, n_components=100)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"DIMENSIONALITY REDUCTION COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
