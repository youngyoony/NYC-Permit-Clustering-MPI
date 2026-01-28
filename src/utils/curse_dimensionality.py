"""
Curse of Dimensionality Analysis
Demonstrate distance concentration and KMeans degradation in high dimensions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

import config


def analyze_distance_distribution(X: np.ndarray, sample_size: int = 10000, 
                                 output_dir=None) -> None:
    """
    Analyze pairwise distance distributions in high-dimensional space.
    Demonstrates that distances become increasingly similar (concentrated).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        sample_size: Number of points to sample for analysis
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "curse_of_dimensionality"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ANALYZING DISTANCE DISTRIBUTIONS IN HIGH DIMENSIONS")
    print("="*80)
    
    # Sample points
    n_samples = min(sample_size, X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[indices]
    
    print(f"\nSample size: {n_samples:,}")
    print(f"Dimensions: {X_sample.shape[1]}")
    
    # Compute pairwise distances
    print("\nComputing pairwise distances...")
    distances = euclidean_distances(X_sample)
    
    # Get upper triangle (exclude diagonal and duplicates)
    dist_upper = distances[np.triu_indices_from(distances, k=1)]
    
    print(f"Number of pairwise distances: {len(dist_upper):,}")
    
    # Statistics
    dist_mean = np.mean(dist_upper)
    dist_std = np.std(dist_upper)
    dist_min = np.min(dist_upper)
    dist_max = np.max(dist_upper)
    dist_cv = dist_std / dist_mean  # Coefficient of variation
    
    print(f"\nDistance Statistics:")
    print(f"  Mean:    {dist_mean:.4f}")
    print(f"  Std Dev: {dist_std:.4f}")
    print(f"  Min:     {dist_min:.4f}")
    print(f"  Max:     {dist_max:.4f}")
    print(f"  CV (std/mean): {dist_cv:.4f}")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(dist_upper, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(dist_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={dist_mean:.2f}')
    axes[0].axvline(dist_mean - dist_std, color='orange', linestyle='--', linewidth=1, label=f'±1 std')
    axes[0].axvline(dist_mean + dist_std, color='orange', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Euclidean Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distance Distribution (d={X_sample.shape[1]})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(dist_upper, vert=True)
    axes[1].set_ylabel('Euclidean Distance')
    axes[1].set_title('Distance Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_distribution.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plot to {output_dir / 'distance_distribution.png'}")
    
    # Curse of dimensionality insight
    print("\n" + "-"*80)
    print("CURSE OF DIMENSIONALITY INSIGHT:")
    print("-"*80)
    if dist_cv < 0.1:
        print(f"⚠ WARNING: Low CV ({dist_cv:.4f}) indicates distance concentration!")
        print("  → Pairwise distances are becoming similar")
        print("  → Nearest and farthest neighbors converge")
        print("  → Distance-based methods (KMeans, KNN) become less effective")
    else:
        print(f"✓ Moderate CV ({dist_cv:.4f}) - distances still discriminative")
    print("-"*80)


def compare_kmeans_dimensions(X: np.ndarray, k_values=[5], 
                             sample_size: int = 50000, output_dir=None) -> None:
    """
    Compare KMeans performance in full-dimensional vs reduced-dimensional space.
    
    Args:
        X: Feature matrix
        k_values: List of k values to test
        sample_size: Sample size for analysis
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "curse_of_dimensionality"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPARING K-MEANS IN HIGH VS LOW DIMENSIONS")
    print("="*80)
    
    # Sample data
    n_samples = min(sample_size, X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[indices]
    
    print(f"\nSample size: {n_samples:,}")
    print(f"Full dimensions: {X_sample.shape[1]}")
    
    results = []
    
    for k in k_values:
        print(f"\n--- Testing k={k} ---")
        
        # Full-dimensional KMeans
        print("Running KMeans on full-dimensional data...")
        kmeans_full = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, 
                            n_init=5, max_iter=100)
        labels_full = kmeans_full.fit_predict(X_sample)
        
        # Compute silhouette score (sample for speed)
        sil_sample_size = min(10000, n_samples)
        sil_indices = np.random.choice(n_samples, sil_sample_size, replace=False)
        
        silhouette_full = silhouette_score(X_sample[sil_indices], labels_full[sil_indices])
        inertia_full = kmeans_full.inertia_
        
        print(f"  Full-dimensional:")
        print(f"    Silhouette: {silhouette_full:.4f}")
        print(f"    Inertia: {inertia_full:.2f}")
        print(f"    Iterations: {kmeans_full.n_iter_}")
        
        # PCA reduction
        from sklearn.decomposition import PCA
        
        # Reduce to dimensions that explain 95% variance
        pca = PCA(n_components=0.95, random_state=config.RANDOM_SEED)
        X_pca = pca.fit_transform(X_sample)
        n_components = X_pca.shape[1]
        
        print(f"\nPCA reduction: {X_sample.shape[1]} → {n_components} dims " +
              f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
        
        # KMeans on PCA-reduced data
        print("Running KMeans on PCA-reduced data...")
        kmeans_pca = KMeans(n_clusters=k, random_state=config.RANDOM_SEED,
                           n_init=5, max_iter=100)
        labels_pca = kmeans_pca.fit_predict(X_pca)
        
        silhouette_pca = silhouette_score(X_pca[sil_indices], labels_pca[sil_indices])
        inertia_pca = kmeans_pca.inertia_
        
        print(f"  PCA-reduced:")
        print(f"    Silhouette: {silhouette_pca:.4f}")
        print(f"    Inertia: {inertia_pca:.2f}")
        print(f"    Iterations: {kmeans_pca.n_iter_}")
        
        # Compare
        silhouette_improvement = ((silhouette_pca - silhouette_full) / 
                                 abs(silhouette_full) * 100)
        
        print(f"\n  Improvement: {silhouette_improvement:+.1f}% silhouette score")
        
        results.append({
            'k': k,
            'full_dims': X_sample.shape[1],
            'pca_dims': n_components,
            'full_silhouette': silhouette_full,
            'pca_silhouette': silhouette_pca,
            'full_inertia': inertia_full,
            'pca_inertia': inertia_pca,
            'improvement_pct': silhouette_improvement
        })
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'kmeans_dimension_comparison.csv', index=False)
    
    print(f"\nSaved results to {output_dir / 'kmeans_dimension_comparison.csv'}")
    
    # Visualization
    if len(k_values) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Silhouette comparison
        x = np.arange(len(k_values))
        width = 0.35
        
        axes[0].bar(x - width/2, results_df['full_silhouette'], width, 
                   label='Full Dimensions', alpha=0.8, color='coral')
        axes[0].bar(x + width/2, results_df['pca_silhouette'], width,
                   label='PCA Reduced', alpha=0.8, color='steelblue')
        axes[0].set_xlabel('k (number of clusters)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score: Full vs PCA-Reduced Dimensions')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(k_values)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Improvement percentage
        axes[1].bar(x, results_df['improvement_pct'], color='green', alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_xlabel('k (number of clusters)')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Silhouette Score Improvement with PCA')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(k_values)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kmeans_comparison.png', dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to {output_dir / 'kmeans_comparison.png'}")


def main():
    """
    Run complete curse of dimensionality analysis.
    """
    import time
    from data_prep import load_processed_data
    
    start_time = time.time()
    
    print("="*80)
    print("CURSE OF DIMENSIONALITY ANALYSIS")
    print("="*80)
    
    # Load processed data (sample immediately to avoid memory issues)
    print("\nLoading processed data...")
    X, df_meta = load_processed_data()
    
    print(f"Loaded data shape: {X.shape}")
    
    # Sample data early to reduce memory footprint
    max_sample = 100000  # Use 100k max for analysis
    if X.shape[0] > max_sample:
        print(f"Sampling {max_sample:,} rows for analysis...")
        np.random.seed(config.RANDOM_SEED)
        sample_idx = np.random.choice(X.shape[0], max_sample, replace=False)
        X = X[sample_idx]
        print(f"Sampled data shape: {X.shape}")
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        print("Converting sparse matrix to dense...")
        X = X.toarray()
    
    # 1. Distance distribution analysis
    analyze_distance_distribution(X, sample_size=10000)
    
    # 2. KMeans comparison
    compare_kmeans_dimensions(X, k_values=[3, 5, 8, 12], sample_size=50000)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
