"""
Validation: Prove MPI produces mathematically equivalent results
Compares baseline single-node vs distributed MPI implementations

FIXED:
1. Proper sampling for ARI/NMI (use same idx everywhere)
2. Greedy matching instead of O(k!) permutation
3. All metrics use sampled data consistently
4. REMOVED PCA loading - uses feature-selected data only!
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path
import json

import config


def load_baseline_results(k: int):
    """Load single-node baseline results."""
    baseline_dir = config.CLUSTERS_DIR / "kmeans"
    
    labels = np.load(baseline_dir / f'labels_k{k}.npy')
    centroids = np.load(baseline_dir / f'centroids_k{k}.npy')
    
    return labels, centroids


def load_mpi_results(k: int, nprocs: int):
    """Load MPI distributed results."""
    mpi_dir = config.CLUSTERS_DIR / "kmeans_mpi"
    
    labels = np.load(mpi_dir / f'labels_k{k}_np{nprocs}.npy')
    centroids = np.load(mpi_dir / f'centroids_k{k}_np{nprocs}.npy')
    
    return labels, centroids


def compare_centroids(centroids_baseline, centroids_mpi):
    """
    Compare centroids from baseline and MPI.
    """
    print("\n" + "="*80)
    print("CENTROID COMPARISON")
    print("="*80)
    
    norms_baseline = np.linalg.norm(centroids_baseline, axis=1)
    norms_mpi = np.linalg.norm(centroids_mpi, axis=1)
    
    sorted_baseline = centroids_baseline[np.argsort(norms_baseline)]
    sorted_mpi = centroids_mpi[np.argsort(norms_mpi)]
    
    centroid_diff = np.linalg.norm(sorted_baseline - sorted_mpi, axis=1)
    
    metrics = {
        'l2_diff_per_centroid': centroid_diff.tolist(),
        'l2_diff_mean': float(np.mean(centroid_diff)),
        'l2_diff_max': float(np.max(centroid_diff)),
        'l2_diff_std': float(np.std(centroid_diff)),
        'l2_diff_total': float(np.linalg.norm(sorted_baseline - sorted_mpi))
    }
    
    print(f"Mean L2 difference per centroid: {metrics['l2_diff_mean']:.6f}")
    print(f"Max L2 difference: {metrics['l2_diff_max']:.6f}")
    print(f"Total L2 difference: {metrics['l2_diff_total']:.6f}")
    
    return metrics


def compare_clustering_quality(X_sample, labels_baseline_sample, labels_mpi_sample):
    """
    Compare clustering quality metrics.
    ALL inputs should already be sampled!
    """
    print("\n" + "="*80)
    print("CLUSTERING QUALITY COMPARISON")
    print("="*80)
    print(f"Sample size: {len(X_sample):,}")
    
    # Baseline metrics
    print("Computing baseline metrics...")
    sil_baseline = silhouette_score(X_sample, labels_baseline_sample)
    db_baseline = davies_bouldin_score(X_sample, labels_baseline_sample)
    ch_baseline = calinski_harabasz_score(X_sample, labels_baseline_sample)
    
    # MPI metrics
    print("Computing MPI metrics...")
    sil_mpi = silhouette_score(X_sample, labels_mpi_sample)
    db_mpi = davies_bouldin_score(X_sample, labels_mpi_sample)
    ch_mpi = calinski_harabasz_score(X_sample, labels_mpi_sample)
    
    metrics = {
        "baseline": {
            "silhouette": float(sil_baseline),
            "davies_bouldin": float(db_baseline),
            "calinski_harabasz": float(ch_baseline),
        },
        "mpi": {
            "silhouette": float(sil_mpi),
            "davies_bouldin": float(db_mpi),
            "calinski_harabasz": float(ch_mpi),
        },
        "differences": {
            "silhouette": float(abs(sil_baseline - sil_mpi)),
            "davies_bouldin": float(abs(db_baseline - db_mpi)),
            "calinski_harabasz": float(abs(ch_baseline - ch_mpi)),
        },
        "relative_differences": {
            "silhouette": float(abs(sil_baseline - sil_mpi) / abs(sil_baseline) * 100) if sil_baseline != 0 else 0,
            "davies_bouldin": float(abs(db_baseline - db_mpi) / abs(db_baseline) * 100) if db_baseline != 0 else 0,
            "calinski_harabasz": float(abs(ch_baseline - ch_mpi) / abs(ch_baseline) * 100) if ch_baseline != 0 else 0,
        },
    }

    print("\nBaseline:")
    print(f"  Silhouette: {sil_baseline:.6f}")
    print(f"  Davies-Bouldin: {db_baseline:.6f}")
    print(f"  Calinski-Harabasz: {ch_baseline:.2f}")

    print("\nMPI:")
    print(f"  Silhouette: {sil_mpi:.6f}")
    print(f"  Davies-Bouldin: {db_mpi:.6f}")
    print(f"  Calinski-Harabasz: {ch_mpi:.2f}")

    print("\nRelative Differences (%):")
    print(f"  Silhouette: {metrics['relative_differences']['silhouette']:.4f}%")
    print(f"  Davies-Bouldin: {metrics['relative_differences']['davies_bouldin']:.4f}%")
    print(f"  Calinski-Harabasz: {metrics['relative_differences']['calinski_harabasz']:.4f}%")

    return metrics


def compare_label_agreement(labels_baseline_sample, labels_mpi_sample):
    """
    Compare label assignments using clustering similarity metrics.
    
    FIXED: 
    - Input should already be sampled!
    - Uses O(k²) greedy matching instead of O(k!) permutation
    """
    print("\n" + "="*80)
    print("LABEL AGREEMENT COMPARISON")
    print("="*80)
    print(f"Sample size: {len(labels_baseline_sample):,}")

    # ARI and NMI on sampled labels
    print("Computing ARI...")
    ari = adjusted_rand_score(labels_baseline_sample, labels_mpi_sample)
    
    print("Computing NMI...")
    nmi = normalized_mutual_info_score(labels_baseline_sample, labels_mpi_sample)

    # Greedy matching (O(k²) instead of O(k!))
    print("Computing greedy label matching...")
    unique_baseline = np.unique(labels_baseline_sample)
    unique_mpi = np.unique(labels_mpi_sample)
    k_baseline = len(unique_baseline)
    k_mpi = len(unique_mpi)
    
    print(f"  Baseline clusters: {k_baseline}")
    print(f"  MPI clusters: {k_mpi}")

    # Build confusion matrix
    confusion = np.zeros((k_baseline, k_mpi), dtype=np.int64)
    for i, lb in enumerate(unique_baseline):
        mask_baseline = (labels_baseline_sample == lb)
        for j, lm in enumerate(unique_mpi):
            confusion[i, j] = np.sum(mask_baseline & (labels_mpi_sample == lm))

    # Greedy assignment (pick best matches iteratively)
    best_mapping = {}
    used = set()
    
    for _ in range(min(k_baseline, k_mpi)):
        best_i, best_j, best_count = -1, -1, -1
        for i in range(k_baseline):
            if unique_baseline[i] in best_mapping:
                continue
            for j in range(k_mpi):
                if j not in used and confusion[i, j] > best_count:
                    best_count = confusion[i, j]
                    best_i, best_j = i, j
        
        if best_i >= 0:
            best_mapping[unique_baseline[best_i]] = unique_mpi[best_j]
            used.add(best_j)

    # Compute agreement
    matches = 0
    for i in range(len(labels_baseline_sample)):
        lb = labels_baseline_sample[i]
        lm = labels_mpi_sample[i]
        if lb in best_mapping and best_mapping[lb] == lm:
            matches += 1
    
    best_agreement = matches / len(labels_baseline_sample)

    metrics = {
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi),
        'direct_agreement': float(best_agreement),
        'direct_agreement_pct': float(best_agreement * 100),
        'label_mapping': {str(k): int(v) for k, v in best_mapping.items()}
    }

    print(f"\nResults:")
    print(f"  Adjusted Rand Index: {ari:.6f}")
    print(f"  Normalized Mutual Info: {nmi:.6f}")
    print(f"  Direct Agreement: {best_agreement*100:.2f}%")

    return metrics


def plot_validation_results(centroid_metrics, quality_metrics, output_dir):
    """Plot validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Centroid differences
    ax = axes[0]
    centroid_diffs = centroid_metrics['l2_diff_per_centroid']
    ax.bar(range(len(centroid_diffs)), centroid_diffs, color='steelblue')
    ax.axhline(centroid_metrics['l2_diff_mean'], color='red', linestyle='--', 
               label=f"Mean={centroid_metrics['l2_diff_mean']:.4f}")
    ax.set_xlabel('Centroid Index')
    ax.set_ylabel('L2 Distance')
    ax.set_title('Centroid Differences (Baseline vs MPI)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Quality metrics comparison
    ax = axes[1]
    metrics_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
    
    # Normalize for comparison
    baseline_vals = [
        quality_metrics['baseline']['silhouette'],
        quality_metrics['baseline']['davies_bouldin'] / 100,  # Scale down
        quality_metrics['baseline']['calinski_harabasz'] / 1000  # Scale down
    ]
    mpi_vals = [
        quality_metrics['mpi']['silhouette'],
        quality_metrics['mpi']['davies_bouldin'] / 100,
        quality_metrics['mpi']['calinski_harabasz'] / 1000
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, mpi_vals, width, label='MPI', alpha=0.8)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value (scaled)')
    ax.set_title('Quality Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_comparison.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved validation plot to {output_dir / 'validation_comparison.png'}")


def generate_validation_report(k, nprocs_list, output_dir):
    """
    Generate comprehensive validation report for multiple process counts.
    
    FIXED: Uses feature-selected data only (no PCA)!
    """
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(f"Validating K={k} across process counts: {nprocs_list}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature-selected data (NO PCA!)
    print("\nLoading feature-selected data...")
    X = np.load(config.PROCESSED_DATA_PATH)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    print(f"Full data shape: {X.shape}")
    
    # Load feature names if available
    feature_names_path = config.DATA_DIR / "feature_names.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Using features: {feature_names}")
    
    # Load reference (MPI with 1 process = single-node)
    print("\nLoading reference results (MPI np=1)...")
    labels_reference_full, centroids_reference = load_mpi_results(k, 1)
    print(f"Reference: {len(labels_reference_full):,} samples, {len(centroids_reference)} clusters")
    
    # Create sample indices ONCE and reuse everywhere
    max_samples = 50000
    n_samples = min(len(X), len(labels_reference_full))
    
    rng = np.random.default_rng(42)
    if n_samples > max_samples:
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        print(f"Subsampled to {max_samples} samples for validation")
    else:
        idx = np.arange(n_samples)
        print(f"Using all {n_samples} samples")
    
    # Apply sampling to X and reference labels
    X_sample = X[idx]
    labels_reference_sample = labels_reference_full[idx]
    
    # Validate each MPI configuration
    all_results = []
    
    for nprocs in nprocs_list:
        print(f"\n{'='*80}")
        print(f"VALIDATING: {nprocs} PROCESSES")
        print(f"{'='*80}")
        
        try:
            # Load MPI results
            labels_mpi_full, centroids_mpi = load_mpi_results(k, nprocs)
            print(f"Loaded: {len(labels_mpi_full):,} samples, {len(centroids_mpi)} clusters")
            
            # Apply SAME sampling to MPI labels
            labels_mpi_sample = labels_mpi_full[idx]
            
            # Compare centroids (no sampling needed - just k centroids)
            centroid_metrics = compare_centroids(centroids_reference, centroids_mpi)
            
            # Compare quality (all sampled)
            quality_metrics = compare_clustering_quality(
                X_sample, labels_reference_sample, labels_mpi_sample
            )
            
            # Compare labels (all sampled)
            label_metrics = compare_label_agreement(labels_reference_sample, labels_mpi_sample)
            
            # Create plot for this configuration
            plot_output_dir = output_dir / f'np{nprocs}'
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            plot_validation_results(centroid_metrics, quality_metrics, plot_output_dir)
            
            # Aggregate results
            result = {
                'k': k,
                'n_processes': nprocs,
                'sample_size': len(idx),
                'centroid_l2_mean': centroid_metrics['l2_diff_mean'],
                'centroid_l2_max': centroid_metrics['l2_diff_max'],
                'silhouette_baseline': quality_metrics['baseline']['silhouette'],
                'silhouette_mpi': quality_metrics['mpi']['silhouette'],
                'silhouette_diff': quality_metrics['differences']['silhouette'],
                'silhouette_rel_diff_pct': quality_metrics['relative_differences']['silhouette'],
                'davies_bouldin_baseline': quality_metrics['baseline']['davies_bouldin'],
                'davies_bouldin_mpi': quality_metrics['mpi']['davies_bouldin'],
                'adjusted_rand_index': label_metrics['adjusted_rand_index'],
                'normalized_mutual_info': label_metrics['normalized_mutual_info'],
                'direct_agreement_pct': label_metrics['direct_agreement_pct']
            }
            
            all_results.append(result)
            
            # Save individual result
            with open(plot_output_dir / 'validation_metrics.json', 'w') as f:
                json.dump({
                    'centroid_comparison': centroid_metrics,
                    'quality_comparison': quality_metrics,
                    'label_comparison': label_metrics
                }, f, indent=2)
            
        except FileNotFoundError as e:
            print(f"WARNING: Could not load results for {nprocs} processes: {e}")
            continue
    
    # Save summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'validation_summary.csv', index=False)
        print(f"\nSaved validation summary to {output_dir / 'validation_summary.csv'}")
        
        # Print summary table
        print("\n" + "="*80)
        print("VALIDATION SUMMARY TABLE")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Print the actual Silhouette score prominently
        print("\n" + "="*80)
        print("★★★ SILHOUETTE SCORE (Feature-Selected Data) ★★★")
        print("="*80)
        print(f"  Silhouette Score: {results_df['silhouette_baseline'].iloc[0]:.4f}")
        print("="*80)
        
        # Create summary visualization
        create_summary_visualization(results_df, output_dir)


def create_summary_visualization(df, output_dir):
    """Create summary visualization across all process counts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Centroid differences vs process count
    ax = axes[0, 0]
    ax.plot(df['n_processes'], df['centroid_l2_mean'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Mean L2 Distance')
    ax.set_title('Centroid Difference vs Process Count')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Silhouette score comparison
    ax = axes[0, 1]
    ax.plot(df['n_processes'], df['silhouette_baseline'], 
            label='Baseline', marker='s', linewidth=2, markersize=8)
    ax.plot(df['n_processes'], df['silhouette_mpi'], 
            label='MPI', marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score: Baseline vs MPI')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Agreement metrics
    ax = axes[1, 0]
    ax.plot(df['n_processes'], df['adjusted_rand_index'], 
            label='ARI', marker='o', linewidth=2, markersize=8)
    ax.plot(df['n_processes'], df['normalized_mutual_info'], 
            label='NMI', marker='s', linewidth=2, markersize=8)
    ax.plot(df['n_processes'], df['direct_agreement_pct']/100, 
            label='Direct Agreement', marker='^', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Agreement Score')
    ax.set_title('Label Agreement Metrics')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # 4. Relative difference percentage
    ax = axes[1, 1]
    ax.bar(df['n_processes'].astype(str), df['silhouette_rel_diff_pct'], color='steelblue')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Relative Difference (%)')
    ax.set_title('Silhouette Relative Difference')
    ax.axhline(y=1, color='green', linestyle='--', label='1% threshold')
    ax.axhline(y=5, color='orange', linestyle='--', label='5% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_summary.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary visualization to {output_dir / 'validation_summary.png'}")


def main():
    """Main validation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate MPI vs Baseline equivalence')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters')
    parser.add_argument('--nprocs', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                       help='Process counts to validate')
    
    args = parser.parse_args()
    
    output_dir = config.RESULTS_DIR / "validation"
    
    generate_validation_report(args.k, args.nprocs, output_dir)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
