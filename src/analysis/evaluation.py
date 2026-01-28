"""
Evaluation Module
Compute clustering metrics, analyze scaling performance, generate comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json

import config


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray, 
                               sample_size: int = None):
    """
    Compute standard clustering quality metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        sample_size: Sample size for silhouette (can be expensive)
        
    Returns:
        Dictionary of metrics
    """
    if sample_size is None:
        sample_size = min(config.SILHOUETTE_SAMPLE_SIZE, len(X))
    
    # Sample for silhouette
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    metrics = {
        'silhouette': silhouette_score(X_sample, labels_sample),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_clusters': len(np.unique(labels)),
        'n_samples': len(X)
    }
    
    return metrics


def load_scaling_data(algorithm: str = 'kmeans_mpi'):
    """
    Load scaling performance data for an algorithm.
    
    Args:
        algorithm: Algorithm name ('kmeans_mpi', 'hierarchical_mpi')
        
    Returns:
        DataFrame with scaling data
    """
    log_path = config.SCALING_DIR / f'{algorithm}_scaling.csv'
    
    if not log_path.exists():
        print(f"Warning: {log_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    return df


def compute_scaling_metrics(df: pd.DataFrame):
    """
    Compute speedup and efficiency from scaling data.
    
    Args:
        df: DataFrame with columns 'n_processes' and 'elapsed_time'
        
    Returns:
        DataFrame with added speedup and efficiency columns
    """
    df = df.copy()
    
    # Find baseline (single process)
    baseline_time = df[df['n_processes'] == 1]['elapsed_time'].values
    
    if len(baseline_time) == 0:
        print("Warning: No single-process baseline found")
        baseline_time = df['elapsed_time'].max()
    else:
        baseline_time = baseline_time[0]
    
    # Compute speedup and efficiency
    df['speedup'] = baseline_time / df['elapsed_time']
    df['efficiency'] = df['speedup'] / df['n_processes']
    df['ideal_speedup'] = df['n_processes']
    
    return df


def plot_scaling_analysis(algorithm: str = 'kmeans_mpi', output_dir=None):
    """
    Generate scaling analysis plots: runtime, speedup, efficiency.
    
    Args:
        algorithm: Algorithm name
        output_dir: Output directory for plots
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SCALING ANALYSIS: {algorithm.upper()}")
    print(f"{'='*80}")
    
    # Load data
    df = load_scaling_data(algorithm)
    
    if df.empty:
        print("No scaling data available")
        return
    
    # Aggregate multiple runs per process count (take mean)
    print(f"\nAggregating {len(df)} runs...")
    agg_dict = {'elapsed_time': 'mean'}
    if 'n_iterations' in df.columns:
        agg_dict['n_iterations'] = 'mean'
    df_agg = df.groupby('n_processes').agg(agg_dict).reset_index()
    print(f"Aggregated to {len(df_agg)} unique process counts: {df_agg['n_processes'].tolist()}")
    
    # Compute metrics on aggregated data
    df = compute_scaling_metrics(df_agg)
    
    # Sort by process count
    df = df.sort_values('n_processes')
    
    print("\nScaling Summary:")
    print(df[['n_processes', 'elapsed_time', 'speedup', 'efficiency']])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Runtime vs processes
    axes[0, 0].plot(df['n_processes'], df['elapsed_time'], 
                   marker='o', linewidth=2, markersize=8, label='Actual')
    axes[0, 0].set_xlabel('Number of Processes')
    axes[0, 0].set_ylabel('Runtime (seconds)')
    axes[0, 0].set_title('Runtime vs Number of Processes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Speedup
    axes[0, 1].plot(df['n_processes'], df['speedup'], 
                   marker='o', linewidth=2, markersize=8, label='Actual Speedup', color='green')
    axes[0, 1].plot(df['n_processes'], df['ideal_speedup'], 
                   linestyle='--', linewidth=2, label='Ideal (Linear)', color='red')
    axes[0, 1].set_xlabel('Number of Processes')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Speedup vs Number of Processes')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Efficiency
    axes[1, 0].plot(df['n_processes'], df['efficiency'], 
                   marker='o', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(1.0, color='red', linestyle='--', label='Ideal (100%)')
    axes[1, 0].set_xlabel('Number of Processes')
    axes[1, 0].set_ylabel('Parallel Efficiency')
    axes[1, 0].set_title('Parallel Efficiency vs Number of Processes')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.1)
    
    # 4. Speedup bar chart
    x = np.arange(len(df))
    width = 0.35
    axes[1, 1].bar(x, df['speedup'], width, label='Actual', alpha=0.8)
    axes[1, 1].bar(x + width, df['ideal_speedup'], width, label='Ideal', alpha=0.8)
    axes[1, 1].set_xlabel('Configuration')
    axes[1, 1].set_ylabel('Speedup')
    axes[1, 1].set_title('Speedup Comparison')
    axes[1, 1].set_xticks(x + width / 2)
    axes[1, 1].set_xticklabels([f"{int(p)}p" for p in df['n_processes']])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_scaling_analysis.png', 
               dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved scaling plot to {output_dir / f'{algorithm}_scaling_analysis.png'}")
    
    # Save summary stats
    summary = {
        'algorithm': algorithm,
        'max_processes': int(df['n_processes'].max()),
        'max_speedup': float(df['speedup'].max()),
        'efficiency_at_max': float(df[df['n_processes'] == df['n_processes'].max()]['efficiency'].values[0])
    }
    
    summary_path = output_dir / f'{algorithm}_scaling_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_path}")


def compare_algorithms(output_dir=None):
    """
    Compare different clustering algorithms on quality and performance.
    
    Args:
        output_dir: Output directory for comparison plots
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*80}")
    
    # Load results from different algorithms
    algorithms = {
        'K-Means (Single)': config.CLUSTERS_DIR / 'kmeans' / 'all_metrics.csv',
        'K-Means (MPI)': config.SCALING_DIR / 'kmeans_mpi_scaling.csv',
        'Hierarchical (Single)': config.CLUSTERS_DIR / 'hierarchical' / 'optimal_k_metrics.csv',
        'Hierarchical (MPI)': config.SCALING_DIR / 'hierarchical_mpi_scaling.csv',
    }
    
    comparison_data = []
    
    for name, path in algorithms.items():
        if path.exists():
            df = pd.read_csv(path)
            
            # Extract relevant metrics (varies by algorithm)
            entry = {'Algorithm': name}
            
            # Silhouette score
            if 'silhouette' in df.columns:
                entry['Silhouette'] = df['silhouette'].max() if len(df) > 0 else None
            elif 'silhouette_score' in df.columns:
                entry['Silhouette'] = df['silhouette_score'].max() if len(df) > 0 else None
            else:
                entry['Silhouette'] = None
            
            # Runtime
            if 'time' in df.columns:
                entry['Runtime (s)'] = df['time'].mean()
            elif 'elapsed_time' in df.columns:
                entry['Runtime (s)'] = df['elapsed_time'].mean()
            else:
                entry['Runtime (s)'] = None
            
            comparison_data.append(entry)
    
    if not comparison_data:
        print("No algorithm results found for comparison")
        return
    
    df_comp = pd.DataFrame(comparison_data)
    print("\n", df_comp)
    
    # Save comparison table
    df_comp.to_csv(output_dir / 'algorithm_comparison.csv', index=False)
    print(f"\nSaved comparison to {output_dir / 'algorithm_comparison.csv'}")


def main():
    """Run complete evaluation pipeline."""
    import time
    
    start_time = time.time()
    
    print("="*80)
    print("CLUSTERING EVALUATION AND ANALYSIS")
    print("="*80)
    
    # Scaling analysis for MPI algorithms
    for algorithm in ['kmeans_mpi', 'hierarchical_mpi']:
        plot_scaling_analysis(algorithm)
    
    # Compare algorithms
    compare_algorithms()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE - Time: {elapsed:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
