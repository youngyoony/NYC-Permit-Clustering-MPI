#!/usr/bin/env python3
"""
Compute Clustering Metrics for MPI Results
Calculates Silhouette, Davies-Bouldin, Calinski-Harabasz for MPI clustering results.

UPDATED: Uses PCA data to match clustering space!
         Clustering was done in PCA space → metrics must be computed in PCA space

Usage:
    python compute_mpi_metrics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json
import time

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SAMPLE_SIZE = 50000  # Sample size for Silhouette (full data too slow)
RANDOM_SEED = 42


def load_data():
    """
    Load PCA-transformed data for metric computation.
    
    IMPORTANT: Clustering was done in PCA space, so metrics must be 
    computed in PCA space to get accurate silhouette scores (~0.25).
    Using original features would give ~0.08 silhouette.
    """
    # FIXED: Use PCA data to match clustering space!
    pca_path = DATA_DIR / "processed_X_pca.npy"
    original_path = DATA_DIR / "processed_X.npy"
    
    if pca_path.exists():
        print(f"✓ Loading PCA data from {pca_path}...")
        X = np.load(pca_path)
        print(f"  Shape: {X.shape} (PCA components)")
        print(f"  Expected silhouette: ~0.25")
    else:
        print(f"⚠ WARNING: PCA data not found at {pca_path}")
        print(f"  Falling back to original features (silhouette will be ~0.08)")
        X = np.load(original_path)
        print(f"  Shape: {X.shape} (original features)")
    
    return X


def compute_metrics(X, labels, sample_size=SAMPLE_SIZE):
    """Compute clustering quality metrics."""
    n_samples = len(X)
    n_clusters = len(np.unique(labels))
    
    print(f"  Samples: {n_samples:,}, Clusters: {n_clusters}")
    
    # Sample for Silhouette (very slow on full data)
    if n_samples > sample_size:
        np.random.seed(RANDOM_SEED)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_idx]
        labels_sample = labels[sample_idx]
        print(f"  Using {sample_size:,} samples for metrics")
    else:
        X_sample = X
        labels_sample = labels
    
    # Compute metrics
    start = time.time()
    
    sil = silhouette_score(X_sample, labels_sample)
    print(f"  Silhouette: {sil:.4f} ({time.time()-start:.1f}s)")
    
    start = time.time()
    db = davies_bouldin_score(X_sample, labels_sample)
    print(f"  Davies-Bouldin: {db:.4f} ({time.time()-start:.1f}s)")
    
    start = time.time()
    ch = calinski_harabasz_score(X_sample, labels_sample)
    print(f"  Calinski-Harabasz: {ch:.2f} ({time.time()-start:.1f}s)")
    
    return {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
        'n_clusters': n_clusters,
        'n_samples': n_samples,
        'sample_size_for_metrics': min(sample_size, n_samples)
    }


def process_kmeans_mpi(X):
    """Process K-Means MPI results."""
    print("\n" + "="*60)
    print("K-MEANS MPI METRICS")
    print("="*60)
    
    kmeans_dir = RESULTS_DIR / "clusters" / "kmeans_mpi"
    results = []
    
    # Find all label files
    label_files = sorted(kmeans_dir.glob("labels_k10_np*.npy"))
    
    if not label_files:
        print(f"No label files found in {kmeans_dir}")
        return pd.DataFrame()
    
    for label_path in label_files:
        # Extract np value from filename
        filename = label_path.stem  # labels_k10_np64
        np_val = int(filename.split('_np')[1])
        
        print(f"\n--- np={np_val} ---")
        labels = np.load(label_path)
        
        # Match data size
        if len(labels) != len(X):
            print(f"  WARNING: labels ({len(labels)}) != X ({len(X)})")
            min_len = min(len(labels), len(X))
            labels = labels[:min_len]
            X_use = X[:min_len]
        else:
            X_use = X
        
        metrics = compute_metrics(X_use, labels)
        metrics['algorithm'] = 'kmeans_mpi'
        metrics['n_processes'] = np_val
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Save results
    output_path = RESULTS_DIR / "clusters" / "kmeans_mpi" / "quality_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df


def process_hierarchical_mpi(X):
    """Process Hierarchical MPI results."""
    print("\n" + "="*60)
    print("HIERARCHICAL MPI METRICS")
    print("="*60)
    
    hier_dir = RESULTS_DIR / "clusters" / "hierarchical_mpi"
    results = []
    
    # Find all label files
    label_files = sorted(hier_dir.glob("labels_k10_np*.npy"))
    
    if not label_files:
        print(f"No label files found in {hier_dir}")
        return pd.DataFrame()
    
    for label_path in label_files:
        filename = label_path.stem
        np_val = int(filename.split('_np')[1])
        
        print(f"\n--- np={np_val} ---")
        labels = np.load(label_path)
        
        # Hierarchical uses 5% sample (~200k), need to match
        if len(labels) != len(X):
            print(f"  Labels: {len(labels):,}, X: {len(X):,}")
            # Use first N samples to match
            min_len = min(len(labels), len(X))
            labels = labels[:min_len]
            X_use = X[:min_len]
        else:
            X_use = X
        
        metrics = compute_metrics(X_use, labels)
        metrics['algorithm'] = 'hierarchical_mpi'
        metrics['n_processes'] = np_val
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Save results
    output_path = RESULTS_DIR / "clusters" / "hierarchical_mpi" / "quality_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df


def process_baselines(X):
    """Process baseline (single-node) results."""
    print("\n" + "="*60)
    print("BASELINE METRICS")
    print("="*60)
    
    results = []
    
    # K-Means Baseline
    kmeans_baseline_path = RESULTS_DIR / "clusters" / "kmeans" / "labels_k10.npy"
    if kmeans_baseline_path.exists():
        print(f"\n--- K-Means Baseline ---")
        labels = np.load(kmeans_baseline_path)
        
        if len(labels) != len(X):
            print(f"  Labels: {len(labels):,}, X: {len(X):,}")
            min_len = min(len(labels), len(X))
            labels = labels[:min_len]
            X_use = X[:min_len]
        else:
            X_use = X
        
        metrics = compute_metrics(X_use, labels)
        metrics['algorithm'] = 'kmeans_baseline'
        metrics['n_processes'] = 1
        results.append(metrics)
    
    # Hierarchical Baseline
    hier_baseline_path = RESULTS_DIR / "clusters" / "hierarchical" / "labels_k10.npy"
    if hier_baseline_path.exists():
        print(f"\n--- Hierarchical Baseline ---")
        labels = np.load(hier_baseline_path)
        
        if len(labels) != len(X):
            print(f"  Labels: {len(labels):,}, X: {len(X):,}")
            min_len = min(len(labels), len(X))
            labels = labels[:min_len]
            X_use = X[:min_len]
        else:
            X_use = X
        
        metrics = compute_metrics(X_use, labels)
        metrics['algorithm'] = 'hierarchical_baseline'
        metrics['n_processes'] = 1
        results.append(metrics)
    
    if results:
        df = pd.DataFrame(results)
        output_path = RESULTS_DIR / "clusters" / "baseline_quality_metrics.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        return df
    
    return pd.DataFrame()


def main():
    print("="*60)
    print("COMPUTING MPI CLUSTERING METRICS (PCA SPACE)")
    print("="*60)
    
    # Load PCA data
    X = load_data()
    
    # Process baselines
    df_baseline = process_baselines(X)
    
    # Process K-Means MPI
    df_kmeans = process_kmeans_mpi(X)
    
    # Process Hierarchical MPI
    df_hier = process_hierarchical_mpi(X)
    
    # Combined Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_results = []
    
    if not df_baseline.empty:
        all_results.append(df_baseline)
        print("\nBaseline Results:")
        for _, row in df_baseline.iterrows():
            print(f"  {row['algorithm']}: Silhouette={row['silhouette']:.4f}, "
                  f"DB={row['davies_bouldin']:.4f}, CH={row['calinski_harabasz']:.2f}")
    
    if not df_kmeans.empty:
        all_results.append(df_kmeans)
        print("\nK-Means MPI (best by silhouette):")
        best_km = df_kmeans.loc[df_kmeans['silhouette'].idxmax()]
        print(f"  np={int(best_km['n_processes'])}: Silhouette={best_km['silhouette']:.4f}, "
              f"DB={best_km['davies_bouldin']:.4f}, CH={best_km['calinski_harabasz']:.2f}")
    
    if not df_hier.empty:
        all_results.append(df_hier)
        print("\nHierarchical MPI (best by silhouette):")
        best_hi = df_hier.loc[df_hier['silhouette'].idxmax()]
        print(f"  np={int(best_hi['n_processes'])}: Silhouette={best_hi['silhouette']:.4f}, "
              f"DB={best_hi['davies_bouldin']:.4f}, CH={best_hi['calinski_harabasz']:.2f}")
    
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = RESULTS_DIR / "clusters" / "all_quality_metrics.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()