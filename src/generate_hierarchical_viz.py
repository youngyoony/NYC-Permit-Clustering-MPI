"""
Generate visualization for hierarchical baseline clustering.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import sys

import config


def main():
    print("="*80)
    print("GENERATING HIERARCHICAL BASELINE VISUALIZATION")
    print("="*80)
    
    # Load hierarchical baseline results
    cluster_dir = config.CLUSTERS_DIR / 'hierarchical'
    labels = np.load(cluster_dir / 'labels_k2.npy')
    k = len(np.unique(labels))
    n_samples = len(labels)
    
    print(f"\nLoaded: {n_samples:,} samples, k={k}")
    unique, counts = np.unique(labels, return_counts=True)
    for i, c in zip(unique, counts):
        print(f"  Cluster {i}: {c:,} ({c/n_samples*100:.1f}%)")
    
    # Load corresponding data
    # Hierarchical was run on a 5k stratified sample
    print("\nLoading data...")
    X = np.load(config.PROCESSED_DATA_PATH)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Try PCA data
    pca_path = config.DATA_DIR / "processed_X_pca.npy"
    if pca_path.exists():
        print("Using PCA-reduced data")
        X = np.load(pca_path)
    
    # Use first n_samples (hierarchical script used first 5k)
    X = X[:n_samples]
    print(f"Data shape: {X.shape}")
    
    # Create output directory
    output_dir = config.VISUALIZATIONS_DIR / 'hierarchical'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    print(f"Running t-SNE on {n_samples:,} samples...")
    
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED,
               perplexity=min(30, n_samples-1), max_iter=500, verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = sns.color_palette("husl", k)
    
    for i, label in enumerate(unique):
        mask = labels == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=[colors[i]], label=f'Cluster {label}',
                  alpha=0.6, s=20, edgecolors='none')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'HIERARCHICAL Clusters (t-SNE, k={k})')
    ax.legend(loc='best', markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'tsne_k{k}.png'
    plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
