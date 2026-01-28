"""
Hybrid Cluster Interpretation Module
=====================================
Clusters in PCA space for better silhouette scores,
but interprets using original feature space for business insights.

Strategy:
1. Load PCA-transformed data and original data
2. Load cluster labels (from PCA-space clustering)
3. Map clusters back to original features
4. Generate interpretable business profiles

Author: Team 4
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional

# Assuming config.py exists with paths
try:
    import config
except ImportError:
    # Fallback for local testing
    class config:
        DATA_DIR = Path("data")
        RESULTS_DIR = Path("results")
        PLOT_DPI = 300


class HybridClusterInterpreter:
    """
    Interprets clusters formed in PCA space using original features.
    
    The key insight: PCA gives better clustering quality (higher silhouette),
    but original features give interpretable business meaning.
    """
    
    def __init__(self, 
                 X_original: np.ndarray,
                 X_pca: np.ndarray,
                 labels: np.ndarray,
                 feature_names: List[str],
                 df_meta: pd.DataFrame = None,
                 pca_model = None):
        """
        Initialize the interpreter.
        
        Args:
            X_original: Original feature matrix (n_samples, n_features)
            X_pca: PCA-transformed matrix (n_samples, n_components)
            labels: Cluster labels from PCA-space clustering
            feature_names: Names of original features
            df_meta: Metadata DataFrame (Borough, Lat, Lon, etc.)
            pca_model: Fitted PCA model (for component analysis)
        """
        self.X_original = X_original
        self.X_pca = X_pca
        self.labels = labels
        self.feature_names = feature_names
        self.df_meta = df_meta
        self.pca_model = pca_model
        
        self.n_clusters = len(np.unique(labels))
        self.n_samples = len(labels)
        self.n_features = len(feature_names)
        
        print(f"HybridClusterInterpreter initialized:")
        print(f"  Samples: {self.n_samples:,}")
        print(f"  Original features: {self.n_features}")
        print(f"  PCA components: {X_pca.shape[1]}")
        print(f"  Clusters: {self.n_clusters}")
    
    def compute_cluster_profiles_original(self) -> pd.DataFrame:
        """
        Compute mean and std of ORIGINAL features for each cluster.
        This is the key to interpretation!
        
        Returns:
            DataFrame with cluster profiles in original feature space
        """
        print("\n" + "="*80)
        print("COMPUTING CLUSTER PROFILES (ORIGINAL FEATURE SPACE)")
        print("="*80)
        
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_data = self.X_original[mask]
            
            profile = {
                'cluster': cluster_id,
                'size': mask.sum(),
                'percentage': 100 * mask.sum() / self.n_samples
            }
            
            # Mean and std for each original feature
            for i, fname in enumerate(self.feature_names):
                profile[f'{fname}_mean'] = cluster_data[:, i].mean()
                profile[f'{fname}_std'] = cluster_data[:, i].std()
            
            profiles.append(profile)
        
        df_profiles = pd.DataFrame(profiles)
        
        # Print summary
        print("\nCluster sizes:")
        for _, row in df_profiles.iterrows():
            print(f"  Cluster {int(row['cluster'])}: {int(row['size']):,} ({row['percentage']:.1f}%)")
        
        return df_profiles
    
    def identify_distinguishing_features(self, top_n: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        For each cluster, identify which original features distinguish it most.
        Uses standardized difference from global mean.
        
        Returns:
            Dict mapping cluster_id to list of (feature_name, z_score) tuples
        """
        print("\n" + "="*80)
        print("IDENTIFYING DISTINGUISHING FEATURES PER CLUSTER")
        print("="*80)
        
        # Global statistics
        global_means = self.X_original.mean(axis=0)
        global_stds = self.X_original.std(axis=0)
        global_stds[global_stds == 0] = 1  # Avoid division by zero
        
        distinguishing = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_means = self.X_original[mask].mean(axis=0)
            
            # Z-score: how many stds is cluster mean from global mean
            z_scores = (cluster_means - global_means) / global_stds
            
            # Sort by absolute z-score
            sorted_indices = np.argsort(np.abs(z_scores))[::-1]
            
            top_features = []
            for idx in sorted_indices[:top_n]:
                fname = self.feature_names[idx]
                zscore = z_scores[idx]
                top_features.append((fname, zscore))
            
            distinguishing[cluster_id] = top_features
            
            # Print
            print(f"\nCluster {cluster_id} - Top distinguishing features:")
            for fname, zscore in top_features:
                direction = "↑ HIGH" if zscore > 0 else "↓ LOW"
                print(f"  {fname}: z={zscore:+.2f} ({direction})")
        
        return distinguishing
    
    def analyze_pca_loadings(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze PCA loadings to understand what each component represents.
        
        Returns:
            DataFrame with top features for each PC
        """
        if self.pca_model is None:
            print("WARNING: No PCA model provided, skipping loadings analysis")
            return None
        
        print("\n" + "="*80)
        print("PCA LOADINGS ANALYSIS")
        print("="*80)
        
        components = self.pca_model.components_
        n_components = components.shape[0]
        
        loadings_data = []
        
        for pc_idx in range(min(n_components, 5)):  # Top 5 PCs
            loadings = components[pc_idx]
            sorted_indices = np.argsort(np.abs(loadings))[::-1]
            
            print(f"\nPC{pc_idx + 1} (var={self.pca_model.explained_variance_ratio_[pc_idx]*100:.1f}%):")
            
            for rank, idx in enumerate(sorted_indices[:top_n]):
                fname = self.feature_names[idx]
                loading = loadings[idx]
                print(f"  {rank+1}. {fname}: {loading:+.3f}")
                
                loadings_data.append({
                    'PC': pc_idx + 1,
                    'rank': rank + 1,
                    'feature': fname,
                    'loading': loading
                })
        
        return pd.DataFrame(loadings_data)
    
    def generate_temporal_interpretation(self) -> Dict[int, str]:
        """
        Special interpretation for temporal patterns (Filing_Year based).
        Based on PDF findings: K-Means reveals Early/Mid/Late eras.
        """
        print("\n" + "="*80)
        print("TEMPORAL PATTERN INTERPRETATION")
        print("="*80)
        
        # Find Filing_Year feature
        year_idx = None
        for i, fname in enumerate(self.feature_names):
            if 'Filing_Year' in fname or 'year' in fname.lower():
                year_idx = i
                break
        
        if year_idx is None:
            print("WARNING: No Filing_Year feature found")
            return {}
        
        interpretations = {}
        cluster_years = []
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            mean_year = self.X_original[mask, year_idx].mean()
            cluster_years.append((cluster_id, mean_year))
        
        # Sort clusters by mean year
        cluster_years.sort(key=lambda x: x[1])
        
        # Assign era labels
        n_clusters = len(cluster_years)
        for i, (cluster_id, mean_year) in enumerate(cluster_years):
            if i < n_clusters // 3:
                era = "Early Era"
            elif i < 2 * n_clusters // 3:
                era = "Mid Era"
            else:
                era = "Late Era"
            
            interpretations[cluster_id] = f"{era} (avg year: {mean_year:.1f})"
            print(f"  Cluster {cluster_id}: {interpretations[cluster_id]}")
        
        return interpretations
    
    def create_radar_chart(self, output_dir: Path, selected_features: List[str] = None):
        """
        Create radar chart showing cluster profiles across key features.
        Great for presentations!
        """
        print("\n" + "="*80)
        print("CREATING RADAR CHART")
        print("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if selected_features is None:
            # Default: select interpretable features
            selected_features = [f for f in self.feature_names 
                               if any(key in f for key in ['Year', 'Month', 'LATITUDE', 'LONGITUDE', 'Residential'])]
            if len(selected_features) < 3:
                selected_features = self.feature_names[:6]
        
        # Get indices
        feature_indices = [self.feature_names.index(f) for f in selected_features if f in self.feature_names]
        selected_features = [self.feature_names[i] for i in feature_indices]
        
        if len(selected_features) < 3:
            print("WARNING: Need at least 3 features for radar chart")
            return
        
        # Normalize cluster means to [0, 1] for radar chart
        cluster_means = []
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            means = self.X_original[mask][:, feature_indices].mean(axis=0)
            cluster_means.append(means)
        
        cluster_means = np.array(cluster_means)
        
        # Min-max normalize across clusters
        mins = cluster_means.min(axis=0)
        maxs = cluster_means.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        cluster_means_norm = (cluster_means - mins) / ranges
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            values = cluster_means_norm[cluster_id].tolist()
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f'Cluster {cluster_id}', color=colors[cluster_id])
            ax.fill(angles, values, alpha=0.15, color=colors[cluster_id])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(selected_features, size=10)
        ax.set_title('Cluster Profiles (Normalized)', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved radar chart to {output_dir / 'cluster_radar_chart.png'}")
    
    def create_feature_heatmap(self, output_dir: Path):
        """
        Create heatmap showing standardized feature means per cluster.
        """
        print("\n" + "="*80)
        print("CREATING FEATURE HEATMAP")
        print("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute z-scores for each cluster
        global_means = self.X_original.mean(axis=0)
        global_stds = self.X_original.std(axis=0)
        global_stds[global_stds == 0] = 1
        
        z_matrix = []
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_means = self.X_original[mask].mean(axis=0)
            z_scores = (cluster_means - global_means) / global_stds
            z_matrix.append(z_scores)
        
        z_matrix = np.array(z_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Truncate feature names for display
        display_names = [f[:20] + '...' if len(f) > 20 else f for f in self.feature_names]
        
        sns.heatmap(z_matrix, 
                   xticklabels=display_names,
                   yticklabels=[f'Cluster {i}' for i in range(self.n_clusters)],
                   cmap='RdBu_r',
                   center=0,
                   annot=True if self.n_features <= 15 else False,
                   fmt='.2f',
                   ax=ax)
        
        ax.set_title('Cluster Feature Profiles (Z-scores from Global Mean)', fontsize=14)
        ax.set_xlabel('Features')
        ax.set_ylabel('Cluster')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_feature_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap to {output_dir / 'cluster_feature_heatmap.png'}")
    
    def generate_presentation_summary(self, output_dir: Path) -> str:
        """
        Generate a presentation-ready summary of cluster interpretations.
        """
        print("\n" + "="*80)
        print("GENERATING PRESENTATION SUMMARY")
        print("="*80)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get distinguishing features
        distinguishing = self.identify_distinguishing_features(top_n=3)
        
        summary_lines = [
            "="*80,
            "HYBRID CLUSTER INTERPRETATION SUMMARY",
            "="*80,
            "",
            "METHODOLOGY:",
            "  - Clustering performed in PCA space (better separation)",
            "  - Interpretation uses original features (business meaning)",
            "",
            f"RESULTS: {self.n_clusters} clusters identified",
            ""
        ]
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            size = mask.sum()
            pct = 100 * size / self.n_samples
            
            summary_lines.append(f"CLUSTER {cluster_id}: {size:,} permits ({pct:.1f}%)")
            summary_lines.append("-" * 40)
            
            # Distinguishing features
            summary_lines.append("  Key characteristics:")
            for fname, zscore in distinguishing[cluster_id]:
                direction = "HIGH" if zscore > 0 else "LOW"
                summary_lines.append(f"    • {fname}: {direction} (z={zscore:+.2f})")
            
            summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        
        # Save
        with open(output_dir / 'interpretation_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\nSaved summary to {output_dir / 'interpretation_summary.txt'}")
        
        return summary_text


def load_data_for_interpretation(k: int = 10, 
                                  method: str = 'kmeans_mpi',
                                  nprocs: int = 4) -> Tuple:
    """
    Load all necessary data for hybrid interpretation.
    
    Returns:
        Tuple of (X_original, X_pca, labels, feature_names, df_meta, pca_model)
    """
    print("="*80)
    print("LOADING DATA FOR HYBRID INTERPRETATION")
    print("="*80)
    
    # Load original data
    X_original = np.load(config.DATA_DIR / "processed_X.npy")
    print(f"Loaded original X: {X_original.shape}")
    
    # Load PCA data
    pca_path = config.DATA_DIR / "processed_X_pca.npy"
    if pca_path.exists():
        X_pca = np.load(pca_path)
        print(f"Loaded PCA X: {X_pca.shape}")
    else:
        print("WARNING: PCA data not found, will need to compute")
        X_pca = None
    
    # Load feature names
    feature_names_path = config.DATA_DIR / "feature_names.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_names)} feature names")
    else:
        feature_names = [f"feature_{i}" for i in range(X_original.shape[1])]
    
    # Load metadata
    df_meta = pd.read_csv(config.DATA_DIR / "processed_meta.csv", index_col=0)
    print(f"Loaded metadata: {df_meta.shape}")
    
    # Load labels
    if method == 'kmeans':
        labels_path = config.RESULTS_DIR / "clusters" / "kmeans" / f"labels_k{k}.npy"
    elif method == 'kmeans_mpi':
        labels_path = config.RESULTS_DIR / "clusters" / "kmeans_mpi" / f"labels_k{k}_np{nprocs}.npy"
    elif method == 'hierarchical':
        labels_path = config.RESULTS_DIR / "clusters" / "hierarchical" / f"labels_k{k}.npy"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    labels = np.load(labels_path)
    print(f"Loaded labels: {labels.shape}")
    
    return X_original, X_pca, labels, feature_names, df_meta, None


def main():
    """
    Main function to run hybrid cluster interpretation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Cluster Interpretation')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters')
    parser.add_argument('--method', type=str, default='kmeans_mpi',
                       choices=['kmeans', 'kmeans_mpi', 'hierarchical'])
    parser.add_argument('--nprocs', type=int, default=4, help='Number of MPI processes')
    
    args = parser.parse_args()
    
    # Load data
    X_original, X_pca, labels, feature_names, df_meta, pca_model = \
        load_data_for_interpretation(args.k, args.method, args.nprocs)
    
    # Initialize interpreter
    interpreter = HybridClusterInterpreter(
        X_original=X_original,
        X_pca=X_pca if X_pca is not None else X_original,
        labels=labels,
        feature_names=feature_names,
        df_meta=df_meta,
        pca_model=pca_model
    )
    
    # Output directory
    output_dir = config.RESULTS_DIR / "hybrid_interpretation" / f"{args.method}_k{args.k}"
    
    # Run analyses
    profiles = interpreter.compute_cluster_profiles_original()
    profiles.to_csv(output_dir / 'cluster_profiles_original.csv', index=False)
    
    distinguishing = interpreter.identify_distinguishing_features(top_n=5)
    
    interpreter.generate_temporal_interpretation()
    
    interpreter.create_feature_heatmap(output_dir)
    
    interpreter.create_radar_chart(output_dir)
    
    interpreter.generate_presentation_summary(output_dir)
    
    print("\n" + "="*80)
    print("HYBRID INTERPRETATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
