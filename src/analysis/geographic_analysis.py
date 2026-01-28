"""
Geographic Analysis and Visualization
Answers: "Where are development hotspots? What are spatial patterns?"

TODO 7: Geographic Analysis
- Map cluster distribution across NYC
- Identify development hotspots
- Borough-level analysis
- Spatial density visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

import config


def load_geodata_and_labels(k: int, method: str = 'kmeans', nprocs: int = None):
    """Load metadata with geographic coordinates and cluster labels."""
    print(f"\nLoading geodata and {method} labels (k={k})...")
    
    # Load metadata
    df_meta = pd.read_csv(config.PROCESSED_META_PATH)
    
    # Load labels
    if method == 'kmeans':
        labels_path = config.CLUSTERS_DIR / "kmeans" / f'labels_k{k}.npy'
    elif method == 'kmeans_mpi':
        if nprocs is None:
            raise ValueError("nprocs required for kmeans_mpi")
        labels_path = config.CLUSTERS_DIR / "kmeans_mpi" / f'labels_k{k}_np{nprocs}.npy'
    elif method == 'hierarchical':
        labels_path = config.CLUSTERS_DIR / "hierarchical" / f'labels_k{k}.npy'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    labels = np.load(labels_path)
    df_meta['cluster'] = labels
    
    # Filter valid coordinates
    if 'Latitude' in df_meta.columns and 'Longitude' in df_meta.columns:
        valid_coords = (
            (df_meta['Latitude'] >= config.NYC_LAT_MIN) &
            (df_meta['Latitude'] <= config.NYC_LAT_MAX) &
            (df_meta['Longitude'] >= config.NYC_LON_MIN) &
            (df_meta['Longitude'] <= config.NYC_LON_MAX)
        )
        df_meta = df_meta[valid_coords].copy()
        print(f"Valid coordinates: {len(df_meta):,} / {len(labels):,}")
    
    return df_meta


def create_spatial_scatter_map(df_meta: pd.DataFrame, output_dir: Path, 
                               sample_size: int = 50000):
    """
    Create scatter plot showing geographic distribution of clusters.
    
    Args:
        df_meta: Metadata with Latitude, Longitude, cluster
        output_dir: Output directory
        sample_size: Number of points to plot (for performance)
    """
    print("\n" + "="*80)
    print("CREATING SPATIAL SCATTER MAP")
    print("="*80)
    
    # Sample for visualization
    if len(df_meta) > sample_size:
        print(f"Sampling {sample_size:,} points for visualization...")
        df_plot = df_meta.sample(n=sample_size, random_state=config.RANDOM_SEED)
    else:
        df_plot = df_meta
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot each cluster with different color
    unique_clusters = sorted(df_plot['cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster_id, color in zip(unique_clusters, colors):
        cluster_data = df_plot[df_plot['cluster'] == cluster_id]
        ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'],
                  c=[color], alpha=0.3, s=5, label=f'Cluster {cluster_id}')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Distribution of Construction Permit Clusters\nNYC Development Patterns', 
                fontsize=14, fontweight='bold')
    ax.legend(markerscale=3, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add borough boundaries (approximate)
    # Manhattan: ~-74.02 to -73.91, 40.70 to 40.88
    # Brooklyn: ~-74.04 to -73.85, 40.57 to 40.74
    # Queens: ~-73.96 to -73.70, 40.54 to 40.80
    # Bronx: ~-73.93 to -73.75, 40.79 to 40.92
    # Staten Island: ~-74.26 to -74.05, 40.50 to 40.65
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution_scatter.png', 
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spatial scatter map to {output_dir / 'spatial_distribution_scatter.png'}")


def create_density_heatmaps(df_meta: pd.DataFrame, output_dir: Path):
    """
    Create density heatmaps for each cluster.
    
    Args:
        df_meta: Metadata with coordinates and cluster labels
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("CREATING DENSITY HEATMAPS")
    print("="*80)
    
    unique_clusters = sorted(df_meta['cluster'].unique())
    n_clusters = len(unique_clusters)
    
    # Calculate grid layout
    ncols = min(3, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes[idx]
        cluster_data = df_meta[df_meta['cluster'] == cluster_id]
        
        if len(cluster_data) > 100:
            # Create 2D histogram (heatmap)
            h = ax.hist2d(cluster_data['Longitude'], cluster_data['Latitude'],
                         bins=50, cmap='YlOrRd', cmin=1)
            plt.colorbar(h[3], ax=ax, label='Permit Count')
        else:
            # Too few points for heatmap, use scatter
            ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'],
                      alpha=0.5, s=20)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Cluster {cluster_id} Density\n({len(cluster_data):,} permits)')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_density_heatmaps.png', 
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved density heatmaps to {output_dir / 'cluster_density_heatmaps.png'}")


def analyze_borough_patterns(df_meta: pd.DataFrame, output_dir: Path):
    """
    Analyze cluster distribution by borough.
    
    Args:
        df_meta: Metadata with Borough and cluster
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("ANALYZING BOROUGH PATTERNS")
    print("="*80)
    
    if 'Borough' not in df_meta.columns:
        print("WARNING: Borough column not found")
        return
    
    # Cross-tabulation
    crosstab = pd.crosstab(df_meta['Borough'], df_meta['cluster'])
    crosstab_pct = pd.crosstab(df_meta['Borough'], df_meta['cluster'], normalize='index') * 100
    
    print("\nPermit counts by Borough and Cluster:")
    print(crosstab)
    print("\nPercentage distribution:")
    print(crosstab_pct)
    
    # Save tables
    crosstab.to_csv(output_dir / 'borough_cluster_counts.csv')
    crosstab_pct.to_csv(output_dir / 'borough_cluster_percentages.csv')
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked bar chart (absolute)
    crosstab.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_xlabel('Borough')
    axes[0].set_ylabel('Number of Permits')
    axes[0].set_title('Cluster Distribution by Borough (Counts)')
    axes[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Stacked bar chart (percentage)
    crosstab_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
    axes[1].set_xlabel('Borough')
    axes[1].set_ylabel('Percentage')
    axes[1].set_title('Cluster Distribution by Borough (Percentages)')
    axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 100])
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'borough_analysis.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved borough analysis to {output_dir / 'borough_analysis.png'}")


def identify_hotspots(df_meta: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """
    Identify development hotspots using spatial clustering.
    
    Args:
        df_meta: Metadata with coordinates
        output_dir: Output directory
        top_n: Number of top hotspots to identify
    """
    print("\n" + "="*80)
    print("IDENTIFYING DEVELOPMENT HOTSPOTS")
    print("="*80)
    
    # Grid-based density analysis
    lat_bins = np.linspace(config.NYC_LAT_MIN, config.NYC_LAT_MAX, 50)
    lon_bins = np.linspace(config.NYC_LON_MIN, config.NYC_LON_MAX, 50)
    
    # Create 2D histogram
    H, lat_edges, lon_edges = np.histogram2d(
        df_meta['Latitude'], df_meta['Longitude'],
        bins=[lat_bins, lon_bins]
    )
    
    # Find hotspot centers
    hotspot_indices = []
    H_flat = H.flatten()
    sorted_indices = np.argsort(H_flat)[::-1]  # Sort descending
    
    for i in range(min(top_n, len(sorted_indices))):
        if H_flat[sorted_indices[i]] > 0:
            lat_idx = sorted_indices[i] // len(lon_bins)
            lon_idx = sorted_indices[i] % len(lon_bins)
            
            lat_center = (lat_edges[lat_idx] + lat_edges[lat_idx + 1]) / 2
            lon_center = (lon_edges[lon_idx] + lon_edges[lon_idx + 1]) / 2
            count = H_flat[sorted_indices[i]]
            
            hotspot_indices.append({
                'rank': i + 1,
                'latitude': lat_center,
                'longitude': lon_center,
                'permit_count': int(count)
            })
    
    hotspots_df = pd.DataFrame(hotspot_indices)
    hotspots_df.to_csv(output_dir / 'development_hotspots.csv', index=False)
    
    print(f"\nTop {len(hotspots_df)} Development Hotspots:")
    for _, hotspot in hotspots_df.iterrows():
        print(f"  Rank {hotspot['rank']}: ({hotspot['latitude']:.4f}, {hotspot['longitude']:.4f}) - " +
              f"{hotspot['permit_count']} permits")
    
    # Visualize hotspots
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot all points
    ax.scatter(df_meta['Longitude'], df_meta['Latitude'], 
              c='lightgray', alpha=0.2, s=1, label='All permits')
    
    # Overlay hotspots
    ax.scatter(hotspots_df['longitude'], hotspots_df['latitude'],
              c='red', s=hotspots_df['permit_count']*2, alpha=0.6,
              edgecolors='darkred', linewidth=1.5, marker='*',
              label='Hotspots')
    
    # Annotate top 5
    for _, hotspot in hotspots_df.head(5).iterrows():
        ax.annotate(f"#{int(hotspot['rank'])}", 
                   (hotspot['longitude'], hotspot['latitude']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='darkred')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('NYC Construction Development Hotspots', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'development_hotspots_map.png', 
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved hotspot analysis to {output_dir / 'development_hotspots_map.png'}")


def analyze_cluster_geography(df_meta: pd.DataFrame, output_dir: Path):
    """
    Analyze geographic characteristics of each cluster.
    
    Args:
        df_meta: Metadata with coordinates and cluster
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("ANALYZING CLUSTER GEOGRAPHIC CHARACTERISTICS")
    print("="*80)
    
    unique_clusters = sorted(df_meta['cluster'].unique())
    
    geo_profiles = []
    
    for cluster_id in unique_clusters:
        cluster_data = df_meta[df_meta['cluster'] == cluster_id]
        
        profile = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'centroid_lat': cluster_data['Latitude'].mean(),
            'centroid_lon': cluster_data['Longitude'].mean(),
            'lat_std': cluster_data['Latitude'].std(),
            'lon_std': cluster_data['Longitude'].std(),
            'lat_range': cluster_data['Latitude'].max() - cluster_data['Latitude'].min(),
            'lon_range': cluster_data['Longitude'].max() - cluster_data['Longitude'].min()
        }
        
        # Calculate spread (approximate area)
        # Use Haversine approximation for small distances
        lat_km = profile['lat_range'] * 111  # 1 degree lat ≈ 111 km
        lon_km = profile['lon_range'] * 111 * np.cos(np.radians(profile['centroid_lat']))
        profile['approx_area_km2'] = lat_km * lon_km
        
        # Density
        profile['density_permits_per_km2'] = profile['size'] / profile['approx_area_km2'] if profile['approx_area_km2'] > 0 else 0
        
        geo_profiles.append(profile)
        
        print(f"\nCluster {cluster_id} Geographic Profile:")
        print(f"  Centroid: ({profile['centroid_lat']:.4f}, {profile['centroid_lon']:.4f})")
        print(f"  Spread: {profile['lat_range']:.4f}° lat × {profile['lon_range']:.4f}° lon")
        print(f"  Approximate area: {profile['approx_area_km2']:.2f} km²")
        print(f"  Density: {profile['density_permits_per_km2']:.1f} permits/km²")
    
    # Save profiles
    geo_df = pd.DataFrame(geo_profiles)
    geo_df.to_csv(output_dir / 'cluster_geographic_profiles.csv', index=False)
    
    print(f"\nSaved geographic profiles to {output_dir / 'cluster_geographic_profiles.csv'}")
    
    return geo_df


def create_comprehensive_geographic_report(df_meta: pd.DataFrame, geo_df: pd.DataFrame,
                                           output_dir: Path):
    """Generate comprehensive geographic analysis report."""
    print("\n" + "="*80)
    print("GENERATING GEOGRAPHIC REPORT")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("GEOGRAPHIC ANALYSIS REPORT")
    report.append("NYC Construction Permit Spatial Patterns")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL SPATIAL DISTRIBUTION:")
    report.append(f"  Total permits analyzed: {len(df_meta):,}")
    report.append(f"  Latitude range: {df_meta['Latitude'].min():.4f} to {df_meta['Latitude'].max():.4f}")
    report.append(f"  Longitude range: {df_meta['Longitude'].min():.4f} to {df_meta['Longitude'].max():.4f}")
    report.append("")
    
    # Cluster-by-cluster
    report.append("="*80)
    report.append("CLUSTER SPATIAL PROFILES:")
    report.append("="*80)
    
    for _, row in geo_df.iterrows():
        report.append(f"\nCluster {row['cluster']}:")
        report.append(f"  📍 Geographic center: ({row['centroid_lat']:.4f}, {row['centroid_lon']:.4f})")
        report.append(f"  📏 Coverage area: ~{row['approx_area_km2']:.2f} km²")
        report.append(f"  📊 Permit density: {row['density_permits_per_km2']:.1f} permits/km²")
        report.append(f"  🎯 Concentration: {'High' if row['density_permits_per_km2'] > geo_df['density_permits_per_km2'].median() else 'Low'}")
    
    report.append("")
    report.append("="*80)
    report.append("KEY FINDINGS:")
    report.append("="*80)
    
    # Most concentrated cluster
    most_dense = geo_df.loc[geo_df['density_permits_per_km2'].idxmax()]
    report.append(f"\n🔥 Most concentrated development:")
    report.append(f"   Cluster {most_dense['cluster']} - {most_dense['density_permits_per_km2']:.1f} permits/km²")
    report.append(f"   Location: ({most_dense['centroid_lat']:.4f}, {most_dense['centroid_lon']:.4f})")
    
    # Most spread out cluster
    most_spread = geo_df.loc[geo_df['approx_area_km2'].idxmax()]
    report.append(f"\n📍 Most geographically dispersed:")
    report.append(f"   Cluster {most_spread['cluster']} - {most_spread['approx_area_km2']:.1f} km²")
    
    # Resource allocation
    report.append("")
    report.append("="*80)
    report.append("RESOURCE ALLOCATION INSIGHTS:")
    report.append("="*80)
    
    for _, row in geo_df.iterrows():
        report.append(f"\nCluster {row['cluster']}:")
        if row['density_permits_per_km2'] > geo_df['density_permits_per_km2'].median():
            report.append(f"  → High density area: Deploy mobile inspection units")
            report.append(f"  → Consider establishing local processing center")
        else:
            report.append(f"  → Dispersed area: Optimize inspector routing")
            report.append(f"  → Remote inspection protocols may apply")
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'geographic_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved geographic report to {output_dir / 'geographic_analysis_report.txt'}")


def main():
    """Main geographic analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Geographic analysis of clusters')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'kmeans_mpi', 'hierarchical'],
                       help='Clustering method')
    parser.add_argument('--nprocs', type=int, default=None,
                       help='Number of processes (for MPI methods)')
    
    args = parser.parse_args()
    
    # Load data
    df_meta = load_geodata_and_labels(args.k, args.method, args.nprocs)
    
    # Create output directory
    output_dir = config.RESULTS_DIR / "geographic_analysis" / f"{args.method}_k{args.k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create spatial visualizations
    create_spatial_scatter_map(df_meta, output_dir)
    create_density_heatmaps(df_meta, output_dir)
    
    # Borough analysis
    analyze_borough_patterns(df_meta, output_dir)
    
    # Hotspot identification
    identify_hotspots(df_meta, output_dir)
    
    # Cluster geographic profiles
    geo_df = analyze_cluster_geography(df_meta, output_dir)
    
    # Comprehensive report
    create_comprehensive_geographic_report(df_meta, geo_df, output_dir)
    
    print("\n" + "="*80)
    print("GEOGRAPHIC ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
