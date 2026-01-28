#!/usr/bin/env python3
"""
Geographic Visualization Script - WITH ERA ANALYSIS
Creates NYC map with cluster points, grouped by temporal era.

Key Insight: K-Means clusters correspond to different time periods!
- Early (2004-2005): Clusters 1, 7, 9 - Lower Equipment Work ratio
- Mid (2006-2008): Clusters 0, 2, 4, 5, 8 - Transition period
- Late (2010-2011): Clusters 3, 6 - Higher Equipment Work & Minor Alteration

Run on SeaWulf:
    python run_geographic_viz_by_era.py --labels results/clusters/kmeans_mpi/labels_k10_np1.npy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION - Team Project paths
# ============================================================================
BASE_DIR = Path("/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RAW_DATA_PATH = DATA_DIR / "DOB_Permit_Issuance_merged.csv"

# NYC geographic bounds
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

# Plot settings
PLOT_DPI = 300
RANDOM_SEED = 42

# ============================================================================
# ERA MAPPING (based on cluster interpretation analysis)
# ============================================================================
CLUSTER_TO_ERA = {
    1: 'Early (2004-2005)',
    7: 'Early (2004-2005)',
    9: 'Early (2004-2005)',
    0: 'Mid (2006-2008)',
    2: 'Mid (2006-2008)',
    4: 'Mid (2006-2008)',
    5: 'Mid (2006-2008)',
    8: 'Mid (2006-2008)',
    3: 'Late (2010-2011)',
    6: 'Late (2010-2011)'
}

CLUSTER_TO_YEAR = {
    0: 2007, 1: 2005, 2: 2006, 3: 2010, 4: 2008,
    5: 2007, 6: 2011, 7: 2004, 8: 2008, 9: 2004
}

ERA_COLORS = {
    'Early (2004-2005)': '#2166ac',   # Blue
    'Mid (2006-2008)': '#4daf4a',     # Green
    'Late (2010-2011)': '#d73027'      # Red
}

ERA_ORDER = ['Early (2004-2005)', 'Mid (2006-2008)', 'Late (2010-2011)']


def load_data_with_coords(sample_size=None):
    """Load data with lat/lon coordinates."""
    print(f"\nLoading data from {RAW_DATA_PATH}...")
    
    usecols = ['LATITUDE', 'LONGITUDE', 'BOROUGH', 'Permit Type', 'Filing Date']
    
    try:
        df = pd.read_csv(RAW_DATA_PATH, usecols=usecols, low_memory=False)
    except:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    print(f"Loaded {len(df):,} rows")
    
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size]
        print(f"Using first {sample_size:,} rows")
    
    # Filter to valid NYC coordinates
    valid_coords = (
        (df['LATITUDE'] >= NYC_LAT_MIN) & (df['LATITUDE'] <= NYC_LAT_MAX) &
        (df['LONGITUDE'] >= NYC_LON_MIN) & (df['LONGITUDE'] <= NYC_LON_MAX)
    )
    df = df[valid_coords].copy()
    print(f"After coordinate filter: {len(df):,} rows")
    
    return df


# ============================================================================
# NEW: ERA-BASED VISUALIZATIONS
# ============================================================================

def plot_by_era_comparison(df: pd.DataFrame, labels: np.ndarray, output_dir: Path,
                           sample_per_era: int = 20000):
    """Create side-by-side maps for each era (3 panels)."""
    print("\n" + "="*70)
    print("CREATING ERA COMPARISON MAP (3 PANELS)")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels
    df['era'] = df['cluster'].map(CLUSTER_TO_ERA)
    
    # Print era statistics
    print("\nEra Distribution:")
    for era in ERA_ORDER:
        count = (df['era'] == era).sum()
        pct = count / len(df) * 100
        clusters = [c for c, e in CLUSTER_TO_ERA.items() if e == era]
        print(f"  {era}: {count:,} ({pct:.1f}%) - Clusters {clusters}")
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    
    for i, era in enumerate(ERA_ORDER):
        era_data = df[df['era'] == era]
        era_count = len(era_data)
        era_pct = era_count / len(df) * 100
        
        # Sample for plotting
        if len(era_data) > sample_per_era:
            np.random.seed(RANDOM_SEED + i)
            idx = np.random.choice(len(era_data), sample_per_era, replace=False)
            plot_data = era_data.iloc[idx]
        else:
            plot_data = era_data
        
        axes[i].scatter(
            plot_data['LONGITUDE'],
            plot_data['LATITUDE'],
            c=ERA_COLORS[era],
            s=2,
            alpha=0.4
        )
        
        axes[i].set_xlim(NYC_LON_MIN, NYC_LON_MAX)
        axes[i].set_ylim(NYC_LAT_MIN, NYC_LAT_MAX)
        axes[i].set_xlabel('Longitude', fontsize=11)
        axes[i].set_ylabel('Latitude', fontsize=11)
        axes[i].set_title(f'{era}\n(n={era_count:,}, {era_pct:.1f}%)', 
                          fontsize=13, fontweight='bold', color=ERA_COLORS[era])
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('NYC Construction Permits: Spatial Distribution by Era\n'
                 '(K-Means Clustering Reveals Temporal Patterns)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'geographic_by_era.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_era_density_heatmaps(df: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    """Create density heatmaps for each era."""
    print("\n" + "="*70)
    print("CREATING ERA DENSITY HEATMAPS")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels
    df['era'] = df['cluster'].map(CLUSTER_TO_ERA)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    
    for i, era in enumerate(ERA_ORDER):
        era_data = df[df['era'] == era]
        era_count = len(era_data)
        
        hb = axes[i].hexbin(
            era_data['LONGITUDE'],
            era_data['LATITUDE'],
            gridsize=40,
            cmap='YlOrRd',
            mincnt=1
        )
        
        axes[i].set_xlim(NYC_LON_MIN, NYC_LON_MAX)
        axes[i].set_ylim(NYC_LAT_MIN, NYC_LAT_MAX)
        axes[i].set_xlabel('Longitude', fontsize=11)
        axes[i].set_ylabel('Latitude', fontsize=11)
        axes[i].set_title(f'{era}\n(n={era_count:,})', fontsize=13, fontweight='bold')
        plt.colorbar(hb, ax=axes[i], label='Permit Count')
    
    plt.suptitle('NYC Construction Permits: Density Heatmap by Era', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'geographic_density_by_era.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_all_eras_overlay(df: pd.DataFrame, labels: np.ndarray, output_dir: Path,
                          sample_per_era: int = 15000):
    """Create single map with all eras overlaid (color-coded)."""
    print("\n" + "="*70)
    print("CREATING ALL ERAS OVERLAY MAP")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels
    df['era'] = df['cluster'].map(CLUSTER_TO_ERA)
    
    fig, ax = plt.subplots(figsize=(14, 16))
    
    # Plot in chronological order (early first, late on top)
    for era in ERA_ORDER:
        era_data = df[df['era'] == era]
        era_count = len(era_data)
        era_pct = era_count / len(df) * 100
        
        if len(era_data) > sample_per_era:
            np.random.seed(RANDOM_SEED)
            idx = np.random.choice(len(era_data), sample_per_era, replace=False)
            plot_data = era_data.iloc[idx]
        else:
            plot_data = era_data
        
        ax.scatter(
            plot_data['LONGITUDE'],
            plot_data['LATITUDE'],
            c=ERA_COLORS[era],
            s=3,
            alpha=0.4,
            label=f'{era} (n={era_count:,}, {era_pct:.1f}%)'
        )
    
    ax.set_xlim(NYC_LON_MIN, NYC_LON_MAX)
    ax.set_ylim(NYC_LAT_MIN, NYC_LAT_MAX)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('NYC Construction Permits: Temporal Evolution\n'
                 '(Colored by K-Means Cluster Era)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', markerscale=4, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'geographic_all_eras_overlay.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_era_borough_breakdown(df: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    """Create bar chart showing borough distribution by era."""
    print("\n" + "="*70)
    print("CREATING ERA-BOROUGH BREAKDOWN CHART")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels
    df['era'] = df['cluster'].map(CLUSTER_TO_ERA)
    
    # Borough 처리 - 숫자든 문자열이든 처리
    borough_names = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'}
    
    if df['BOROUGH'].dtype in ['int64', 'float64']:
        df['borough_name'] = df['BOROUGH'].map(borough_names)
    else:
        # 이미 문자열인 경우 - 대소문자 통일
        df['borough_name'] = df['BOROUGH'].str.strip().str.title()
    
    # NaN 제거
    df = df.dropna(subset=['borough_name'])
    print(f"Borough unique values: {df['borough_name'].unique().tolist()}")
    
    # Calculate percentages
    era_borough = df.groupby(['era', 'borough_name']).size().unstack(fill_value=0)
    era_borough_pct = era_borough.div(era_borough.sum(axis=1), axis=0) * 100
    
    # Reorder - 존재하는 컬럼만 사용
    era_borough_pct = era_borough_pct.reindex(ERA_ORDER)
    borough_order = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    existing_boroughs = [b for b in borough_order if b in era_borough_pct.columns]
    era_borough_pct = era_borough_pct[existing_boroughs]
    print(f"Using boroughs: {existing_boroughs}")
    
    print("\nBorough Distribution by Era (%):")
    print(era_borough_pct.round(1))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(ERA_ORDER))
    width = 0.15
    borough_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for i, borough in enumerate(existing_boroughs):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, era_borough_pct[borough], width, 
                      label=borough, color=borough_colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Era', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Borough Distribution by Era\n(How NYC Construction Activity Shifted Over Time)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_ORDER, fontsize=11)
    ax.legend(title='Borough', loc='upper right')
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'era_borough_breakdown.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_temporal_trend_summary(df: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    """Create summary visualization showing temporal trends."""
    print("\n" + "="*70)
    print("CREATING TEMPORAL TREND SUMMARY")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels
    df['avg_year'] = df['cluster'].map(CLUSTER_TO_YEAR)
    
    # Cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'LATITUDE': 'count',  # size
        'avg_year': 'first'
    }).rename(columns={'LATITUDE': 'size'})
    cluster_stats = cluster_stats.sort_values('avg_year')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cluster size by year
    colors = [ERA_COLORS[CLUSTER_TO_ERA[c]] for c in cluster_stats.index]
    axes[0, 0].bar(range(len(cluster_stats)), cluster_stats['size'], color=colors, alpha=0.8)
    axes[0, 0].set_xticks(range(len(cluster_stats)))
    axes[0, 0].set_xticklabels([f"C{c}\n({y})" for c, y in zip(cluster_stats.index, cluster_stats['avg_year'])], fontsize=9)
    axes[0, 0].set_xlabel('Cluster (Year)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Permits', fontsize=11)
    axes[0, 0].set_title('Cluster Size (Sorted by Avg Filing Year)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Era pie chart
    era_sizes = df['cluster'].map(CLUSTER_TO_ERA).value_counts().reindex(ERA_ORDER)
    axes[0, 1].pie(era_sizes, labels=ERA_ORDER, autopct='%1.1f%%',
                   colors=[ERA_COLORS[e] for e in ERA_ORDER], startangle=90)
    axes[0, 1].set_title('Permit Distribution by Era', fontsize=12, fontweight='bold')
    
    # 3. Permit Type evolution (if available)
    if 'Permit Type' in df.columns:
        df['era'] = df['cluster'].map(CLUSTER_TO_ERA)
        permit_era = df.groupby(['era', 'Permit Type']).size().unstack(fill_value=0)
        permit_era_pct = permit_era.div(permit_era.sum(axis=1), axis=0) * 100
        permit_era_pct = permit_era_pct.reindex(ERA_ORDER)
        
        # Top 3 permit types
        top_permits = df['Permit Type'].value_counts().head(3).index.tolist()
        x = np.arange(len(ERA_ORDER))
        width = 0.25
        
        for i, permit in enumerate(top_permits):
            if permit in permit_era_pct.columns:
                offset = (i - 1) * width
                axes[1, 0].bar(x + offset, permit_era_pct[permit], width, 
                              label=permit, alpha=0.8)
        
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(ERA_ORDER, fontsize=10)
        axes[1, 0].set_xlabel('Era', fontsize=11)
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 0].set_title('Permit Type Distribution by Era', fontsize=12, fontweight='bold')
        axes[1, 0].legend(title='Permit Type')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Text summary
    axes[1, 1].axis('off')
    summary_text = """
    KEY FINDINGS FROM K-MEANS CLUSTERING:
    
    1. TEMPORAL SEGMENTATION
       K-Means identified 3 distinct time periods:
       • Early (2004-2005): Diverse construction activities
       • Mid (2006-2008): Transition period
       • Late (2010-2011): Standardized operations
    
    2. CONSTRUCTION TRENDS
       • Equipment Work ratio: 42% → 46% (increasing)
       • Minor Alterations (A2): 56% → 62% (increasing)
       → NYC shifted from "building new" to "renovating"
    
    3. SPATIAL PATTERNS
       • Manhattan remains dominant (~40% across all eras)
       • Outer boroughs show consistent activity
       • Geographic distribution stable over time
    
    4. INSIGHT FOR URBAN PLANNING
       Clustering reveals NYC's urban maturation:
       More renovation, less new construction
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('NYC Construction Permits: Temporal Analysis Summary', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'temporal_trend_summary.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")


# ============================================================================
# ORIGINAL FUNCTIONS (kept for compatibility)
# ============================================================================

def plot_all_clusters_map(df: pd.DataFrame, labels: np.ndarray, output_dir: Path,
                          sample_for_plot: int = 50000):
    """Create map showing ALL clusters with different colors."""
    print("\n" + "="*70)
    print("CREATING ALL CLUSTERS MAP")
    print("="*70)
    
    min_len = min(len(labels), len(df))
    labels_matched = labels[:min_len]
    df = df.iloc[:min_len].copy()
    df['cluster'] = labels_matched
    
    if len(df) > sample_for_plot:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(len(df), sample_for_plot, replace=False)
        df_plot = df.iloc[idx]
    else:
        df_plot = df
    
    print(f"Plotting {len(df_plot):,} points...")
    
    n_clusters = len(np.unique(labels_matched))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    fig, ax = plt.subplots(figsize=(14, 16))
    
    for i, cluster_id in enumerate(sorted(df_plot['cluster'].unique())):
        mask = df_plot['cluster'] == cluster_id
        cluster_size = mask.sum()
        year = CLUSTER_TO_YEAR.get(cluster_id, '?')
        
        ax.scatter(
            df_plot.loc[mask, 'LONGITUDE'],
            df_plot.loc[mask, 'LATITUDE'],
            c=[colors[i]],
            s=3,
            alpha=0.4,
            label=f'Cluster {cluster_id} ({year}) n={cluster_size:,}'
        )
    
    ax.set_xlim(NYC_LON_MIN, NYC_LON_MAX)
    ax.set_ylim(NYC_LAT_MIN, NYC_LAT_MAX)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'NYC Construction Permits - All K-Means Clusters (k={n_clusters})', fontsize=14)
    ax.legend(loc='upper right', markerscale=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'geographic_all_clusters.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create geographic cluster visualizations with ERA analysis')
    parser.add_argument('--labels', type=str,
                       default='results/clusters/kmeans_mpi/labels_k10_np1.npy',
                       help='Path to labels .npy file')
    parser.add_argument('--output', type=str,
                       default='results/visualizations/geographic',
                       help='Output directory')
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = BASE_DIR / labels_path
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GEOGRAPHIC VISUALIZATION WITH ERA ANALYSIS")
    print("="*70)
    
    print(f"\nLoading labels from {labels_path}...")
    labels = np.load(labels_path)
    print(f"Loaded {len(labels):,} labels, {len(np.unique(labels))} clusters")
    
    df = load_data_with_coords(sample_size=len(labels))
    
    # Original visualization
    plot_all_clusters_map(df, labels, output_dir)
    
    # NEW: Era-based visualizations
    plot_by_era_comparison(df, labels, output_dir)
    plot_era_density_heatmaps(df, labels, output_dir)
    plot_all_eras_overlay(df, labels, output_dir)
    plot_era_borough_breakdown(df, labels, output_dir)
    plot_temporal_trend_summary(df, labels, output_dir)
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print("\nGenerated files:")
    print("  - geographic_all_clusters.png")
    print("  - geographic_by_era.png           <- NEW: 3-panel era comparison")
    print("  - geographic_density_by_era.png   <- NEW: density heatmaps by era")
    print("  - geographic_all_eras_overlay.png <- NEW: all eras on one map")
    print("  - era_borough_breakdown.png       <- NEW: borough % by era")
    print("  - temporal_trend_summary.png      <- NEW: summary dashboard")


if __name__ == "__main__":
    main()
