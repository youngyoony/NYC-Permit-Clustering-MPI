"""
Cluster Interpretation for Business Insights
Answers: "What types of construction projects exist?"

TODO 6: Cluster Interpretation
- Analyze cluster characteristics by permit type, cost, borough
- Generate business-friendly summaries
- Create visualization of cluster profiles
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config


def load_data_and_labels(k: int, method: str = 'kmeans', nprocs: int = None):
    """
    Load data, metadata, and cluster labels.
    
    Args:
        k: Number of clusters
        method: 'kmeans' or 'kmeans_mpi' or 'hierarchical'
        nprocs: Number of processes (for MPI methods)
        
    Returns:
        Tuple of (X, df_meta, labels)
    """
    print(f"\nLoading data and {method} labels (k={k})...")
    
    # Load data
    X = np.load(config.PROCESSED_DATA_PATH)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
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
    
    print(f"Data shape: {X.shape}")
    print(f"Metadata shape: {df_meta.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Add cluster labels to metadata
    df_meta['cluster'] = labels
    
    return X, df_meta, labels


def analyze_cluster_business_profiles(df_meta: pd.DataFrame, output_dir: Path):
    """
    Analyze each cluster from a business perspective.
    
    Business Questions:
    - What types of projects (permit types)?
    - What's the average cost?
    - Which boroughs?
    - Residential vs commercial?
    - Project sizes?
    
    Args:
        df_meta: Metadata with cluster labels
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("BUSINESS-FOCUSED CLUSTER ANALYSIS")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_clusters = sorted(df_meta['cluster'].unique())
    n_clusters = len(unique_clusters)
    
    print(f"Number of clusters: {n_clusters}")
    
    # Aggregate statistics per cluster
    cluster_profiles = []
    
    for cluster_id in unique_clusters:
        cluster_data = df_meta[df_meta['cluster'] == cluster_id]
        
        profile = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': 100 * len(cluster_data) / len(df_meta)
        }
        
        # Cost analysis
        if 'Job Cost' in cluster_data.columns:
            profile['avg_cost'] = cluster_data['Job Cost'].mean()
            profile['median_cost'] = cluster_data['Job Cost'].median()
            profile['total_cost'] = cluster_data['Job Cost'].sum()
        
        # Permit type
        if 'Permit Type' in cluster_data.columns:
            top_permit = cluster_data['Permit Type'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
            top_permit_pct = 100 * (cluster_data['Permit Type'] == top_permit).sum() / len(cluster_data)
            profile['top_permit_type'] = top_permit
            profile['top_permit_pct'] = top_permit_pct
        
        # Borough
        if 'Borough' in cluster_data.columns:
            top_borough = cluster_data['Borough'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
            top_borough_pct = 100 * (cluster_data['Borough'] == top_borough).sum() / len(cluster_data)
            profile['top_borough'] = top_borough
            profile['top_borough_pct'] = top_borough_pct
        
        # Residential flag
        if 'Residential' in cluster_data.columns:
            profile['residential_pct'] = 100 * cluster_data['Residential'].mean()
        
        # Work type
        if 'Work Type' in cluster_data.columns:
            top_work = cluster_data['Work Type'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
            profile['top_work_type'] = top_work
        
        # Owner type
        if 'Owner Type' in cluster_data.columns:
            top_owner = cluster_data['Owner Type'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
            profile['top_owner_type'] = top_owner
        
        # Proposed dwelling units
        if 'Proposed Dwelling Units' in cluster_data.columns:
            profile['avg_dwelling_units'] = cluster_data['Proposed Dwelling Units'].mean()
        
        # Zoning sqft
        if 'Proposed Zoning Sqft' in cluster_data.columns:
            profile['avg_zoning_sqft'] = cluster_data['Proposed Zoning Sqft'].mean()
        
        cluster_profiles.append(profile)
        
        # Print cluster summary
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}: Business Profile")
        print(f"{'='*80}")
        print(f"Size: {profile['size']:,} permits ({profile['percentage']:.1f}% of total)")
        
        if 'avg_cost' in profile:
            print(f"\n📊 FINANCIAL IMPACT:")
            print(f"  Average cost: ${profile['avg_cost']:,.0f}")
            print(f"  Median cost: ${profile['median_cost']:,.0f}")
            print(f"  Total value: ${profile['total_cost']:,.0f}")
        
        if 'top_permit_type' in profile:
            print(f"\n📝 PROJECT TYPE:")
            print(f"  Dominant permit: {profile['top_permit_type']} ({profile['top_permit_pct']:.1f}%)")
        
        if 'top_work_type' in profile:
            print(f"  Work type: {profile['top_work_type']}")
        
        if 'top_borough' in profile:
            print(f"\n🗺️  GEOGRAPHY:")
            print(f"  Primary borough: {profile['top_borough']} ({profile['top_borough_pct']:.1f}%)")
        
        if 'residential_pct' in profile:
            print(f"\n🏠 RESIDENTIAL vs COMMERCIAL:")
            print(f"  Residential: {profile['residential_pct']:.1f}%")
            print(f"  Commercial: {100 - profile['residential_pct']:.1f}%")
        
        if 'avg_dwelling_units' in profile:
            print(f"\n📏 PROJECT SCALE:")
            print(f"  Avg dwelling units: {profile['avg_dwelling_units']:.1f}")
        
        if 'avg_zoning_sqft' in profile:
            print(f"  Avg zoning sqft: {profile['avg_zoning_sqft']:,.0f}")
    
    # Save profiles
    profiles_df = pd.DataFrame(cluster_profiles)
    profiles_df.to_csv(output_dir / 'cluster_business_profiles.csv', index=False)
    print(f"\nSaved business profiles to {output_dir / 'cluster_business_profiles.csv'}")
    
    return profiles_df


def create_business_visualizations(df_meta: pd.DataFrame, profiles_df: pd.DataFrame, 
                                   output_dir: Path):
    """
    Create business-focused visualizations.
    """
    print("\n" + "="*80)
    print("CREATING BUSINESS VISUALIZATIONS")
    print("="*80)
    
    n_clusters = len(profiles_df)
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cluster sizes
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(profiles_df['cluster'].astype(str), profiles_df['size'], color='steelblue')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Permits')
    ax.set_title('Cluster Sizes')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(profiles_df.iterrows()):
        ax.text(i, row['size'], f"{row['percentage']:.1f}%", 
               ha='center', va='bottom', fontsize=9)
    
    # 2. Average cost per cluster
    if 'avg_cost' in profiles_df.columns:
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(profiles_df['cluster'].astype(str), profiles_df['avg_cost'] / 1000, 
               color='green')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Average Cost ($1000s)')
        ax.set_title('Average Project Cost by Cluster')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Total value per cluster
    if 'total_cost' in profiles_df.columns:
        ax = fig.add_subplot(gs[0, 2])
        ax.bar(profiles_df['cluster'].astype(str), profiles_df['total_cost'] / 1e9,
               color='orange')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Total Value ($Billions)')
        ax.set_title('Total Economic Impact by Cluster')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Permit type distribution
    if 'Permit Type' in df_meta.columns:
        ax = fig.add_subplot(gs[1, :])
        permit_counts = pd.crosstab(df_meta['cluster'], df_meta['Permit Type'], normalize='index') * 100
        permit_counts = permit_counts[permit_counts.sum().sort_values(ascending=False).head(10).index]
        permit_counts.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_xlabel('Permit Type')
        ax.set_ylabel('Percentage')
        ax.set_title('Permit Type Distribution by Cluster (Top 10 Types)')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Borough distribution
    if 'Borough' in df_meta.columns:
        ax = fig.add_subplot(gs[2, 0])
        borough_counts = pd.crosstab(df_meta['cluster'], df_meta['Borough'], normalize='index') * 100
        borough_counts.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage')
        ax.set_title('Borough Distribution by Cluster')
        ax.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Residential vs Commercial
    if 'residential_pct' in profiles_df.columns:
        ax = fig.add_subplot(gs[2, 1])
        residential_pct = profiles_df['residential_pct'].values
        commercial_pct = 100 - residential_pct
        
        x = np.arange(len(profiles_df))
        width = 0.7
        
        ax.bar(x, residential_pct, width, label='Residential', color='lightblue')
        ax.bar(x, commercial_pct, width, bottom=residential_pct, 
               label='Commercial', color='lightcoral')
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Percentage')
        ax.set_title('Residential vs Commercial Mix')
        ax.set_xticks(x)
        ax.set_xticklabels(profiles_df['cluster'].astype(str))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 7. Average dwelling units
    if 'avg_dwelling_units' in profiles_df.columns:
        ax = fig.add_subplot(gs[2, 2])
        ax.bar(profiles_df['cluster'].astype(str), profiles_df['avg_dwelling_units'],
               color='purple')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Average Dwelling Units')
        ax.set_title('Project Scale by Cluster')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / 'business_dashboard.png', dpi=config.PLOT_DPI, 
                bbox_inches='tight')
    plt.close()
    
    print(f"Saved business dashboard to {output_dir / 'business_dashboard.png'}")


def generate_business_summary(profiles_df: pd.DataFrame, output_dir: Path):
    """
    Generate text summary for presentation.
    """
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS SUMMARY")
    print("="*80)
    
    summary = []
    summary.append("="*80)
    summary.append("CLUSTER INTERPRETATION: BUSINESS INSIGHTS")
    summary.append("="*80)
    summary.append("")
    summary.append(f"Total clusters identified: {len(profiles_df)}")
    summary.append("")
    
    # Identify key clusters
    if 'avg_cost' in profiles_df.columns:
        high_value_cluster = profiles_df.loc[profiles_df['avg_cost'].idxmax()]
        summary.append(f"💰 HIGHEST VALUE CLUSTER: Cluster {high_value_cluster['cluster']}")
        summary.append(f"   - Average cost: ${high_value_cluster['avg_cost']:,.0f}")
        if 'top_permit_type' in high_value_cluster:
            summary.append(f"   - Primary type: {high_value_cluster['top_permit_type']}")
        if 'top_borough' in high_value_cluster:
            summary.append(f"   - Location: {high_value_cluster['top_borough']}")
        summary.append("")
    
    if 'size' in profiles_df.columns:
        largest_cluster = profiles_df.loc[profiles_df['size'].idxmax()]
        summary.append(f"📊 LARGEST CLUSTER: Cluster {largest_cluster['cluster']}")
        summary.append(f"   - Size: {largest_cluster['size']:,} permits ({largest_cluster['percentage']:.1f}%)")
        if 'top_permit_type' in largest_cluster:
            summary.append(f"   - Primary type: {largest_cluster['top_permit_type']}")
        summary.append("")
    
    # Resource allocation recommendations
    summary.append("="*80)
    summary.append("RESOURCE ALLOCATION RECOMMENDATIONS")
    summary.append("="*80)
    summary.append("")
    
    for idx, row in profiles_df.iterrows():
        summary.append(f"Cluster {row['cluster']}:")
        summary.append(f"  • Volume: {row['size']:,} permits ({row['percentage']:.1f}%)")
        
        if 'avg_cost' in row:
            summary.append(f"  • Average value: ${row['avg_cost']:,.0f}")
        
        if 'top_borough' in row:
            summary.append(f"  • Primary location: {row['top_borough']}")
        
        # Recommendation based on characteristics
        if 'avg_cost' in row and row['avg_cost'] > profiles_df['avg_cost'].median():
            summary.append(f"  → Recommendation: Assign senior inspectors (high-value projects)")
        elif row['percentage'] > 20:
            summary.append(f"  → Recommendation: Increase inspector headcount (high volume)")
        else:
            summary.append(f"  → Recommendation: Standard processing workflow")
        
        summary.append("")
    
    # Save summary
    summary_text = "\n".join(summary)
    with open(output_dir / 'business_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSaved business summary to {output_dir / 'business_summary.txt'}")


def main():
    """Main cluster interpretation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Business-focused cluster interpretation')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'kmeans_mpi', 'hierarchical'],
                       help='Clustering method')
    parser.add_argument('--nprocs', type=int, default=None,
                       help='Number of processes (for MPI methods)')
    
    args = parser.parse_args()
    
    # Load data
    X, df_meta, labels = load_data_and_labels(args.k, args.method, args.nprocs)
    
    # Create output directory
    output_dir = config.RESULTS_DIR / "cluster_interpretation" / f"{args.method}_k{args.k}"
    
    # Analyze business profiles
    profiles_df = analyze_cluster_business_profiles(df_meta, output_dir)
    
    # Create visualizations
    create_business_visualizations(df_meta, profiles_df, output_dir)
    
    # Generate summary
    generate_business_summary(profiles_df, output_dir)
    
    print("\n" + "="*80)
    print("CLUSTER INTERPRETATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
