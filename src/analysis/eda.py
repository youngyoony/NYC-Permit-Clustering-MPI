"""
Exploratory Data Analysis (EDA) Module
Generate visualizations and statistical summaries of the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import config


# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create histograms for all numeric columns.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating numeric distribution plots...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Latitude', 'Longitude', 'Job']]
    
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Regular histogram
        axes[0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {col}')
        axes[0].grid(True, alpha=0.3)
        
        # Log-scale histogram
        positive_vals = df[col].dropna()
        positive_vals = positive_vals[positive_vals > 0]
        if len(positive_vals) > 0:
            axes[1].hist(positive_vals, bins=50, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Distribution of {col} (log scale)')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_{col.replace(" ", "_").replace("/", "_")}.png', 
                   dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
    print(f"  Saved {len(numeric_cols)} distribution plots")


def plot_categorical_distributions(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create bar charts for categorical columns.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating categorical distribution plots...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Limit to reasonable number of unique values
    categorical_cols = [col for col in categorical_cols if df[col].nunique() < 50]
    
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {col} (Top 20)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'cat_{col.replace(" ", "_").replace("/", "_")}.png',
                   dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(categorical_cols)} categorical distribution plots")


def plot_time_series(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Plot permit issuance over time.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating time series plots...")
    
    date_cols = [col for col in df.columns if 'Date' in col]
    
    for date_col in date_cols:
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Aggregate by month
            df_temp = df[[date_col]].copy()
            df_temp['year_month'] = df_temp[date_col].dt.to_period('M')
            counts = df_temp['year_month'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(14, 6))
            counts.plot(kind='line', ax=ax, linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Year-Month')
            ax.set_ylabel('Number of Permits')
            ax.set_title(f'Permits Over Time ({date_col})')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'timeseries_{date_col.replace(" ", "_")}.png',
                       dpi=config.PLOT_DPI, bbox_inches='tight')
            plt.close()
    
    print(f"  Saved {len(date_cols)} time series plots")


def plot_spatial_distribution(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create spatial visualizations: scatter plots and density heatmaps.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating spatial distribution plots...")
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("  Skipping: Latitude/Longitude columns not found")
        return
    
    # Sample for plotting (full dataset is too dense)
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
    
    # 1. Basic scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(df_sample['Longitude'], df_sample['Latitude'], 
              alpha=0.1, s=1, c='blue')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Spatial Distribution of Permits (n={sample_size:,})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_scatter.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # 2. Colored by borough
    if 'Borough' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        boroughs = df_sample['Borough'].unique()
        colors = sns.color_palette("husl", len(boroughs))
        
        for borough, color in zip(boroughs, colors):
            mask = df_sample['Borough'] == borough
            ax.scatter(df_sample.loc[mask, 'Longitude'], 
                      df_sample.loc[mask, 'Latitude'],
                      alpha=0.3, s=2, c=[color], label=borough)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Permits by Borough (n={sample_size:,})')
        ax.legend(loc='best', markerscale=5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_by_borough.png', dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    # 3. Hexbin density plot
    fig, ax = plt.subplots(figsize=(12, 10))
    hexbin = ax.hexbin(df_sample['Longitude'], df_sample['Latitude'],
                       gridsize=50, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Permit Density Heatmap')
    plt.colorbar(hexbin, ax=ax, label='Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_density.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print("  Saved 3 spatial distribution plots")


def plot_correlations(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Create correlation heatmap for numeric features.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating correlation heatmap...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Latitude', 'Longitude', 'Job']]
    
    if len(numeric_cols) < 2:
        print("  Skipping: Not enough numeric columns")
        return
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"\n  Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
        for col1, col2, corr in high_corr_pairs[:10]:
            print(f"    {col1} <-> {col2}: {corr:.3f}")
    
    print("  Saved correlation heatmap")


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path = None) -> None:
    """
    Generate and save summary statistics to CSV.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save statistics
    """
    if output_dir is None:
        output_dir = config.EDA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating summary statistics...")
    
    # Numeric summary
    numeric_summary = df.describe().T
    numeric_summary.to_csv(output_dir / 'summary_numeric.csv')
    print(f"  Saved numeric summary ({len(numeric_summary)} features)")
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append({
                'Feature': col,
                'Unique': df[col].nunique(),
                'Most_Common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'Most_Common_Count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                'Missing': df[col].isnull().sum(),
                'Missing_Pct': 100 * df[col].isnull().sum() / len(df)
            })
        
        cat_summary_df = pd.DataFrame(cat_summary)
        cat_summary_df.to_csv(output_dir / 'summary_categorical.csv', index=False)
        print(f"  Saved categorical summary ({len(categorical_cols)} features)")


def main():
    """
    Run complete EDA pipeline.
    """
    import time
    from data_prep import load_raw_data, clean_data
    
    start_time = time.time()
    
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Load and clean data
    print("\nLoading data...")
    df = load_raw_data()
    df = clean_data(df)
    
    # Generate all visualizations
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)
    plot_time_series(df)
    plot_spatial_distribution(df)
    plot_correlations(df)
    generate_summary_statistics(df)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EDA COMPLETE")
    print(f"Output directory: {config.EDA_DIR}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
