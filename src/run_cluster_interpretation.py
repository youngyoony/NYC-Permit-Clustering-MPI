#!/usr/bin/env python3
"""
Cluster Interpretation Script
Analyzes cluster characteristics using original data + cluster labels.

Run on SeaWulf:
    python run_cluster_interpretation.py --labels results/clusters/kmeans_mpi/labels_k10_np1.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

# ============================================================================
# CONFIGURATION - Team Project paths
# ============================================================================
BASE_DIR = Path("/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RAW_DATA_PATH = DATA_DIR / "DOB_Permit_Issuance_merged.csv"

# Feature name mappings for interpretation
PERMIT_TYPE_NAMES = {
    'DM': 'Demolition',
    'EQ': 'Equipment',
    'EW': 'Equipment Work',
    'FO': 'Foundation',
    'NB': 'New Building',
    'PL': 'Plumbing',
    'SG': 'Sign',
    'AL': 'Alteration'
}

JOB_TYPE_NAMES = {
    'A1': 'Alteration Type 1 (Major)',
    'A2': 'Alteration Type 2 (Minor)',
    'A3': 'Alteration Type 3 (Cosmetic)',
    'DM': 'Demolition',
    'NB': 'New Building',
    'SG': 'Sign'
}

BOROUGH_NAMES = {
    1: 'Manhattan',
    2: 'Bronx',
    3: 'Brooklyn',
    4: 'Queens',
    5: 'Staten Island'
}


def load_raw_data(sample_size=None):
    """Load original CSV data."""
    print(f"\nLoading raw data from {RAW_DATA_PATH}...")
    
    usecols = [
        'BOROUGH', 'LATITUDE', 'LONGITUDE', 'COUNCIL_DISTRICT', 'CENSUS_TRACT',
        'Permit Type', 'Job Type', 'Permit Status', 'Residential',
        'Filing Date', 'Issuance Date', 'Bldg Type', 'NTA_NAME',
        'Zip Code', 'Community Board'
    ]
    
    try:
        df = pd.read_csv(RAW_DATA_PATH, usecols=usecols, low_memory=False)
    except ValueError:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    print(f"Loaded {len(df):,} rows")
    
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size]
        print(f"Using first {sample_size:,} rows to match labels")
    
    return df


def interpret_clusters(df: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    """Generate semantic interpretation for each cluster."""
    print("\n" + "="*70)
    print("CLUSTER INTERPRETATION")
    print("="*70)
    
    if len(labels) != len(df):
        print(f"WARNING: labels ({len(labels)}) != data ({len(df)})")
        min_len = min(len(labels), len(df))
        labels = labels[:min_len]
        df = df.iloc[:min_len]
    
    df = df.copy()
    df['cluster'] = labels
    
    n_clusters = len(np.unique(labels))
    total_samples = len(df)
    
    results = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        n_samples = len(cluster_data)
        pct = n_samples / total_samples * 100
        
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id}: {n_samples:,} samples ({pct:.1f}%)")
        print(f"{'='*50}")
        
        result = {
            'Cluster': cluster_id,
            'Size': n_samples,
            'Percentage': f"{pct:.1f}%"
        }
        
        # Borough Distribution
        if 'BOROUGH' in cluster_data.columns:
            borough_counts = cluster_data['BOROUGH'].value_counts()
            if len(borough_counts) > 0:
                top_borough_code = borough_counts.index[0]
                top_borough = BOROUGH_NAMES.get(top_borough_code, str(top_borough_code))
                top_borough_pct = borough_counts.iloc[0] / n_samples * 100
                result['Top_Borough'] = top_borough
                result['Borough_Pct'] = f"{top_borough_pct:.1f}%"
                print(f"  Top Borough: {top_borough} ({top_borough_pct:.1f}%)")
        
        # Permit Type
        if 'Permit Type' in cluster_data.columns:
            permit_counts = cluster_data['Permit Type'].value_counts()
            if len(permit_counts) > 0:
                top_permit = permit_counts.index[0]
                top_permit_name = PERMIT_TYPE_NAMES.get(top_permit, top_permit)
                top_permit_pct = permit_counts.iloc[0] / n_samples * 100
                result['Top_Permit'] = top_permit_name
                result['Permit_Pct'] = f"{top_permit_pct:.1f}%"
                print(f"  Top Permit Type: {top_permit_name} ({top_permit_pct:.1f}%)")
        
        # Job Type
        if 'Job Type' in cluster_data.columns:
            job_counts = cluster_data['Job Type'].value_counts()
            if len(job_counts) > 0:
                top_job = job_counts.index[0]
                top_job_name = JOB_TYPE_NAMES.get(top_job, top_job)
                top_job_pct = job_counts.iloc[0] / n_samples * 100
                result['Top_Job'] = top_job_name
                result['Job_Pct'] = f"{top_job_pct:.1f}%"
                print(f"  Top Job Type: {top_job_name} ({top_job_pct:.1f}%)")
        
        # Residential Ratio
        if 'Residential' in cluster_data.columns:
            res_counts = cluster_data['Residential'].value_counts()
            yes_count = res_counts.get('YES', 0)
            no_count = res_counts.get('NO', 0)
            known_total = yes_count + no_count
            if known_total > 0:
                res_pct = yes_count / known_total * 100
                res_label = 'Residential' if res_pct > 50 else 'Commercial/Mixed'
                result['Residential_Pct'] = f"{res_pct:.1f}%"
                result['Property_Type'] = res_label
                print(f"  Residential: {res_pct:.1f}% -> {res_label}")
        
        # Permit Status
        if 'Permit Status' in cluster_data.columns:
            status_counts = cluster_data['Permit Status'].value_counts()
            if len(status_counts) > 0:
                top_status = status_counts.index[0]
                top_status_pct = status_counts.iloc[0] / n_samples * 100
                result['Top_Status'] = top_status
                print(f"  Top Permit Status: {top_status} ({top_status_pct:.1f}%)")
        
        # Geographic Center
        if 'LATITUDE' in cluster_data.columns and 'LONGITUDE' in cluster_data.columns:
            lat_mean = cluster_data['LATITUDE'].mean()
            lon_mean = cluster_data['LONGITUDE'].mean()
            result['Lat_Center'] = f"{lat_mean:.4f}"
            result['Lon_Center'] = f"{lon_mean:.4f}"
            print(f"  Geographic Center: ({lat_mean:.4f}, {lon_mean:.4f})")
        
        # Temporal Features
        if 'Filing Date' in cluster_data.columns:
            try:
                filing_dates = pd.to_datetime(cluster_data['Filing Date'], errors='coerce')
                avg_year = filing_dates.dt.year.mean()
                if not pd.isna(avg_year):
                    result['Avg_Filing_Year'] = f"{avg_year:.0f}"
                    print(f"  Avg Filing Year: {avg_year:.0f}")
            except:
                pass
        
        # Building Type
        if 'Bldg Type' in cluster_data.columns:
            bldg_counts = cluster_data['Bldg Type'].value_counts()
            if len(bldg_counts) > 0:
                top_bldg = bldg_counts.index[0]
                print(f"  Top Building Type: {top_bldg}")
        
        # Generate Cluster Label
        size_label = "Large" if pct >= 15 else ("Medium" if pct >= 5 else "Small")
        property_label = result.get('Property_Type', 'Mixed')
        permit_label = result.get('Top_Permit', 'Various')
        borough_label = result.get('Top_Borough', '')
        
        cluster_label = f"{size_label} {property_label} - {permit_label}"
        if borough_label:
            cluster_label += f" ({borough_label})"
        
        result['Label'] = cluster_label
        print(f"\n  -> CLUSTER LABEL: {cluster_label}")
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'cluster_interpretation.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    
    # Print markdown table
    print("\n" + "="*70)
    print("MARKDOWN TABLE (Copy for Presentation)")
    print("="*70)
    print("\n| Cluster | Size | % | Borough | Permit Type | Residential | Label |")
    print("|---------|------|---|---------|-------------|-------------|-------|")
    for r in results:
        print(f"| {r['Cluster']} | {r['Size']:,} | {r['Percentage']} | "
              f"{r.get('Top_Borough', 'N/A')} | {r.get('Top_Permit', 'N/A')} | "
              f"{r.get('Residential_Pct', 'N/A')} | {r.get('Label', '')} |")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Interpret K-Means clusters')
    parser.add_argument('--labels', type=str, 
                       default='results/clusters/kmeans_mpi/labels_k10_np1.npy',
                       help='Path to labels .npy file')
    parser.add_argument('--output', type=str,
                       default='results/clusters/kmeans_mpi',
                       help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (use if labels are for subset)')
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = BASE_DIR / labels_path
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    
    print(f"\nLoading labels from {labels_path}...")
    labels = np.load(labels_path)
    print(f"Loaded {len(labels):,} labels, {len(np.unique(labels))} clusters")
    
    df = load_raw_data(sample_size=len(labels))
    results = interpret_clusters(df, labels, output_dir)
    
    print("\n" + "="*70)
    print("CLUSTER INTERPRETATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
