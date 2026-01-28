"""
Data Preparation Module for NYC DOB Permit Issuance Dataset
Handles loading, cleaning, feature engineering, and encoding.

FIXED: Now applies FEATURES_FOR_CLUSTERING selection at the end!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
import config


def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load raw CSV data from specified path.
    Supports both single CSV file and split files (data_0.csv to data_9.csv).
    
    Args:
        path: Path to CSV file. If None, uses config.RAW_DATA_PATH
        
    Returns:
        DataFrame with raw data
    """
    if path is None:
        path = config.RAW_DATA_PATH
    
    path = Path(path)
    
    # Check if single file exists
    if path.exists():
        print(f"Loading data from {path}...")
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    # Check for split files (data_0.csv to data_9.csv)
    data_dir = path.parent
    split_files = sorted(data_dir.glob("data_*.csv"))
    
    if split_files:
        print(f"Found {len(split_files)} split data files in {data_dir}")
        print("Merging split CSV files...")
        
        dfs = []
        total_rows = 0
        for i, f in enumerate(split_files):
            print(f"  Loading {f.name}...", end=" ")
            chunk_df = pd.read_csv(f, low_memory=False)
            print(f"{len(chunk_df):,} rows")
            dfs.append(chunk_df)
            total_rows += len(chunk_df)
        
        print(f"Concatenating {len(dfs)} files...")
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total: {len(df):,} rows and {len(df.columns)} columns")
        
        # Optional: Save merged file for future use
        merged_path = data_dir / "DOB_Permit_Issuance_merged.csv"
        print(f"Saving merged file to {merged_path}...")
        df.to_csv(merged_path, index=False)
        print("Merged file saved!")
        
        return df
    
    raise FileNotFoundError(
        f"Could not find data file at {path} or split files (data_*.csv) in {data_dir}"
    )


def basic_explore(df: pd.DataFrame) -> None:
    """
    Print basic exploratory information about the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*80)
    print("BASIC DATA EXPLORATION")
    print("="*80)
    
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percent': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False).head(20))
    
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Total duplicates: {duplicates:,} ({100*duplicates/len(df):.2f}%)")
    
    print("\n--- Memory Usage ---")
    print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n--- Column Names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: handle missing values, remove outliers, fix data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)
    
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df):,} duplicate rows")
    
    # 2. Geographic filtering (NYC bounds)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print("\nFiltering by geographic bounds...")
        before = len(df)
        df = df[
            (df['Latitude'].between(config.NYC_LAT_MIN, config.NYC_LAT_MAX)) &
            (df['Longitude'].between(config.NYC_LON_MIN, config.NYC_LON_MAX))
        ]
        print(f"Removed {before - len(df):,} rows outside NYC bounds")
    
    # 3. Cost outliers
    if 'Job Cost' in df.columns:
        print("\nHandling cost outliers...")
        df['Job Cost'] = pd.to_numeric(df['Job Cost'], errors='coerce')
        
        # Remove negative costs
        before = len(df)
        df = df[df['Job Cost'] >= config.MIN_COST]
        print(f"Removed {before - len(df):,} rows with negative costs")
        
        # Cap extreme costs
        cost_cap = df['Job Cost'].quantile(config.OUTLIER_PERCENTILE / 100)
        extreme_count = (df['Job Cost'] > cost_cap).sum()
        df.loc[df['Job Cost'] > cost_cap, 'Job Cost'] = cost_cap
        print(f"Capped {extreme_count:,} extreme cost values at ${cost_cap:,.0f}")
    
    # 4. Parse datetime columns
    print("\nParsing datetime columns...")
    datetime_cols = [col for col in df.columns if 'Date' in col or 'date' in col.lower()]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"  {col}: {null_count:,} unparseable dates set to NaT")
    
    # 5. Handle missing values in key columns
    print("\nHandling missing values...")
    
    # Numeric columns: impute with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            null_count = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            print(f"  {col}: filled {null_count:,} nulls with median={median_val:.2f}")
    
    # Categorical columns: impute with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            df[col].fillna('Unknown', inplace=True)
            print(f"  {col}: filled {null_count:,} nulls with 'Unknown'")
    
    # 6. Remove rows with critical missing values
    critical_cols = ['Latitude', 'Longitude', 'Borough']
    before = len(df)
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    print(f"\nRemoved {before - len(df):,} rows with missing critical values")
    
    print(f"\nFinal dataset: {len(df):,} rows ({100*len(df)/initial_rows:.1f}% retained)")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing ones.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df = df.copy()
    
    # 1. Temporal features from Filing Date
    print("\nExtracting temporal features...")
    if 'Filing Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Filing Date']):
        df['Filing_Year'] = df['Filing Date'].dt.year
        df['Filing_Month'] = df['Filing Date'].dt.month
        df['Filing_Quarter'] = df['Filing Date'].dt.quarter
        print("  Created Filing_Year, Filing_Month, Filing_Quarter")
    
    # 2. Permit Age (days from filing to now or to issuance)
    if 'Filing Date' in df.columns and 'Issuance Date' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['Filing Date']) and \
           pd.api.types.is_datetime64_any_dtype(df['Issuance Date']):
            df['Permit_Age_Days'] = (df['Issuance Date'] - df['Filing Date']).dt.days
            df['Permit_Age_Days'] = df['Permit_Age_Days'].clip(lower=0)
            df['Permit_Age_Days'].fillna(df['Permit_Age_Days'].median(), inplace=True)
            print("  Created Permit_Age_Days")
    
    # 3. Geographic features - ensure LATITUDE, LONGITUDE are uppercase
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['LATITUDE'] = df['Latitude']
        df['LONGITUDE'] = df['Longitude']
        print("  Created LATITUDE, LONGITUDE")
    
    # 4. Ensure COUNCIL_DISTRICT, CENSUS_TRACT exist
    if 'Council District' in df.columns:
        df['COUNCIL_DISTRICT'] = pd.to_numeric(df['Council District'], errors='coerce').fillna(0)
        print("  Created COUNCIL_DISTRICT")
    
    if 'Census Tract' in df.columns:
        df['CENSUS_TRACT'] = pd.to_numeric(df['Census Tract'], errors='coerce').fillna(0)
        print("  Created CENSUS_TRACT")
    
    print(f"\nTotal features after engineering: {len(df.columns)}")
    
    return df


def encode_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, list]:
    """
    Encode features for clustering: one-hot encode categoricals, 
    standardize numerics, then SELECT only FEATURES_FOR_CLUSTERING.
    
    FIXED: Now applies feature selection at the end!
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Tuple of (X: encoded feature matrix, df_meta: metadata, feature_names: list)
    """
    print("\n" + "="*80)
    print("ENCODING FEATURES")
    print("="*80)
    
    # Save metadata (for interpretation later)
    meta_cols = ['Latitude', 'Longitude', 'Borough']
    if 'Job' in df.columns:
        meta_cols.append('Job')
    df_meta = df[[col for col in meta_cols if col in df.columns]].copy()
    
    feature_matrices = []
    feature_names = []
    
    # =====================================================
    # 1. Numeric features (INCLUDING LATITUDE, LONGITUDE)
    # =====================================================
    print("\nProcessing numeric features...")
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID-like columns but KEEP geographic columns
    exclude_numeric = ['Job']  # Only exclude Job ID
    numeric_cols = [col for col in numeric_cols if col not in exclude_numeric]
    
    if numeric_cols:
        X_numeric = df[numeric_cols].values.astype(np.float64)
        
        # Handle NaN/Inf before scaling
        X_numeric = np.nan_to_num(X_numeric, nan=0.0, posinf=1e10, neginf=-1e10)
        
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        
        feature_matrices.append(X_numeric_scaled)
        feature_names.extend(numeric_cols)
        print(f"  Standardized {len(numeric_cols)} numeric features")
        print(f"  Numeric features: {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"  Numeric features: {numeric_cols}")
    
    # =====================================================
    # 2. Categorical features (one-hot encoding)
    # =====================================================
    print("\nProcessing categorical features...")
    
    # These are the categorical columns we want to one-hot encode
    categorical_cols_to_encode = ['Permit Type', 'Job Type', 'Permit Status', 'Residential']
    categorical_cols = [col for col in categorical_cols_to_encode if col in df.columns]
    
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=50)
        X_categorical = encoder.fit_transform(df[categorical_cols])
        
        feature_matrices.append(X_categorical)
        
        # Get feature names (e.g., "Permit Type_EQ", "Job Type_A2")
        cat_feature_names = []
        for i, col in enumerate(categorical_cols):
            categories = encoder.categories_[i]
            cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
        feature_names.extend(cat_feature_names)
        
        print(f"  One-hot encoded {len(categorical_cols)} categorical features")
        print(f"  Created {X_categorical.shape[1]} binary features")
    
    # =====================================================
    # 3. Combine all features
    # =====================================================
    print("\nCombining all features...")
    if len(feature_matrices) == 0:
        raise ValueError("No features were encoded!")
    
    X_all = np.hstack(feature_matrices)
    
    print(f"\nFull feature matrix shape: {X_all.shape}")
    print(f"  Total features before selection: {len(feature_names)}")
    
    # =====================================================
    # 4. FEATURE SELECTION - Apply FEATURES_FOR_CLUSTERING
    # =====================================================
    print("\n" + "="*80)
    print("APPLYING FEATURE SELECTION")
    print("="*80)
    
    print(f"\nTarget features: {config.FEATURES_FOR_CLUSTERING}")
    print(f"Number of target features: {len(config.FEATURES_FOR_CLUSTERING)}")
    
    # Find indices of selected features
    selected_indices = []
    selected_names = []
    missing_features = []
    
    for target_feature in config.FEATURES_FOR_CLUSTERING:
        if target_feature in feature_names:
            idx = feature_names.index(target_feature)
            selected_indices.append(idx)
            selected_names.append(target_feature)
        else:
            missing_features.append(target_feature)
    
    if missing_features:
        print(f"\n⚠ WARNING: Missing features: {missing_features}")
        print(f"  Available features: {feature_names[:20]}..." if len(feature_names) > 20 else f"  Available features: {feature_names}")
    
    # Select only the target features
    X = X_all[:, selected_indices]
    feature_names_final = selected_names
    
    print(f"\n✓ Selected {len(selected_indices)} features")
    print(f"  Final X shape: {X.shape}")
    print(f"  Selected features: {feature_names_final}")
    
    # =====================================================
    # 5. Final cleanup
    # =====================================================
    print("\nCleaning final feature matrix...")
    
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    if nan_count > 0:
        print(f"  Found {nan_count:,} NaN values, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if inf_count > 0:
        print(f"  Found {inf_count:,} Inf values, replacing with large finite values")
        X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)
    
    print(f"  Final matrix clean: NaN={np.isnan(X).any()}, Inf={np.isinf(X).any()}")
    
    return X, df_meta, feature_names_final


def save_processed_data(X: np.ndarray, df_meta: pd.DataFrame, feature_names: list = None) -> None:
    """
    Save processed feature matrix and metadata to disk.
    
    Args:
        X: Feature matrix
        df_meta: Metadata DataFrame
        feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    # Save feature matrix
    np.save(config.PROCESSED_DATA_PATH, X)
    print(f"Saved feature matrix to {config.PROCESSED_DATA_PATH}")
    print(f"  Shape: {X.shape}")
    print(f"  Size: {config.PROCESSED_DATA_PATH.stat().st_size / 1024**2:.2f} MB")
    
    # Save metadata
    df_meta.to_csv(config.PROCESSED_META_PATH, index=True)
    print(f"\nSaved metadata to {config.PROCESSED_META_PATH}")
    print(f"  Shape: {df_meta.shape}")
    print(f"  Size: {config.PROCESSED_META_PATH.stat().st_size / 1024**2:.2f} MB")
    
    # Save feature names
    if feature_names:
        feature_names_path = config.DATA_DIR / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"\nSaved feature names to {feature_names_path}")
        print(f"  Features: {feature_names}")


def load_processed_data() -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load preprocessed feature matrix and metadata from disk.
    
    Returns:
        Tuple of (X: feature matrix, df_meta: metadata)
    """
    print("Loading processed data...")
    
    X = np.load(config.PROCESSED_DATA_PATH)
    df_meta = pd.read_csv(config.PROCESSED_META_PATH, index_col=0)
    
    print(f"Loaded X: {X.shape}")
    print(f"Loaded metadata: {df_meta.shape}")
    
    # Load feature names if available
    feature_names_path = config.DATA_DIR / "feature_names.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded feature names: {feature_names}")
    
    return X, df_meta


def main():
    """
    Main pipeline: load → explore → clean → engineer → encode → save.
    """
    import time
    start_time = time.time()
    
    print("="*80)
    print("DATA PREPARATION PIPELINE (WITH FEATURE SELECTION)")
    print("="*80)
    
    # Step 1: Load
    df = load_raw_data()
    
    # Step 2: Explore
    basic_explore(df)
    
    # Step 3: Clean
    df = clean_data(df)
    
    # Step 4: Engineer features
    df = engineer_features(df)
    
    # Step 5: Encode AND SELECT features
    X, df_meta, feature_names = encode_features(df)
    
    # Step 6: Save
    save_processed_data(X, df_meta, feature_names)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"Final feature matrix: {X.shape}")
    print(f"Features used: {feature_names}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
