"""
Configuration file for SeaWulf HPC Clustering Project
Centralized paths, constants, and settings for all scripts.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Base directory - Team Project path
BASE_DIR = Path("/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project")
SCRATCH_DIR = Path("/gpfs/scratch") / os.environ.get('USER', 'user') / "clustering_project"

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "DOB_Permit_Issuance.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_X.npy"
PROCESSED_META_PATH = DATA_DIR / "processed_meta.csv"

# Output paths
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
SLURM_DIR = BASE_DIR / "slurm"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, SLURM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Results subdirectories
EDA_DIR = RESULTS_DIR / "eda"
SCALING_DIR = RESULTS_DIR / "scaling"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

for dir_path in [EDA_DIR, SCALING_DIR, CLUSTERS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================
# Geographic bounds for NYC
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

# Manhattan center (approximate) for distance calculations
MANHATTAN_CENTER = (40.7831, -73.9712)

# Data cleaning thresholds
MAX_COST = 1e9  # Cap extreme costs at $1B
MIN_COST = 0    # Minimum valid cost
OUTLIER_PERCENTILE = 99.5  # Percentile for outlier removal

# Feature engineering
DATETIME_FEATURES = ['Filing Date', 'Issuance Date', 'Expiration Date']
NUMERIC_FEATURES = ['Proposed Dwelling Units', 'Proposed Occupancy', 
                   'Job Cost', 'Existing Zoning Sqft', 'Proposed Zoning Sqft']
CATEGORICAL_FEATURES = ['Borough', 'Permit Type', 'Permit Status', 
                       'Work Type', 'Residential', 'Owner Type']
TEXT_FEATURES = ['Job Description']

# SELECTED FEATURES FOR CLUSTERING (23 features)
FEATURES_FOR_CLUSTERING = [
    "LATITUDE",
    "LONGITUDE",
    "COUNCIL_DISTRICT",
    "CENSUS_TRACT",
    "Bldg Type",
    "Filing_Year",
    "Filing_Month",
    "Filing_Quarter",
    "Permit_Age_Days",
    "Permit Type_EQ",
    "Permit Type_EW",
    "Permit Type_FO",
    "Permit Type_NB",
    "Permit Type_PL",
    "Permit Type_SG",
    "Job Type_A2",
    "Job Type_A3",
    "Job Type_DM",
    "Job Type_NB",
    "Job Type_SG",
    "Permit Status_ISSUED",
    "Permit Status_RE ISSUED",
    "Residential_YES",
]

# TF-IDF parameters
TFIDF_MAX_FEATURES = 100
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.5

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================
# Dimensionality reduction
PCA_VARIANCE_THRESHOLD = 0.95  # Keep 95% variance
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# K-Means
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
KMEANS_TOL = 1e-4
KMEANS_K_RANGE = [2, 4, 5, 8, 10, 12, 16, 20]  # Borough count = 5 is key

# Hierarchical clustering
HIERARCHICAL_SAMPLE_SIZE = 5000
HIERARCHICAL_SAMPLE_PER_BOROUGH = 1000
HIERARCHICAL_LINKAGE = 'ward'
HIERARCHICAL_MAX_K = 20

# BFR algorithm
BFR_CHUNK_SIZE = 100000  # Process 100K rows at a time
BFR_MAHALANOBIS_THRESHOLD = 4.0  # 4 standard deviations
BFR_INITIAL_SAMPLE_SIZE = 50000
BFR_MIN_CLUSTER_SIZE = 100

# CURE algorithm
CURE_SAMPLE_SIZE = 50000
CURE_NUM_REPRESENTATIVES = 10
CURE_SHRINK_FACTOR = 0.2
CURE_MERGE_THRESHOLD = 0.5

# Stream clustering (BDMO)
STREAM_BASE_BUCKET_SIZE = 3
STREAM_DECAY_FACTOR = 0.9

# MPI parameters
MPI_ROOT_RANK = 0
MPI_CONVERGENCE_TOL = 1e-4
MPI_MAX_ITERATIONS = 100

# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================
# Sampling for visualization
VIS_SAMPLE_SIZE = 10000  # Sample for t-SNE/UMAP plots
SILHOUETTE_SAMPLE_SIZE = 50000  # Silhouette scoring can be expensive

# Random seed for reproducibility
RANDOM_SEED = 42

# Plot settings
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# SEAWULF HPC SETTINGS
# ============================================================================
# Module names (adjust based on SeaWulf configuration)
SEAWULF_MODULES = [
    "slurm",
    "python/3.11.2",
    "mpi4py/latest"
]

# Slurm defaults
SLURM_PARTITION = "short-28core"  # Adjust based on queue availability
SLURM_TIME_LIMIT = "04:00:00"     # 4 hours default
SLURM_MEM_PER_CPU = "4G"

# Process counts for scaling experiments
SCALING_PROCESS_COUNTS = [1, 2, 4, 8, 16]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_timestamp():
    """Return current timestamp string for logging."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_experiment_path(experiment_name: str, create: bool = True) -> Path:
    """Get path for experiment results with timestamp."""
    timestamp = get_timestamp()
    exp_path = RESULTS_DIR / experiment_name / timestamp
    if create:
        exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path

def get_log_path(job_name: str) -> Path:
    """Get path for log file."""
    timestamp = get_timestamp()
    return LOGS_DIR / f"{job_name}_{timestamp}.log"

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Raw data path: {RAW_DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Features for clustering: {len(FEATURES_FOR_CLUSTERING)} features")
