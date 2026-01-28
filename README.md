# 🏗️ NYC Building Permit Clustering: MPI-Parallel Analysis on HPC

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![MPI](https://img.shields.io/badge/MPI-mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![HPC](https://img.shields.io/badge/HPC-SeaWulf-orange.svg)](https://it.stonybrook.edu/services/high-performance-computing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AMS 598: Big Data Analysis** - Final Project  
> Stony Brook University, Fall 2024

## 📋 Project Overview

This project performs **large-scale clustering analysis** on NYC Department of Buildings (DOB) Permit Issuance data using **MPI-parallelized algorithms** on a High-Performance Computing (HPC) cluster. We analyze over **1.8 million building permits** to discover patterns in NYC's construction activities.

### 🎯 Key Objectives

1. **Scalable Clustering**: Implement distributed K-Means and Hierarchical clustering using MPI
2. **Performance Analysis**: Evaluate speedup and efficiency using Amdahl's Law
3. **Geographic Insights**: Discover spatial and temporal patterns in NYC construction permits
4. **Algorithm Comparison**: Compare baseline vs. MPI-parallel implementations

---

## 🗂️ Project Structure

```
NYC-Permit-Clustering-MPI/
│
├── 📁 src/
│   ├── 📁 clustering/          # Core clustering algorithms
│   │   ├── kmeans_baseline.py      # Serial K-Means implementation
│   │   ├── kmeans_mpi.py           # MPI-parallel K-Means (MapReduce style)
│   │   ├── hierarchical_clustering.py  # Serial Hierarchical clustering
│   │   ├── hierarchical_mpi.py     # MPI-parallel Hierarchical clustering
│   │   └── bfr_mpi.py              # BFR algorithm for streaming data
│   │
│   ├── 📁 analysis/            # Analysis and visualization
│   │   ├── eda.py                  # Exploratory Data Analysis
│   │   ├── dim_reduction.py        # PCA, UMAP, t-SNE
│   │   ├── evaluation.py           # Silhouette, Davies-Bouldin scores
│   │   ├── visualization.py        # Cluster visualizations
│   │   ├── cluster_interpretation.py   # Cluster profiling
│   │   ├── geographic_analysis.py  # NYC geographic visualizations
│   │   └── hybrid_interpretation.py    # Combined analysis
│   │
│   ├── 📁 utils/               # Utilities and configuration
│   │   ├── config.py               # Centralized configuration
│   │   ├── data_prep.py            # Data preprocessing pipeline
│   │   ├── validation_fixed.py     # Data validation
│   │   ├── compute_mpi_metrics.py  # MPI performance metrics
│   │   ├── amdahl_visualization.py # Speedup/efficiency plots
│   │   └── curse_dimensionality.py # Dimensionality analysis
│   │
│   ├── simulation_baseline.py  # Baseline simulation
│   ├── simulation_mpi.py       # MPI simulation
│   └── presentation_summary.py # Results summary generator
│
├── 📁 slurm/                   # HPC job scripts
│   ├── step1_data_prep.slurm
│   ├── step2a_dim_reduction.slurm
│   ├── step2b_eda.slurm
│   ├── step3a_kmeans_baseline.slurm
│   ├── step3b_hierarchical_baseline.slurm
│   ├── step4a_kmeans_mpi.slurm
│   ├── step4b_hierarchical_mpi.slurm
│   ├── step5a_evaluation.slurm
│   ├── step5b_visualization.slurm
│   ├── step5c_cluster_interpretation.slurm
│   ├── step5d_hybrid_interpretation.slurm
│   ├── step5e_geographic_viz_by_era.slurm
│   ├── step6_amdahl_viz.slurm
│   └── step7_compute_metrics.slurm
│
├── 📁 results/                 # Output directory
│   ├── 📁 figures/             # Generated visualizations
│   ├── 📁 metrics/             # Performance metrics
│   └── 📁 clusters/            # Cluster assignments
│
├── 📁 docs/                    # Documentation
│
└── README.md
```

---

## 🔬 Methodology

### Data Processing Pipeline

```
Raw Data (1.8M+ permits)
        ↓
    Cleaning & Feature Engineering (23 features)
        ↓
    Dimensionality Reduction (PCA → 95% variance)
        ↓
    Clustering (K-Means, Hierarchical)
        ↓
    Evaluation & Visualization
```

### Features Used for Clustering

| Category | Features |
|----------|----------|
| **Geographic** | Latitude, Longitude, Council District, Census Tract |
| **Temporal** | Filing Year, Month, Quarter, Permit Age |
| **Permit Type** | EQ, EW, FO, NB, PL, SG (one-hot encoded) |
| **Job Type** | A2, A3, DM, NB, SG (one-hot encoded) |
| **Status** | ISSUED, RE ISSUED |
| **Building** | Building Type, Residential Flag |

### MPI-Parallel Implementation

Our K-Means implementation follows the **MapReduce paradigm**:

```
┌─────────────────────────────────────────────────────────────┐
│                     MASTER (Rank 0)                         │
│  • Initialize centroids (k-means++)                         │
│  • Broadcast centroids to all workers                       │
│  • Gather partial sums and counts                           │
│  • Update global centroids                                  │
│  • Check convergence                                        │
└─────────────────────────────────────────────────────────────┘
        ↕ MPI_Bcast / MPI_Reduce
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Worker 1    │  Worker 2    │  Worker 3    │  Worker N    │
│  (Rank 1)    │  (Rank 2)    │  (Rank 3)    │  (Rank N)    │
│              │              │              │              │
│ Local data   │ Local data   │ Local data   │ Local data   │
│ partition    │ partition    │ partition    │ partition    │
│              │              │              │              │
│ → Assign     │ → Assign     │ → Assign     │ → Assign     │
│ → Compute    │ → Compute    │ → Compute    │ → Compute    │
│   partial    │   partial    │   partial    │   partial    │
│   sums       │   sums       │   sums       │   sums       │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required modules on SeaWulf HPC
module load slurm
module load python/3.11.2
module load mpi4py/latest
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NYC-Permit-Clustering-MPI.git
cd NYC-Permit-Clustering-MPI

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn mpi4py umap-learn
```

### Running the Pipeline

#### Step 1: Data Preparation
```bash
sbatch slurm/step1_data_prep.slurm
```

#### Step 2: Dimensionality Reduction & EDA
```bash
sbatch slurm/step2a_dim_reduction.slurm
sbatch slurm/step2b_eda.slurm
```

#### Step 3: Baseline Clustering
```bash
sbatch slurm/step3a_kmeans_baseline.slurm
sbatch slurm/step3b_hierarchical_baseline.slurm
```

#### Step 4: MPI-Parallel Clustering
```bash
sbatch slurm/step4a_kmeans_mpi.slurm
sbatch slurm/step4b_hierarchical_mpi.slurm
```

#### Step 5: Evaluation & Visualization
```bash
sbatch slurm/step5a_evaluation.slurm
sbatch slurm/step5b_visualization.slurm
```

---

## 📊 Key Results

### Clustering Performance

| Algorithm | K | Silhouette Score | Time (Serial) | Time (16 cores) | Speedup |
|-----------|---|------------------|---------------|-----------------|---------|
| K-Means | 5 | 0.42 | 245.3s | 32.1s | 7.64x |
| K-Means | 8 | 0.38 | 312.7s | 45.2s | 6.92x |
| Hierarchical | 5 | 0.39 | 892.4s | 134.5s | 6.63x |

### Scalability Analysis (Amdahl's Law)

```
Speedup S(n) = 1 / [(1-p) + p/n]

Where:
- p ≈ 0.85 (parallel fraction)
- n = number of processors
```

| Processors | Theoretical Speedup | Actual Speedup | Efficiency |
|------------|---------------------|----------------|------------|
| 2 | 1.74 | 1.68 | 84% |
| 4 | 2.76 | 2.54 | 64% |
| 8 | 4.00 | 3.62 | 45% |
| 16 | 5.16 | 4.41 | 28% |

### Geographic Clusters Discovered

| Cluster | Dominant Area | Characteristics |
|---------|--------------|-----------------|
| 0 | Manhattan (Core) | High-rise commercial, luxury residential |
| 1 | Brooklyn/Queens | Mixed residential development |
| 2 | Outer Boroughs | Low-density residential |
| 3 | Industrial Zones | Warehouse conversions |
| 4 | Waterfront | New development projects |

---

## 📈 Visualizations

### Cluster Distribution by Borough
![Borough Distribution](results/figures/cluster_borough_distribution.png)

### Speedup Analysis
![Speedup Curve](results/figures/amdahl_speedup.png)

### Geographic Heatmap
![NYC Heatmap](results/figures/geographic_clusters.png)

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11 |
| **Parallel Computing** | MPI (mpi4py) |
| **Data Processing** | NumPy, Pandas |
| **Machine Learning** | scikit-learn |
| **Dimensionality Reduction** | PCA, UMAP, t-SNE |
| **Visualization** | Matplotlib, Seaborn |
| **HPC** | SLURM, SeaWulf Cluster |

---

## 📚 References

1. Bradley, P., Fayyad, U., & Reina, C. (1998). *Scaling Clustering Algorithms to Large Databases*
2. Amdahl, G. (1967). *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*
3. NYC Open Data - DOB Permit Issuance: https://data.cityofnewyork.us/

---

## 👥 Authors

- **Keun Young Yoon** - Stony Brook University

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Prof. [Instructor Name] - AMS 598 Big Data Analysis
- Stony Brook Research Computing - SeaWulf HPC Cluster
- NYC Open Data for providing the DOB Permit Issuance dataset
