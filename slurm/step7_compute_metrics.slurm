#!/bin/bash
#SBATCH -J compute_mpi_metrics
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# Compute MPI Clustering Metrics (Silhouette, Davies-Bouldin, etc.)
# Run AFTER step4a_kmeans_mpi.slurm and step4b_hierarchical_mpi.slurm
# ============================================================

echo "============================================================"
echo " Computing MPI Clustering Quality Metrics"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs

module purge
module load python/3.11.2
module load mpi4py/latest

# Install required packages
pip install scikit-learn matplotlib seaborn pandas numpy --quiet --break-system-packages

export PYTHONUNBUFFERED=1

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if [ ! -f "data/processed_X.npy" ]; then
    echo "ERROR: processed_X.npy not found!"
    exit 1
fi
echo "  ✓ processed_X.npy found"

if [ -d "results/clusters/kmeans_mpi" ]; then
    echo "  ✓ K-Means MPI results found"
    ls results/clusters/kmeans_mpi/labels_k10_np*.npy 2>/dev/null | wc -l | xargs -I {} echo "    {} label files"
else
    echo "  ✗ K-Means MPI results not found"
fi

if [ -d "results/clusters/hierarchical_mpi" ]; then
    echo "  ✓ Hierarchical MPI results found"
    ls results/clusters/hierarchical_mpi/labels_k10_np*.npy 2>/dev/null | wc -l | xargs -I {} echo "    {} label files"
else
    echo "  ✗ Hierarchical MPI results not found"
fi

echo ""
echo "Running metrics computation..."
echo ""

python -u compute_mpi_metrics.py

echo ""
echo "============================================================"
echo " Results Summary"
echo "============================================================"

if [ -f "results/clusters/kmeans_mpi/quality_metrics.csv" ]; then
    echo ""
    echo "K-Means MPI Quality Metrics:"
    cat results/clusters/kmeans_mpi/quality_metrics.csv
fi

if [ -f "results/clusters/hierarchical_mpi/quality_metrics.csv" ]; then
    echo ""
    echo "Hierarchical MPI Quality Metrics:"
    cat results/clusters/hierarchical_mpi/quality_metrics.csv
fi

echo ""
echo "============================================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
