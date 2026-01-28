#!/bin/bash
#SBATCH -J step5d_hybrid_interp
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# STEP 5d: Hybrid Cluster Interpretation
# Interprets PCA-space clusters using original feature space
# Run AFTER step4a (kmeans_mpi) or step4b (hierarchical_mpi)
# ============================================================

echo "============================================================"
echo " STEP 5d: Hybrid Cluster Interpretation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs

module purge
module load python/3.11.2
module load mpi4py/latest

# Install required packages
pip install scikit-learn matplotlib seaborn pandas numpy --quiet --break-system-packages

export PYTHONUNBUFFERED=1

# ============================================================
# Configuration - adjust these as needed
# ============================================================
K=10
METHOD="kmeans_mpi"   # Options: kmeans, kmeans_mpi, hierarchical
NPROCS=4              # Number of MPI processes used in clustering

# ============================================================
# Check prerequisites
# ============================================================
echo ""
echo "Configuration:"
echo "  K = ${K}"
echo "  Method = ${METHOD}"
echo "  NProcs = ${NPROCS}"
echo ""

# Check for original processed data
if [ ! -f "data/processed_X.npy" ]; then
    echo "ERROR: processed_X.npy not found!"
    echo "Run step1_data_prep first!"
    exit 1
fi
echo "✓ Original data found: data/processed_X.npy"

# Check for PCA data
if [ ! -f "data/processed_X_pca.npy" ]; then
    echo "ERROR: processed_X_pca.npy not found!"
    echo "Run step2a_dim_reduction first!"
    exit 1
fi
echo "✓ PCA data found: data/processed_X_pca.npy"

# Check for feature names
if [ ! -f "data/feature_names.txt" ]; then
    echo "WARNING: feature_names.txt not found, will use generic names"
fi

# Check for labels file based on method
if [ "${METHOD}" == "kmeans_mpi" ]; then
    LABELS_FILE="results/clusters/kmeans_mpi/labels_k${K}_np${NPROCS}.npy"
elif [ "${METHOD}" == "kmeans" ]; then
    LABELS_FILE="results/clusters/kmeans/labels_k${K}.npy"
elif [ "${METHOD}" == "hierarchical" ]; then
    LABELS_FILE="results/clusters/hierarchical/labels_k${K}.npy"
else
    echo "ERROR: Unknown method: ${METHOD}"
    exit 1
fi

if [ ! -f "${LABELS_FILE}" ]; then
    echo "ERROR: Labels file not found: ${LABELS_FILE}"
    echo "Run the appropriate clustering step first!"
    exit 1
fi
echo "✓ Labels found: ${LABELS_FILE}"

# Create output directory
OUTPUT_DIR="results/hybrid_interpretation/${METHOD}_k${K}"
mkdir -p ${OUTPUT_DIR}
echo "✓ Output directory: ${OUTPUT_DIR}"
echo ""

# ============================================================
# Run hybrid interpretation
# ============================================================
START_TIME=$(date +%s)

echo "Running hybrid cluster interpretation..."
echo ""

python -u hybrid_interpretation.py \
    --k ${K} \
    --method ${METHOD} \
    --nprocs ${NPROCS} \
    2>&1 | tee logs/hybrid_interpretation_${SLURM_JOB_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo " Job Summary - Hybrid Cluster Interpretation"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes $((RUNTIME%60)) seconds"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Hybrid interpretation completed"
    echo ""
    echo "Output files:"
    ls -lh ${OUTPUT_DIR}/ 2>/dev/null
    echo ""
    echo "Key outputs:"
    echo "  - ${OUTPUT_DIR}/cluster_profiles_original.csv"
    echo "  - ${OUTPUT_DIR}/cluster_feature_heatmap.png"
    echo "  - ${OUTPUT_DIR}/cluster_radar_chart.png"
    echo "  - ${OUTPUT_DIR}/interpretation_summary.txt"
else
    echo "✗ FAILED - Check logs for errors"
    echo "Log file: logs/hybrid_interpretation_${SLURM_JOB_ID}.log"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
