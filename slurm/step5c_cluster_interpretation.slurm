#!/bin/bash
#SBATCH -J step5c_cluster_interp
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
# STEP 5c: Cluster Interpretation
# Analyzes cluster characteristics using raw data + labels
# Can run in parallel with step5a, step5b, step5d
# ============================================================

echo "============================================================"
echo " STEP 5c: Cluster Interpretation"
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

# Check prerequisites
LABELS_FILE="results/clusters/kmeans_mpi/labels_k10_np1.npy"

if [ ! -f "${LABELS_FILE}" ]; then
    echo "ERROR: Labels file not found: ${LABELS_FILE}"
    echo "Run step4a_kmeans_mpi first!"
    exit 1
fi

# FIXED: Check for merged CSV file
if [ ! -f "data/DOB_Permit_Issuance_merged.csv" ]; then
    echo "ERROR: Raw data file not found!"
    exit 1
fi

echo "Labels file found: ${LABELS_FILE}"
echo ""

# Run cluster interpretation
START_TIME=$(date +%s)

python -u run_cluster_interpretation.py \
    --labels ${LABELS_FILE} \
    --output results/clusters/kmeans_mpi \
    2>&1 | tee logs/cluster_interpretation_${SLURM_JOB_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Cluster Interpretation"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Cluster interpretation completed"
    echo ""
    echo "Output files:"
    ls -lh results/clusters/kmeans_mpi/cluster_interpretation.csv 2>/dev/null
else
    echo "✗ FAILED - Check logs for errors"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
