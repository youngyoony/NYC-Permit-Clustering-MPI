#!/bin/bash
#SBATCH -J step3a_kmeans_baseline
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# STEP 3a: K-Means Baseline (Single Node)
# Can run in parallel with step3b_hierarchical_baseline
# ============================================================

echo "============================================================"
echo " STEP 3a: K-Means Baseline (Single Node)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/clusters/kmeans

module purge
module load python/3.11.2
module load mpi4py/latest

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Check prerequisites
if [ ! -f "data/processed_X.npy" ]; then
    echo "ERROR: processed_X.npy not found. Run step1_data_prep first!"
    exit 1
fi

# Check for PCA data (optional but recommended)
if [ -f "data/processed_X_pca.npy" ]; then
    echo "PCA data found - will use reduced dimensions"
else
    echo "WARNING: PCA data not found - using full dimensions (slower)"
fi
echo ""

# Run K-Means baseline
START_TIME=$(date +%s)

python -u kmeans_baseline.py 2>&1 | tee logs/kmeans_baseline_${SLURM_JOB_ID}.log
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - K-Means Baseline"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - K-Means baseline completed"
    echo ""
    echo "Output files:"
    ls -lh results/clusters/kmeans/ 2>/dev/null
else
    echo "✗ FAILED - Check logs for errors"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
