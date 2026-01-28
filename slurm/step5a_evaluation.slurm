#!/bin/bash
#SBATCH -J step5a_evaluation
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# STEP 5a: Evaluation - Scaling Analysis & Comparisons
# Can run in parallel with step5b_visualization
# ============================================================

echo "============================================================"
echo " STEP 5a: Evaluation & Scaling Analysis"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/scaling results/comparison

module purge
module load python/3.11.2
module load mpi4py/latest

# FIXED: Install required packages
pip install scikit-learn matplotlib seaborn pandas numpy --quiet --break-system-packages

export PYTHONUNBUFFERED=1

# Check prerequisites
echo "Checking for scaling results..."
if [ -f "results/scaling/kmeans_mpi_scaling.csv" ]; then
    echo "  ✓ K-Means MPI scaling data found"
else
    echo "  ⚠ K-Means MPI scaling data not found"
fi

if [ -f "results/scaling/hierarchical_mpi_scaling.csv" ]; then
    echo "  ✓ Hierarchical MPI scaling data found"
else
    echo "  ⚠ Hierarchical MPI scaling data not found"
fi
echo ""

# Run evaluation
START_TIME=$(date +%s)

python -u evaluation.py 2>&1 | tee logs/evaluation_${SLURM_JOB_ID}.log
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Evaluation"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Evaluation completed"
    echo ""
    echo "Generated files:"
    ls -lh results/scaling/*.png results/comparison/*.csv 2>/dev/null
else
    echo "✗ FAILED - Check logs for errors"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
