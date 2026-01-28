#!/bin/bash
#SBATCH -J step5b_visualization
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
# STEP 5b: Visualization - t-SNE plots, Geographic maps
# Can run in parallel with step5a_evaluation
# ============================================================

echo "============================================================"
echo " STEP 5b: Visualization"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/visualizations

module purge
module load python/3.11.2
module load mpi4py/latest

# FIXED: Install required packages
pip install scikit-learn matplotlib seaborn pandas numpy --quiet --break-system-packages

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Check prerequisites
echo "Checking for clustering results..."
ls results/clusters/*/labels_*.npy 2>/dev/null | head -5
echo ""

# Run visualization
START_TIME=$(date +%s)

python -u visualization.py 2>&1 | tee logs/visualization_${SLURM_JOB_ID}.log
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Visualization"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Visualization completed"
    echo ""
    echo "Generated visualizations:"
    find results/visualizations -name "*.png" -exec ls -lh {} \; 2>/dev/null
else
    echo "✗ FAILED - Check logs for errors"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
