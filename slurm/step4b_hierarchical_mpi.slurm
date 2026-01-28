#!/bin/bash
#SBATCH -J step4b_hierarchical_mpi
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH -t 3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# STEP 4b: Hierarchical MPI - Scaling Experiments
# Tests np={1,2,4,8,16,32,64} for strong scaling analysis
# Using 5% sample (~200k rows) - O(n²) memory constraint
# ============================================================

echo "============================================================"
echo " STEP 4b: Hierarchical MPI Scaling Experiments"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Total Tasks: ${SLURM_NTASKS}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/clusters/hierarchical_mpi results/scaling

module purge
module load python/3.11.2
module load mpi4py/latest

export PYTHONUNBUFFERED=1

# Check prerequisites
if [ ! -f "data/processed_X.npy" ]; then
    echo "ERROR: processed_X.npy not found. Run step1_data_prep first!"
    exit 1
fi

echo ""

# Clear previous scaling log
rm -f results/scaling/hierarchical_mpi_scaling.csv

START_TIME=$(date +%s)

# Run scaling experiments
K_CLUSTERS=10  # Number of global clusters
SAMPLE_FRAC=0.05  # Use 5% sample for hierarchical (~200k rows)

echo "Running Hierarchical MPI scaling experiments (k=${K_CLUSTERS}, sample=${SAMPLE_FRAC} = ~200k rows)..."
echo "Note: np=1,2,4 may fail due to O(n²) memory constraint - this demonstrates why MPI is needed!"
echo ""

for NP in 1 2 4 8 16 32 64; do
    echo "----------------------------------------"
    echo "Testing with ${NP} MPI processes..."
    echo "----------------------------------------"
    
    RUN_START=$(date +%s)
    
    mpirun -np ${NP} python -u hierarchical_mpi.py \
        --k ${K_CLUSTERS} \
        --sample ${SAMPLE_FRAC} \
        --method ward \
        2>&1 | tee -a logs/hierarchical_mpi_np${NP}_${SLURM_JOB_ID}.log
    
    RUN_END=$(date +%s)
    RUN_TIME=$((RUN_END - RUN_START))
    
    echo "np=${NP} completed in ${RUN_TIME} seconds"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Hierarchical MPI Scaling"
echo "============================================================"
echo "Total Runtime: $((TOTAL_RUNTIME/60)) minutes"
echo ""

echo "Scaling Results:"
if [ -f "results/scaling/hierarchical_mpi_scaling.csv" ]; then
    cat results/scaling/hierarchical_mpi_scaling.csv
else
    echo "  (scaling log not found)"
fi

echo ""
echo "Output files:"
ls -lh results/clusters/hierarchical_mpi/ 2>/dev/null

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
