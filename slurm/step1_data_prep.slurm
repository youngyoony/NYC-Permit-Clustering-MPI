#!/bin/bash
#SBATCH -J step1_data_prep
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
# STEP 1: Data Preparation
# Loads raw CSV, cleans, engineers features, encodes, saves
# ============================================================

echo "============================================================"
echo " STEP 1: Data Preparation"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs

module purge
module load python/3.11.2
module load mpi4py/latest

# CPU optimization
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo ""
echo "Python version:"
python --version
echo ""

# Check data file (single or split files)
if [ -f "data/DOB_Permit_Issuance.csv" ]; then
    echo "Data file found: $(ls -lh data/DOB_Permit_Issuance.csv | awk '{print $5}')"
elif ls data/data_*.csv 1> /dev/null 2>&1; then
    echo "Split data files found:"
    ls -lh data/data_*.csv
    echo "Total files: $(ls data/data_*.csv | wc -l)"
else
    echo "ERROR: No data files found in data/ directory"
    echo "Expected: data/DOB_Permit_Issuance.csv OR data/data_*.csv"
    exit 1
fi
echo ""

# Run data preparation
START_TIME=$(date +%s)

python -u data_prep.py 2>&1 | tee logs/data_prep_${SLURM_JOB_ID}.log
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Data Preparation"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes ($((RUNTIME/3600))h $((RUNTIME%3600/60))m)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Data preparation completed"
    echo ""
    echo "Output files:"
    ls -lh data/processed_*.npy data/processed_*.csv 2>/dev/null || echo "  (checking...)"
else
    echo "✗ FAILED - Check logs for errors"
    echo "Log file: logs/data_prep_${SLURM_JOB_ID}.log"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
