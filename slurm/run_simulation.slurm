#!/bin/bash
#SBATCH -J sim_kmeans
#SBATCH -o logs/simulation_%j.out
#SBATCH -e logs/simulation_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 2
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH -t 02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# AMS 598 - Simulation K-Means MPI Scaling Study
# SeaWulf Cluster Job Script
# ============================================================

echo "============================================================"
echo " Simulation K-Means MPI Scaling Study"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/simulation

module purge
module load python/3.11.2
module load mpi4py/latest

export PYTHONUNBUFFERED=1

echo ""
echo "============================================================"
echo "STEP 1: Generate Synthetic Data + Baseline"
echo "============================================================"
python -u simulation_baseline.py
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "STEP 2: MPI Scaling Study"
echo "============================================================"

# Run with different process counts
for NP in 1 2 4 8 16 32 64; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running with $NP processes..."
    echo "------------------------------------------------------------"
    mpirun -np $NP python -u simulation_mpi.py
    if [ $? -ne 0 ]; then
        echo "WARNING: MPI with $NP processes failed!"
    fi
done

echo ""
echo "============================================================"
echo "STEP 3: Aggregate Results"
echo "============================================================"
python -u simulation_scaling_analysis.py

echo ""
echo "============================================================"
echo "Job completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
