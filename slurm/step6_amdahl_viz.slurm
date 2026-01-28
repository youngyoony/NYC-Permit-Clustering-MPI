#!/bin/bash
#SBATCH -J amdahl_viz
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p hbm-short-96core
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=keunyoung.yoon@stonybrook.edu

# ============================================================
# Amdahl's Law Visualization
# Run AFTER step4a_kmeans_mpi.slurm completes
# ============================================================

echo "============================================================"
echo " Amdahl's Law Visualization"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/visualizations

module purge
module load python/3.11.2

# Install required packages
pip install matplotlib seaborn pandas numpy --quiet --break-system-packages

# Check if scaling results exist
if [ -f "results/scaling/kmeans_mpi_scaling.csv" ]; then
    echo "Found MPI scaling results - using actual data"
else
    echo "Warning: MPI scaling results not found - using expected values"
fi

python -u amdahl_visualization.py

echo ""
echo "============================================================"
echo "Output files:"
ls -lh results/visualizations/amdahl*.png results/visualizations/parallel*.png 2>/dev/null
echo "============================================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
