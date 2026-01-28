#!/bin/bash
#SBATCH -J step5e_geo_era
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
# STEP 5e: Geographic Visualization BY ERA
# Creates NYC maps grouped by temporal era (2004-2005, 2006-2008, 2010-2011)
# Key insight: K-Means clusters correspond to time periods!
# ============================================================

echo "============================================================"
echo " STEP 5e: Geographic Visualization BY ERA"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Setup environment
PROJECT_DIR="/gpfs/projects/AMS598/class2025/Yoon_KeunYoung/Team_Project"
cd ${PROJECT_DIR} || exit 1
mkdir -p logs results/visualizations/geographic

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

if [ ! -f "data/DOB_Permit_Issuance_merged.csv" ]; then
    echo "ERROR: Raw data file not found!"
    exit 1
fi

echo "Labels file found: ${LABELS_FILE}"
echo ""

# Run ERA-based geographic visualization
START_TIME=$(date +%s)

python -u run_geographic_viz_by_era.py \
    --labels ${LABELS_FILE} \
    --output results/visualizations/geographic \
    2>&1 | tee logs/geographic_viz_era_${SLURM_JOB_ID}.log

EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Summary
echo ""
echo "============================================================"
echo " Job Summary - Geographic Visualization BY ERA"
echo "============================================================"
echo "Exit Code: ${EXIT_CODE}"
echo "Runtime: $((RUNTIME/60)) minutes"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS - Era-based geographic visualization completed"
    echo ""
    echo "Generated plots:"
    ls -lh results/visualizations/geographic/*.png 2>/dev/null
    echo ""
    echo "NEW Era-based visualizations:"
    echo "  - geographic_by_era.png           (3-panel comparison)"
    echo "  - geographic_density_by_era.png   (density heatmaps)"
    echo "  - geographic_all_eras_overlay.png (all eras overlay)"
    echo "  - era_borough_breakdown.png       (borough % by era)"
    echo "  - temporal_trend_summary.png      (summary dashboard)"
else
    echo "✗ FAILED - Check logs for errors"
fi

echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

exit $EXIT_CODE
