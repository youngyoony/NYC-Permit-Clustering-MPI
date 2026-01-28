#!/bin/bash
# =============================================================================
# NYC Building Permit Clustering Pipeline
# Run this script to execute the entire analysis pipeline on SeaWulf HPC
# =============================================================================

echo "=================================================="
echo "NYC Building Permit Clustering - MPI Pipeline"
echo "=================================================="

# Step 1: Data Preparation
echo "[Step 1/7] Submitting data preparation job..."
JOB1=$(sbatch --parsable slurm/step1_data_prep.slurm)
echo "  Job ID: $JOB1"

# Step 2: Dimensionality Reduction & EDA (can run in parallel)
echo "[Step 2/7] Submitting dimensionality reduction and EDA jobs..."
JOB2A=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2a_dim_reduction.slurm)
JOB2B=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2b_eda.slurm)
echo "  Job IDs: $JOB2A, $JOB2B"

# Step 3: Baseline Clustering
echo "[Step 3/7] Submitting baseline clustering jobs..."
JOB3A=$(sbatch --parsable --dependency=afterok:$JOB2A slurm/step3a_kmeans_baseline.slurm)
JOB3B=$(sbatch --parsable --dependency=afterok:$JOB2A slurm/step3b_hierarchical_baseline.slurm)
echo "  Job IDs: $JOB3A, $JOB3B"

# Step 4: MPI-Parallel Clustering
echo "[Step 4/7] Submitting MPI clustering jobs..."
JOB4A=$(sbatch --parsable --dependency=afterok:$JOB2A slurm/step4a_kmeans_mpi.slurm)
JOB4B=$(sbatch --parsable --dependency=afterok:$JOB2A slurm/step4b_hierarchical_mpi.slurm)
echo "  Job IDs: $JOB4A, $JOB4B"

# Step 5: Evaluation & Visualization
echo "[Step 5/7] Submitting evaluation and visualization jobs..."
JOB5A=$(sbatch --parsable --dependency=afterok:$JOB4A:$JOB4B slurm/step5a_evaluation.slurm)
JOB5B=$(sbatch --parsable --dependency=afterok:$JOB5A slurm/step5b_visualization.slurm)
JOB5C=$(sbatch --parsable --dependency=afterok:$JOB5A slurm/step5c_cluster_interpretation.slurm)
JOB5D=$(sbatch --parsable --dependency=afterok:$JOB5A slurm/step5d_hybrid_interpretation.slurm)
JOB5E=$(sbatch --parsable --dependency=afterok:$JOB5A slurm/step5e_geographic_viz_by_era.slurm)
echo "  Job IDs: $JOB5A, $JOB5B, $JOB5C, $JOB5D, $JOB5E"

# Step 6: Amdahl's Law Visualization
echo "[Step 6/7] Submitting Amdahl's law analysis..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB4A:$JOB4B slurm/step6_amdahl_viz.slurm)
echo "  Job ID: $JOB6"

# Step 7: Compute Final Metrics
echo "[Step 7/7] Submitting final metrics computation..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5A:$JOB6 slurm/step7_compute_metrics.slurm)
echo "  Job ID: $JOB7"

echo ""
echo "=================================================="
echo "Pipeline submitted successfully!"
echo "=================================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo "Results will be in: results/"
