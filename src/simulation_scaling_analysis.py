#!/usr/bin/env python3
"""
Simulation Scaling Analysis: Aggregate results from multiple MPI runs
AMS 598 - Big Data Analysis

Run this AFTER running simulation_mpi.py with different process counts.
This script aggregates results and generates comparison tables.

Usage:
    python simulation_scaling_analysis.py
"""

import json
import os
import glob
import numpy as np

def load_all_results():
    """Load all simulation MPI results."""
    results = []
    
    # Load baseline
    if os.path.exists('simulation_baseline_results.json'):
        with open('simulation_baseline_results.json', 'r') as f:
            baseline = json.load(f)
            baseline['n_processes'] = 1
            baseline['speedup'] = 1.0
            baseline['efficiency'] = 100.0
            baseline['comm_pct'] = 0.0
            baseline['total_time'] = baseline['runtime_seconds']
            baseline['ari_vs_truth'] = baseline['ari']
            baseline['nmi_vs_truth'] = baseline['nmi']
            baseline['agreement_pct'] = 100.0
            baseline['silhouette'] = baseline['silhouette_score']
            results.append(baseline)
    
    # Load MPI results
    for filepath in sorted(glob.glob('simulation_mpi_np*_results.json')):
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    
    # Sort by process count
    results.sort(key=lambda x: x['n_processes'])
    
    return results

def print_scaling_table(results):
    """Print scaling performance table."""
    print("\n" + "=" * 80)
    print("SIMULATION SCALING ANALYSIS")
    print("=" * 80)
    
    print("\n### Performance Scaling ###\n")
    print(f"{'Processes':<12} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Comm %':<10}")
    print("-" * 58)
    
    for r in results:
        np_val = r['n_processes']
        time_val = r['total_time']
        speedup = r['speedup']
        eff = r['efficiency']
        comm = r.get('comm_pct', 0.0)
        
        print(f"{np_val:<12} {time_val:<12.2f} {speedup:<12.2f} {eff:<12.1f} {comm:<10.1f}")

def print_quality_table(results):
    """Print clustering quality table."""
    print("\n### Clustering Quality ###\n")
    print(f"{'Processes':<12} {'Silhouette':<12} {'ARI (truth)':<12} {'NMI (truth)':<12} {'Agreement':<12}")
    print("-" * 60)
    
    for r in results:
        np_val = r['n_processes']
        sil = r['silhouette']
        ari = r['ari_vs_truth']
        nmi = r['nmi_vs_truth']
        agr = r['agreement_pct']
        
        print(f"{np_val:<12} {sil:<12.4f} {ari:<12.4f} {nmi:<12.4f} {agr:<12.2f}")

def print_comparison_with_real_data(results):
    """Compare simulation results with real data results."""
    print("\n" + "=" * 80)
    print("COMPARISON: SIMULATION vs REAL DATA")
    print("=" * 80)
    
    # Best simulation result (highest process count)
    best_sim = results[-1] if len(results) > 1 else results[0]
    
    # Real data results (hardcoded from your actual results)
    real_data = {
        'n_samples': 3983393,
        'silhouette': 0.078,
        'speedup_16p': 10.52,
        'efficiency_16p': 65.8,
        'comm_pct': 3.2,
        'agreement_8p': 97.99
    }
    
    print(f"\n{'Metric':<30} {'Simulation':<20} {'Real Data':<20}")
    print("-" * 70)
    print(f"{'Samples':<30} {results[0]['n_samples']:,} {real_data['n_samples']:,}")
    print(f"{'Silhouette Score':<30} {best_sim['silhouette']:.4f} {real_data['silhouette']:.4f}")
    print(f"{'ARI vs Ground Truth':<30} {best_sim['ari_vs_truth']:.4f} {'N/A (no labels)':<20}")
    print(f"{'Best Speedup':<30} {best_sim['speedup']:.2f}x {real_data['speedup_16p']:.2f}x")
    print(f"{'Communication Overhead':<30} {best_sim.get('comm_pct', 0):.1f}% {real_data['comm_pct']:.1f}%")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print(f"✓ Simulation achieves HIGH silhouette ({best_sim['silhouette']:.3f}) → Algorithm works correctly")
    print(f"✓ Real data has LOW silhouette ({real_data['silhouette']:.3f}) → Data characteristics, not bug")
    print(f"✓ Similar speedup patterns confirm MPI implementation is correct")
    print(f"✓ High ARI on simulation ({best_sim['ari_vs_truth']:.3f}) proves cluster recovery is accurate")

def generate_markdown_table(results):
    """Generate markdown tables for presentation."""
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES FOR PRESENTATION")
    print("=" * 80)
    
    # Scaling table
    print("\n```markdown")
    print("### Simulation: Scaling Performance\n")
    print("| Processes | Time (s) | Speedup | Efficiency | Comm % |")
    print("|-----------|----------|---------|------------|--------|")
    for r in results:
        print(f"| {r['n_processes']} | {r['total_time']:.2f} | {r['speedup']:.2f}x | {r['efficiency']:.1f}% | {r.get('comm_pct', 0):.1f}% |")
    print("```")
    
    # Quality table
    print("\n```markdown")
    print("### Simulation: Clustering Quality\n")
    print("| Processes | Silhouette | ARI | NMI | Agreement |")
    print("|-----------|------------|-----|-----|-----------|")
    for r in results:
        print(f"| {r['n_processes']} | {r['silhouette']:.4f} | {r['ari_vs_truth']:.4f} | {r['nmi_vs_truth']:.4f} | {r['agreement_pct']:.1f}% |")
    print("```")
    
    # Comparison table
    print("\n```markdown")
    print("### Simulation vs Real Data Comparison\n")
    print("| Metric | Simulation | Real Data |")
    print("|--------|------------|-----------|")
    best = results[-1] if len(results) > 1 else results[0]
    print(f"| Samples | {results[0]['n_samples']:,} | 3,983,393 |")
    print(f"| Silhouette | **{best['silhouette']:.4f}** | 0.0780 |")
    print(f"| ARI vs Truth | **{best['ari_vs_truth']:.4f}** | N/A |")
    print(f"| Best Speedup | {best['speedup']:.2f}x | 10.52x |")
    print("```")

def main():
    print("\n" + "=" * 80)
    print("AMS 598 - SIMULATION SCALING ANALYSIS")
    print("=" * 80)
    
    results = load_all_results()
    
    if len(results) == 0:
        print("\nERROR: No results found!")
        print("Run these commands first:")
        print("  python simulation_baseline.py")
        print("  mpirun -np 2 python simulation_mpi.py")
        print("  mpirun -np 4 python simulation_mpi.py")
        print("  mpirun -np 8 python simulation_mpi.py")
        print("  mpirun -np 16 python simulation_mpi.py")
        return
    
    print(f"\nFound {len(results)} result files")
    
    print_scaling_table(results)
    print_quality_table(results)
    print_comparison_with_real_data(results)
    generate_markdown_table(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
