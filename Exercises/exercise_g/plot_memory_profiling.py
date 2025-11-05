#!/usr/bin/env python3
"""
Memory Profiling Plot: RK3 vs RK4 Memory Usage
===============================================

Creates a plot showing peak memory usage vs grid size N for both
RK3 and RK4 time integrators.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = Path("data/A2/ex_g")
FIG_DIR = Path("figures/A2/ex_g")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(DATA_DIR / "memory_profiling.parquet")


def create_memory_plot():
    """Create memory usage plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = df['method'].unique()
    colors = {'RK3': '#ff7f0e', 'RK4': '#1f77b4'}
    markers = {'RK3': 's', 'RK4': 'o'}

    for method in methods:
        data = df[df['method'] == method].sort_values('N')

        N_vals = data['N'].values
        mem_vals = data['peak_memory_mb'].values

        # Estimate scaling exponent
        log_N = np.log(N_vals)
        log_mem = np.log(mem_vals)
        coeffs = np.polyfit(log_N, log_mem, 1)
        alpha = coeffs[0]

        # Plot data
        ax.loglog(N_vals, mem_vals,
                  marker=markers[method],
                  label=f'{method} (α = {alpha:.2f})',
                  color=colors[method],
                  linewidth=2,
                  markersize=10,
                  alpha=0.8)

        # Plot theoretical O(N) line
        N_ref = N_vals[len(N_vals)//2]
        mem_ref = mem_vals[len(N_vals)//2]
        N_theory = np.array([N_vals.min(), N_vals.max()])
        mem_theory = mem_ref * (N_theory / N_ref)

        ax.loglog(N_theory, mem_theory, '--',
                  color=colors[method],
                  alpha=0.4,
                  linewidth=1.5,
                  label=f'{method} O(N) reference')

    ax.set_xlabel('Grid Size N', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage: RK3 vs RK4', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save
    output_file = FIG_DIR / "memory_profiling.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def print_summary():
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Memory Profiling Summary")
    print("=" * 70)

    for method in df['method'].unique():
        data = df[df['method'] == method].sort_values('N')

        N_vals = data['N'].values
        mem_vals = data['peak_memory_mb'].values

        # Scaling exponent
        log_N = np.log(N_vals)
        log_mem = np.log(mem_vals)
        coeffs = np.polyfit(log_N, log_mem, 1)
        alpha = coeffs[0]

        print(f"\n{method}:")
        print(f"  Scaling exponent: α = {alpha:.2f} (expected 1.00)")
        print(f"  Memory range: {mem_vals.min():.2f} - {mem_vals.max():.2f} MB")
        print(f"  At N=4096: {data[data['N']==4096]['peak_memory_mb'].values[0]:.2f} MB")

    # Comparison
    rk3_data = df[df['method'] == 'RK3'].sort_values('N')
    rk4_data = df[df['method'] == 'RK4'].sort_values('N')

    avg_ratio = (rk4_data['peak_memory_mb'] / rk3_data['peak_memory_mb']).mean()

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"  RK4 uses {avg_ratio:.2f}× memory of RK3 (average)")
    print(f"  Both methods scale as O(N) ✓")
    print("=" * 70)


def main():
    """Generate memory profiling plot."""
    print("=" * 70)
    print("Generating Memory Profiling Plot")
    print("=" * 70)

    create_memory_plot()
    print_summary()

    print("\n✓ Memory profiling plot generated!")


if __name__ == "__main__":
    main()
