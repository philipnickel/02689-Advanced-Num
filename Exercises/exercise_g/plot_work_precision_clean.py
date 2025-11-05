#!/usr/bin/env python3
"""
Work-Precision Plot: RK3 vs RK4 Efficiency Comparison
======================================================

Creates a single work-precision diagram showing error vs computational cost.
This demonstrates which time integrator is more efficient for achieving
a given accuracy.
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
df = pd.read_parquet(DATA_DIR / "work_precision.parquet")


def estimate_convergence_rate(dt_values, errors):
    """Estimate convergence rate from error vs dt."""
    log_dt = np.log(dt_values)
    log_err = np.log(errors)
    coeffs = np.polyfit(log_dt, log_err, 1)
    return coeffs[0]


def create_work_precision_plot():
    """Create work-precision diagram (error vs wall time)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = df['method'].unique()
    colors = {'RK3': '#ff7f0e', 'RK4': '#1f77b4'}
    markers = {'RK3': 's', 'RK4': 'o'}

    for method in methods:
        data = df[df['method'] == method].sort_values('wall_time')

        # Get convergence rate for label
        dt_vals = data['dt'].values
        err_vals = data['error_l2'].values
        rate = estimate_convergence_rate(dt_vals, err_vals)
        order = 3 if method == "RK3" else 4

        ax.loglog(data['wall_time'], data['error_l2'],
                  marker=markers[method],
                  label=f'{method} (order {order}, rate={rate:.2f})',
                  color=colors[method],
                  linewidth=2,
                  markersize=10,
                  alpha=0.8)

    ax.set_xlabel('Wall Time (s)', fontsize=12)
    ax.set_ylabel('L² Error', fontsize=12)
    ax.set_title('Work-Precision Diagram: RK3 vs RK4', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Add annotation
    ax.text(0.05, 0.05,
            'Lower-left is better\n(less time, less error)',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    output_file = FIG_DIR / "work_precision.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def print_summary():
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Work-Precision Analysis Summary")
    print("=" * 70)

    for method in df['method'].unique():
        data = df[df['method'] == method].sort_values('dt')

        print(f"\n{method}:")
        print("-" * 70)

        # Best accuracy achieved
        best_idx = data['error_l2'].idxmin()
        best = data.loc[best_idx]
        print(f"  Best accuracy: L² = {best['error_l2']:.3e} in {best['wall_time']:.3f}s")

        # Convergence rate
        dt_vals = data['dt'].values
        err_vals = data['error_l2'].values
        rate = estimate_convergence_rate(dt_vals, err_vals)
        expected_rate = 3 if method == "RK3" else 4

        print(f"  Convergence rate: {rate:.2f} (expected {expected_rate:.0f})")

    # Efficiency comparison
    print("\n" + "=" * 70)
    print("Efficiency Comparison")
    print("=" * 70)

    # Find comparable errors
    rk3_data = df[df['method'] == 'RK3'].sort_values('error_l2')
    rk4_data = df[df['method'] == 'RK4'].sort_values('error_l2')

    # Target error: median of RK3 errors
    target_err = rk3_data['error_l2'].median()

    # Find closest RK3 and RK4 runs
    rk3_close = rk3_data.iloc[(rk3_data['error_l2'] - target_err).abs().argsort()[:1]]
    rk4_close = rk4_data.iloc[(rk4_data['error_l2'] - target_err).abs().argsort()[:1]]

    rk3_time = rk3_close['wall_time'].values[0]
    rk4_time = rk4_close['wall_time'].values[0]
    rk3_err = rk3_close['error_l2'].values[0]
    rk4_err = rk4_close['error_l2'].values[0]

    print(f"\nFor similar error (≈ {target_err:.2e}):")
    print(f"  RK3: {rk3_err:.3e} in {rk3_time:.3f}s")
    print(f"  RK4: {rk4_err:.3e} in {rk4_time:.3f}s")
    print(f"  → RK4 is {rk3_time/rk4_time:.2f}× faster for comparable accuracy")

    print("\n" + "=" * 70)


def main():
    """Generate work-precision plot."""
    print("=" * 70)
    print("Generating Work-Precision Plot")
    print("=" * 70)

    create_work_precision_plot()
    print_summary()

    print("\n✓ Work-precision plot generated!")


if __name__ == "__main__":
    main()
