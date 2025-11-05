#!/usr/bin/env python3
"""
Scalability Plot: Time per Timestep vs Grid Size
=================================================

Creates a single plot showing time/step vs N for RK3 and RK4,
demonstrating O(N log N) complexity from FFT operations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for consistent styling
sns.set_theme(style="darkgrid")

# Configuration
DATA_DIR = Path("data/A2/ex_g")
FIG_DIR = Path("figures/A2/ex_g")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(DATA_DIR / "scalability_timing.parquet")


def fit_power_law(N, time):
    """Fit time ~ N^α to estimate scaling exponent."""
    log_N = np.log(N)
    log_time = np.log(time)
    coeffs = np.polyfit(log_N, log_time, 1)
    return coeffs[0]


def create_scalability_plot():
    """Create scalability plot: time per step vs N."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = df['method'].unique()
    colors = {'RK4': '#1f77b4', 'RK3': '#ff7f0e'}
    markers = {'RK4': 'o', 'RK3': 's'}

    for method in methods:
        method_data = df[df['method'] == method]

        # Aggregate: compute mean and std of time_per_step for each N
        agg_data = method_data.groupby('N').agg({
            'time_per_step': ['mean', 'std']
        }).reset_index()
        agg_data.columns = ['N', 'time_mean', 'time_std']
        agg_data = agg_data.sort_values('N')

        N_vals = agg_data['N'].values
        time_mean = agg_data['time_mean'].values
        time_std = agg_data['time_std'].values

        # Compute scaling exponent
        alpha = fit_power_law(N_vals, time_mean)

        # Plot line with markers
        ax.loglog(N_vals, time_mean,
                  marker=markers[method],
                  label=f'{method} (α = {alpha:.2f})',
                  color=colors[method],
                  linewidth=2,
                  markersize=8,
                  alpha=0.8)

        # Plot shaded error band
        ax.fill_between(N_vals, time_mean - time_std, time_mean + time_std,
                        color=colors[method], alpha=0.2)

        # Plot O(N log N) reference line
        N_ref = N_vals[len(N_vals)//2]
        time_ref = time_mean[len(N_vals)//2]
        N_theory = np.array([N_vals.min(), N_vals.max()])
        time_theory = time_ref * (N_theory * np.log(N_theory)) / (N_ref * np.log(N_ref))

        ax.loglog(N_theory, time_theory, '--',
                  color=colors[method],
                  alpha=0.4,
                  linewidth=1.5,
                  label=f'{method} O(N log N) reference')

    ax.set_xlabel('Grid Size N', fontsize=12)
    ax.set_ylabel('Time per Step (s)', fontsize=12)
    ax.set_title('Scalability Analysis: RK3 vs RK4', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save
    output_file = FIG_DIR / "scalability_analysis_clean.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.close()


def print_summary():
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Scalability Analysis Summary")
    print("=" * 70)

    for method in df['method'].unique():
        data = df[df['method'] == method].sort_values('N')
        N_vals = data['N'].values
        time_vals = data['time_per_step'].values

        # Overall scaling
        alpha_all = fit_power_law(N_vals, time_vals)

        # Large N scaling (N >= 1024)
        data_large = data[data['N'] >= 1024]
        N_large = data_large['N'].values
        time_large = data_large['time_per_step'].values
        alpha_large = fit_power_law(N_large, time_large)

        print(f"\n{method}:")
        print(f"  Overall scaling exponent:       α = {alpha_all:.3f}")
        print(f"  Large N scaling (N ≥ 1024):     α = {alpha_large:.3f}")
        print(f"  Expected O(N log N):            α ≈ 1.00")
        print(f"  Time/step at N=8192:            {time_vals[-1]:.6f} s")

    print("=" * 70)


def main():
    """Generate scalability plot."""
    print("=" * 70)
    print("Generating Scalability Plot")
    print("=" * 70)

    create_scalability_plot()
    print_summary()

    print("\n✓ Scalability plot generated!")


if __name__ == "__main__":
    main()
