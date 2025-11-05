#!/usr/bin/env python3
"""
Scalability Plotting: Publication-Quality Figures
==================================================

Creates comprehensive scalability analysis plots showing:
1. Wall time vs N (log-log plot with fitted α)
2. Time decomposition (N log N vs N vs constant)
3. Scaling exponent vs N range

These plots demonstrate the emergence of O(N log N) scaling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configuration
DATA_DIR = Path("data/A2/ex_g")
FIG_DIR = Path("figures/A2/ex_g")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(DATA_DIR / "scalability_timing.parquet")

# Plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100


def fit_power_law(N, time):
    """
    Fit time ~ N^α to estimate scaling exponent.

    Parameters
    ----------
    N : array
        Grid sizes
    time : array
        Timings

    Returns
    -------
    α : float
        Scaling exponent
    """
    # Fit in log space: log(time) = α·log(N) + b
    log_N = np.log(N)
    log_time = np.log(time)

    # Linear fit
    coeffs = np.polyfit(log_N, log_time, 1)
    alpha = coeffs[0]

    return alpha


def fit_decomposition(N, time):
    """
    Fit time = a·N·log(N) + b·N + c

    Parameters
    ----------
    N : array
        Grid sizes
    time : array
        Timings

    Returns
    -------
    a, b, c : float
        Fitted coefficients
    """
    def model(N, a, b, c):
        return a * N * np.log(N) + b * N + c

    try:
        params, _ = curve_fit(model, N, time, p0=[1e-9, 1e-8, 1e-5])
        return params
    except:
        return [np.nan, np.nan, np.nan]


def create_scalability_plot():
    """Create main scalability analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('KdV Solver Scalability Analysis: O(N log N) Complexity', fontsize=14, fontweight='bold')

    methods = df['method'].unique()
    colors = {'RK4': '#1f77b4', 'RK3': '#ff7f0e'}
    markers = {'RK4': 'o', 'RK3': 's'}

    # Plot 1: Time per step vs N (log-log)
    ax = axes[0, 0]
    for method in methods:
        data = df[df['method'] == method]
        N = data['N'].values
        time = data['time_per_step'].values

        # Fit scaling exponent
        alpha = fit_power_law(N, time)

        # Plot data
        ax.loglog(N, time, marker=markers[method], label=f'{method}: α = {alpha:.3f}',
                  color=colors[method], linewidth=2, markersize=6)

        # Plot fitted line
        N_fit = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)
        time_fit = time[0] * (N_fit / N[0])**alpha
        ax.loglog(N_fit, time_fit, '--', color=colors[method], alpha=0.5, linewidth=1)

    # Reference lines
    N_ref = np.array([64, 32768])
    ax.loglog(N_ref, 1e-6 * N_ref, 'k:', alpha=0.3, linewidth=1, label='O(N)')
    ax.loglog(N_ref, 1e-7 * N_ref * np.log(N_ref), 'k--', alpha=0.3, linewidth=1, label='O(N log N)')

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Time per Step (s)')
    ax.set_title('(a) Overall Scaling: Time per Step vs N')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Time decomposition
    ax = axes[0, 1]
    for method in methods:
        data = df[df['method'] == method]
        N = data['N'].values
        time = data['time_per_step'].values

        # Fit decomposition
        a, b, c = fit_decomposition(N, time)

        if not np.isnan(a):
            # Plot components
            t_nlogn = a * N * np.log(N)
            t_n = b * N
            t_const = np.full_like(N, c)

            ax.semilogy(N, time, marker=markers[method], label=f'{method} (total)',
                       color=colors[method], linewidth=2, markersize=6)
            ax.semilogy(N, t_nlogn, '--', label=f'{method} (N log N term)',
                       color=colors[method], alpha=0.5)

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Time per Step (s)')
    ax.set_title('(b) Decomposition: time = a·N·log(N) + b·N + c')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 3: Percentage contribution
    ax = axes[1, 0]
    for method in methods:
        data = df[df['method'] == method]
        N = data['N'].values
        time = data['time_per_step'].values

        # Fit decomposition
        a, b, c = fit_decomposition(N, time)

        if not np.isnan(a):
            # Calculate percentages
            t_nlogn = a * N * np.log(N)
            t_n = b * N
            t_const = np.full_like(N, c)
            t_total = t_nlogn + t_n + t_const

            pct_nlogn = 100 * t_nlogn / t_total
            pct_n = 100 * t_n / t_total
            pct_const = 100 * t_const / t_total

            ax.plot(N, pct_nlogn, marker=markers[method], label=f'{method} (N log N)',
                   color=colors[method], linewidth=2, markersize=6)

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('N log N Contribution (%)')
    ax.set_title('(c) N log N Term Dominance')
    ax.set_xscale('log')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color='k', linestyle=':', alpha=0.3, linewidth=1)

    # Plot 4: Scaling exponent vs N range
    ax = axes[1, 1]
    for method in methods:
        data = df[df['method'] == method]
        N_all = data['N'].values
        time_all = data['time_per_step'].values

        # Calculate α for increasing N ranges
        alphas = []
        N_max_vals = []

        for i in range(4, len(N_all) + 1):
            N_subset = N_all[:i]
            time_subset = time_all[:i]
            alpha = fit_power_law(N_subset, time_subset)
            alphas.append(alpha)
            N_max_vals.append(N_subset[-1])

        ax.plot(N_max_vals, alphas, marker=markers[method], label=method,
               color=colors[method], linewidth=2, markersize=6)

    ax.set_xlabel('Maximum N in Fit Range')
    ax.set_ylabel('Scaling Exponent α')
    ax.set_title('(d) Scaling Exponent vs N Range')
    ax.set_xscale('log')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=1, label='Ideal N log N')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.6, 1.1])

    plt.tight_layout()

    # Save
    output_file = FIG_DIR / "scalability_analysis_clean.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def print_summary():
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Scalability Analysis Summary")
    print("=" * 70)

    for method in df['method'].unique():
        data = df[df['method'] == method]
        N = data['N'].values
        time = data['time_per_step'].values

        # Overall α
        alpha = fit_power_law(N, time)

        # α for large N only (N >= 1024)
        large_data = data[data['N'] >= 1024]
        alpha_large = fit_power_law(large_data['N'].values, large_data['time_per_step'].values)

        # Decomposition
        a, b, c = fit_decomposition(N, time)

        print(f"\n{method}:")
        print(f"  Overall scaling exponent:       α = {alpha:.3f}")
        print(f"  Large N scaling (N ≥ 1024):     α = {alpha_large:.3f}")
        print(f"  Decomposition coefficients:")
        print(f"    a (N log N term):  {a:.3e}")
        print(f"    b (N term):        {b:.3e}")
        print(f"    c (constant):      {c:.3e}")

        # Calculate N log N contribution at largest N
        N_max = N[-1]
        t_nlogn = a * N_max * np.log(N_max)
        t_total = a * N_max * np.log(N_max) + b * N_max + c
        pct_nlogn = 100 * t_nlogn / t_total if t_total > 0 else 0

        print(f"  At N = {N_max}:")
        print(f"    N log N contribution: {pct_nlogn:.1f}%")

    print("=" * 70)


def main():
    """Generate all plots."""
    print("=" * 70)
    print("Generating Scalability Plots")
    print("=" * 70)

    create_scalability_plot()
    print_summary()

    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
