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

# Configuration
DATA_DIR = Path("data/A2/ex_g")
FIG_DIR = Path("figures/A2/ex_g")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(DATA_DIR / "scalability_timing.parquet")


def create_scalability_plot():
    """Create scalability plot: time per step vs N."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use seaborn lineplot with log scales
    sns.lineplot(data=df, x='N', y='time_per_step', hue='method',
                 style='method', markers=True, errorbar='sd',
                 linewidth=2, markersize=8, ax=ax)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add O(N log N) reference line
    N_ref = np.array([df['N'].min(), df['N'].max()])
    time_ref_base = df['time_per_step'].max() * 0.5
    N_mid = df['N'].median()
    time_theory = time_ref_base * (N_ref * np.log(N_ref)) / (N_mid * np.log(N_mid))

    ax.plot(N_ref, time_theory, 'k--', linewidth=1.5, alpha=0.5,
            label=r'$\mathcal{O}(N \log N)$')

    ax.set_xlabel('Grid Size N', fontsize=12)
    ax.set_ylabel('Time per Step (s)', fontsize=12)
    ax.set_title('Scalability Analysis: RK3 vs RK4', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save
    output_file = FIG_DIR / "scalability_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")



def main():
    """Generate scalability plot."""
    print("=" * 70)
    print("Generating Scalability Plot")
    print("=" * 70)

    create_scalability_plot()

    print("\n✓ Scalability plot generated!")


if __name__ == "__main__":
    main()
