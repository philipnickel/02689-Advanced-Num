"""
Work-Precision Plot: Error vs Computational Cost
=================================================

Creates a work-precision diagram showing L² error vs wall time,
demonstrating which integrator is more efficient for given accuracy.
"""

# %% Imports and setup -------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_g")
FIG_DIR = Path("figures/A2/ex_g")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set seaborn theme for consistent styling
sns.set_theme(style="whitegrid")

df = pd.read_parquet(DATA_DIR / "work_precision.parquet")

# Filter out NaN errors (unstable runs)
df = df[np.isfinite(df['error_l2'])]

# %% Aggregate data manually -------------------------------------------------
# Group by method and error_l2, compute mean and std of wall_time
agg_data = df.groupby(['method', 'error_l2']).agg({
    'wall_time': ['mean', 'std']
}).reset_index()
agg_data.columns = ['method', 'error_l2', 'wall_time_mean', 'wall_time_std']

# Sort by wall_time for proper line plotting
agg_data = agg_data.sort_values(['method', 'wall_time_mean'])

# %% Create single work-precision plot ---------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Colors for methods
colors = {'RK3': '#ff7f0e', 'RK4': '#1f77b4'}

for method in ['RK3', 'RK4']:
    data = agg_data[agg_data['method'] == method]

    x = data['wall_time_mean'].values
    y = data['error_l2'].values
    x_std = data['wall_time_std'].values

    # Plot line
    ax.plot(x, y, '-', color=colors[method], linewidth=2.5,
            label=method, alpha=0.9)

    # Plot shaded error band (horizontal since x varies)
    ax.fill_betweenx(y, x - x_std, x + x_std,
                     color=colors[method], alpha=0.2)

# Add reference lines for theoretical convergence rates
# Reference point (use middle of RK3 data)
rk3_data = agg_data[agg_data['method'] == 'RK3']
idx_ref = len(rk3_data) // 2
ref_time = rk3_data.iloc[idx_ref]['wall_time_mean']
ref_error = rk3_data.iloc[idx_ref]['error_l2']

# Create reference lines
# For fixed spatial resolution, error ~ dt^p and time ~ 1/dt
# So error ~ time^(-p), or log(error) = -p*log(time) + const

# RK3: 3rd order
time_ref_3 = np.array([ref_time * 0.3, ref_time * 3.0])
error_ref_3 = ref_error * (time_ref_3 / ref_time)**(-3)

# RK4: 4th order
time_ref_4 = np.array([ref_time * 0.3, ref_time * 3.0])
error_ref_4 = ref_error * (time_ref_4 / ref_time)**(-4)

ax.plot(time_ref_3, error_ref_3, '--', color='#ff7f0e', alpha=0.4,
        linewidth=1.5, label='O(dt³) reference')
ax.plot(time_ref_4, error_ref_4, '--', color='#1f77b4', alpha=0.4,
        linewidth=1.5, label='O(dt⁴) reference')

# Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Wall Time (s)', fontsize=13)
ax.set_ylabel('L² Error', fontsize=13)
ax.set_title('Work-Precision Diagram: RK3 vs RK4', fontsize=15, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3, which='both', linestyle='--')

# Legend
ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)

plt.tight_layout()

# %% Save --------------------------------------------------------------------
output_file = FIG_DIR / "work_precision.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# %% Print summary -----------------------------------------------------------
print("\n" + "=" * 70)
print("Work-Precision Analysis Summary")
print("=" * 70)

for method in ['RK3', 'RK4']:
    data = df[df['method'] == method]

    # Best accuracy
    best_idx = data['error_l2'].idxmin()
    best = data.loc[best_idx]

    # Worst accuracy
    worst_idx = data['error_l2'].idxmax()
    worst = data.loc[worst_idx]

    print(f"\n{method}:")
    print(f"  Best accuracy:  L² = {best['error_l2']:.3e} in {best['wall_time']:.3f}s")
    print(f"  Worst accuracy: L² = {worst['error_l2']:.3e} in {worst['wall_time']:.3f}s")
    print(f"  Total configurations: {len(data.groupby(['N', 'dt']))}")

print("=" * 70)
print("\n✓ Work-precision plot generated!")
