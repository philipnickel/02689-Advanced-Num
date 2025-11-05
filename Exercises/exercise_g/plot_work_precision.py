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
sns.set_theme(style="darkgrid")

df = pd.read_parquet(DATA_DIR / "work_precision.parquet")

# %% Create work-precision plot ----------------------------------------------
# Compute convergence rates for labels (using unique dt values)
methods = df['method'].unique()
markers = {'RK3': 's', 'RK4': 'o'}

labels = {}
for method in methods:
    method_data = df[df['method'] == method]
    unique_data = method_data.groupby('dt').first().reset_index()

    dt_vals = unique_data['dt'].values
    err_vals = unique_data['error_l2'].values
    log_dt = np.log(dt_vals)
    log_err = np.log(err_vals)
    coeffs = np.polyfit(log_dt, log_err, 1)
    rate = coeffs[0]
    order = 3 if method == "RK3" else 4

    labels[method] = f'{method} (order {order}, rate={rate:.2f})'

# Aggregate data: compute mean and std for wall_time at each error_l2 level
fig, ax = plt.subplots(figsize=(8, 6))

colors = {'RK3': '#ff7f0e', 'RK4': '#1f77b4'}

for method in methods:
    method_data = df[df['method'] == method]

    # Aggregate: compute mean and std of wall_time for each dt (same error_l2)
    agg_data = method_data.groupby('error_l2').agg({
        'wall_time': ['mean', 'std']
    }).reset_index()
    agg_data.columns = ['error_l2', 'wall_time_mean', 'wall_time_std']

    # Sort by wall_time for proper line plotting
    agg_data = agg_data.sort_values('wall_time_mean')

    x = agg_data['wall_time_mean'].values
    y = agg_data['error_l2'].values
    x_std = agg_data['wall_time_std'].values

    # Plot line
    ax.plot(x, y, marker=markers[method], label=labels[method],
            color=colors[method], linewidth=2, markersize=8, alpha=0.8)

    # Plot shaded error band
    ax.fill_betweenx(y, x - x_std, x + x_std,
                     color=colors[method], alpha=0.2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Wall Time (s)', fontsize=12)
ax.set_ylabel('L² Error', fontsize=12)
ax.set_title('Work-Precision Diagram: RK3 vs RK4', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)

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

for method in df['method'].unique():
    data = df[df['method'] == method].sort_values('dt')

    # Best accuracy
    best_idx = data['error_l2'].idxmin()
    best = data.loc[best_idx]
    print(f"\n{method}:")
    print(f"  Best accuracy: L² = {best['error_l2']:.3e} in {best['wall_time']:.3f}s")

    # Convergence rate
    dt_vals = data['dt'].values
    err_vals = data['error_l2'].values
    log_dt = np.log(dt_vals)
    log_err = np.log(err_vals)
    rate = np.polyfit(log_dt, log_err, 1)[0]
    expected_rate = 3 if method == "RK3" else 4
    print(f"  Convergence rate: {rate:.2f} (expected {expected_rate:.0f})")

print("=" * 70)
print("\n✓ Work-precision plot generated!")
