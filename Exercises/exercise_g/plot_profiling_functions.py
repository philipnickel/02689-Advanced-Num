"""
Plot function-level profiling results from cProfile.

Usage
-----
    uv run python Exercises/exercise_g/plot_profiling_functions.py
"""

# %% Imports
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_context("talk")
    sns.set_palette("deep")
except ModuleNotFoundError:
    sns = None

from spectral.utils.plotting import get_repo_root

# %% Load data
repo_root = get_repo_root()
data_path = repo_root / "data/A2/ex_g/cprofile_functions.parquet"
save_dir = repo_root / "figures/A2/ex_g"
save_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)

# Filter to key functions only
key_functions = ['rhs', 'step', 'fft', 'ifft', 'solve']
df_key = df[df['function'].isin(key_functions)].copy()

# %% Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# %% Plot 1: Total time by function for each method
func_time = df_key.groupby(['method', 'function']).agg({
    'tottime': 'sum'
}).reset_index()

func_time['tottime_ms'] = func_time['tottime'] * 1000

if sns:
    sns.barplot(
        data=func_time,
        x='function',
        y='tottime_ms',
        hue='method',
        ax=axes[0],
    )
else:
    # Fallback matplotlib
    methods = func_time['method'].unique()
    functions = func_time['function'].unique()
    x = range(len(functions))
    width = 0.35

    for i, method in enumerate(methods):
        method_data = func_time[func_time['method'] == method]
        values = [method_data[method_data['function'] == f]['tottime_ms'].iloc[0]
                 if len(method_data[method_data['function'] == f]) > 0 else 0
                 for f in functions]
        axes[0].bar([p + i*width for p in x], values, width, label=method)

    axes[0].set_xticks([p + width/2 for p in x])
    axes[0].set_xticklabels(functions)
    axes[0].legend()

axes[0].set_ylabel('Time (ms)')
axes[0].set_xlabel('Function')
axes[0].set_title('Total Time by Function')
axes[0].grid(axis='y', alpha=0.3)

# %% Plot 2: Scaling with N (focus on rhs function)
rhs_data = df[df['function'] == 'rhs'].copy()
rhs_data['tottime_ms'] = rhs_data['tottime'] * 1000

if sns:
    sns.lineplot(
        data=rhs_data,
        x='N',
        y='tottime_ms',
        hue='method',
        marker='o',
        markersize=8,
        ax=axes[1],
    )
else:
    for method in rhs_data['method'].unique():
        method_data = rhs_data[rhs_data['method'] == method].sort_values('N')
        axes[1].plot(method_data['N'], method_data['tottime_ms'],
                    marker='o', label=method)
    axes[1].legend()

axes[1].set_xlabel('Grid Resolution (N)')
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('RHS Function Scaling')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# %% Save figure
output_path = save_dir / "function_profiling.pdf"
fig.savefig(output_path, bbox_inches='tight')
print(f"✓ Saved to {output_path}")

# %%
