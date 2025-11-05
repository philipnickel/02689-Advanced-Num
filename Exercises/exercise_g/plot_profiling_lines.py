"""
Plot line-by-line profiling results from line_profiler.

Usage
-----
    uv run python Exercises/exercise_g/plot_profiling_lines.py
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
data_path = repo_root / "data/A2/ex_g/line_profiler_data.parquet"
save_dir = repo_root / "figures/A2/ex_g"
save_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)

# Filter out function call overhead
df_work = df[df['category'] != 'Function Call'].copy()

# %% Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# %% Plot 1: Time breakdown by category (pie chart)
category_time = df_work.groupby('category')['time_s'].sum().sort_values(ascending=False)
category_pct = (category_time / category_time.sum()) * 100

# Combine small categories into "Other" if needed
if len(category_time) > 5:
    major_cats = category_time.head(4)
    other_sum = category_time.iloc[4:].sum()
    category_time = pd.concat([major_cats, pd.Series({'Other': other_sum})])
    category_pct = (category_time / category_time.sum()) * 100

colors = sns.color_palette("deep", len(category_time)) if sns else None

axes[0].pie(
    category_pct,
    labels=category_pct.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
)
axes[0].set_title('Time Distribution by Operation Category')

# %% Plot 2: Top 10 slowest lines
top_lines = df_work.nlargest(10, 'time_s').copy()
top_lines['time_ms'] = top_lines['time_s'] * 1000

# Create short labels
top_lines['label'] = (
    top_lines['category'] + ': ' +
    top_lines['source'].str[:35]
)

# Sort by time for plotting
top_lines = top_lines.sort_values('time_ms', ascending=True)

if sns:
    sns.barplot(
        data=top_lines,
        y='label',
        x='time_ms',
        hue='category',
        dodge=False,
        ax=axes[1],
    )
    axes[1].legend_.remove()
else:
    y_pos = range(len(top_lines))
    axes[1].barh(y_pos, top_lines['time_ms'])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_lines['label'])

axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('')
axes[1].set_title('Top 10 Slowest Lines')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()

# %% Save figure
output_path = save_dir / "line_profiling.pdf"
fig.savefig(output_path, bbox_inches='tight')
print(f"✓ Saved to {output_path}")

# %%
