"""
Two-Soliton Spatial Convergence Plot
=====================================

Visualizes spatial convergence study comparing aliased vs dealiased for
two-soliton collision.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root
from spectral.utils.io import ensure_output_dir

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_e"
OUTPUT_DIR = ensure_output_dir(repo_root / "figures/A2/ex_e")

print("Loading convergence data...")

convergence_path = DATA_DIR / "convergence.parquet"
df = pd.read_parquet(convergence_path)

print(f"Loaded {len(df)} data points")

# Create convergence plot
fig, ax = plt.subplots(figsize=(7, 5))

sns.lineplot(
    data=df,
    x="N",
    y="L2_error",
    hue="Treatment",
    style="Treatment",
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=8,
    ax=ax,
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"Number of modes ($N$)", fontsize=12)
ax.set_ylabel(r"$L^2$ Error", fontsize=12)

ax.grid(True, alpha=0.3, which="both", linestyle=":")
ax.legend(fontsize=10, framealpha=0.9)

# Add parameter information to title
N_min = df["N"].min()
N_max = df["N"].max()
L_val = df["L"].iloc[0]
T_val = df["T"].iloc[0]
C_val = df["C"].iloc[0]

param_text = (
    rf"Single soliton: $c={C_val:.1f}$" + "\n" +
    rf"$N \in [{N_min}, {N_max}]$, $L = {L_val:.1f}$, $T = {T_val:.1f}$"
)

ax.set_title(
    "KdV Spatial Convergence" + "\n" + param_text,
    fontsize=13,
    pad=15
)

# Save figure
convergence_fig = OUTPUT_DIR / "ex_e_convergence.pdf"
plt.savefig(convergence_fig, dpi=300, bbox_inches="tight")
print(f"\nSaved: {convergence_fig}")
