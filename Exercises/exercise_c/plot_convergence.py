"""
KdV Convergence Studies
=======================

Visualizes spatial and temporal convergence studies for the Fourier KdV solver.
"""

# %%
# Spatial convergence
# -------------------
# Analyze how error decreases with increasing number of modes.

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root
from spectral.utils.io import ensure_output_dir
from spectral.utils.formatting import format_dt_latex

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_c"
OUTPUT_DIR = ensure_output_dir(repo_root / "figures/A2/ex_c")

print("Creating spatial convergence plots...")

spatial_path = DATA_DIR / "kdv_spatial_convergence.parquet"
df_spatial = pd.read_parquet(spatial_path)

# Add parameter information to title
N_min_sp = df_spatial["N"].min()
N_max_sp = df_spatial["N"].max()
dt_sp = df_spatial["dt"].iloc[0]
L_sp = df_spatial["L"].iloc[0]
T_sp = df_spatial["T"].iloc[0]
dt_latex = format_dt_latex(dt_sp)

param_text = rf"\tiny $N \in [{N_min_sp}, {N_max_sp}]$, $\Delta t = {dt_latex}$, $L = {L_sp:.1f}$, $T = {T_sp:.2f}$"

# 1. Log-log plot with reference line
fig, ax = plt.subplots()

sns.lineplot(
    data=df_spatial,
    x="N",
    y="Error",
    hue="method",
    style="dealias",
    markers=True,
    dashes=False,
    ax=ax,
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"Number of modes ($N$)")
ax.set_ylabel(r"$L^2$ error")

# Add O(N^-2) reference line
N_ref = np.array([N_min_sp, N_max_sp])
# Scale the reference line to match the data
error_ref_base = df_spatial["Error"].max() * 10  # Position reference line near the data
error_ref = error_ref_base * (N_ref / N_min_sp) ** (-2)
ax.plot(N_ref, error_ref, 'k--', linewidth=1, alpha=0.5, label=r'$\mathcal{O}(N^{-2})$')
ax.legend()

ax.set_title(
    "KdV Spatial Convergence" + "\n" + param_text,
)

spatial_fig_loglog = OUTPUT_DIR / "spatial_convergence_loglog.pdf"
plt.savefig(spatial_fig_loglog, dpi=300, bbox_inches="tight")
print(f"Saved: {spatial_fig_loglog}")

# 2. Semi-log plot
fig, ax = plt.subplots()

sns.lineplot(
    data=df_spatial,
    x="N",
    y="Error",
    hue="method",
    style="dealias",
    markers=True,
    dashes=False,
    ax=ax,
)

ax.set_yscale("log")
ax.set_xlabel(r"Number of modes ($N$)")
ax.set_ylabel(r"$L^2$ error")

ax.set_title(
    "KdV Spatial Convergence" + "\n" + param_text,
)

spatial_fig_semilog = OUTPUT_DIR / "spatial_convergence_semilog.pdf"
plt.savefig(spatial_fig_semilog, dpi=300, bbox_inches="tight")
print(f"Saved: {spatial_fig_semilog}")

# %%
# Temporal convergence
# --------------------
# Analyze convergence in time for different time integrators.

print("Creating temporal convergence plot...")

temporal_path = DATA_DIR / "kdv_temporal_convergence.parquet"
df_temporal = pd.read_parquet(temporal_path)

fig, ax = plt.subplots()

sns.lineplot(
    data=df_temporal,
    x="dt",
    y="Error",
    hue="method",
    style="dealias",
    markers=True,
    dashes=False,
    ax=ax,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel(r"Time step $\Delta t$")
ax.set_ylabel(r"$L^2$ error")

# Add parameter information to title
N_temp = df_temporal["N"].iloc[0]
dt_min_t = df_temporal["dt"].min()
dt_max_t = df_temporal["dt"].max()
L_temp = df_temporal["L"].iloc[0]
T_temp = df_temporal["T"].iloc[0]
dt_min_latex = format_dt_latex(dt_min_t)
dt_max_latex = format_dt_latex(dt_max_t)

# Add reference lines for O(dt^3) and O(dt^4)
dt_ref = np.array([dt_min_t, dt_max_t])
error_ref_base = df_temporal["Error"].max() * 100  # Position reference lines near the data

# O(dt^3) reference line for RK3
error_ref_3 = error_ref_base * (dt_ref / dt_max_t) ** 3
ax.plot(dt_ref, error_ref_3, 'k:', linewidth=1.5, alpha=0.6, label=r'$\mathcal{O}(\Delta t^3)$')

# O(dt^4) reference line for RK4
error_ref_4 = error_ref_base * (dt_ref / dt_max_t) ** 4
ax.plot(dt_ref, error_ref_4, 'k--', linewidth=1.5, alpha=0.6, label=r'$\mathcal{O}(\Delta t^4)$')

ax.legend()

param_text_temp = rf"\tiny $N = {N_temp}$, $\Delta t \in [{dt_min_latex}, {dt_max_latex}]$, $L = {L_temp:.1f}$, $T = {T_temp:.2f}$"

ax.set_title(
    "KdV Temporal Convergence" + "\n" + param_text_temp,
)

#plt.tight_layout()
temporal_fig = OUTPUT_DIR / "temporal_convergence.pdf"
plt.savefig(temporal_fig, dpi=300)
print(f"Saved: {temporal_fig}")

print("\nAll plots created successfully!")
