"""
KdV Spatial Convergence (Complex Differentiation)
=================================================

Visualizes spatial convergence results generated with the complex-valued
Fourier differentiation matrix. Temporal plots are omitted in this variant.
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

print("Creating spatial convergence plots (complex representation)...")

spatial_path = DATA_DIR / "kdv_spatial_convergence_complex.parquet"
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
    "KdV Spatial Convergence (Complex DFT)" + "\n" + param_text,
)

spatial_fig_loglog = OUTPUT_DIR / "spatial_convergence_loglog_complex.pdf"
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
    "KdV Spatial Convergence (Complex DFT)" + "\n" + param_text,
)

spatial_fig_semilog = OUTPUT_DIR / "spatial_convergence_semilog_complex.pdf"
plt.savefig(spatial_fig_semilog, dpi=300, bbox_inches="tight")
print(f"Saved: {spatial_fig_semilog}")

print("\nTemporal convergence plot omitted (complex-differentiation variant).")
