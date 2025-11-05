"""
Scalability Analysis Results
=============================

Creates plot showing computational complexity:

- Wall time vs N (comparing all methods: RK4, RK3)
- Reference line showing expected O(N log N) scaling
"""

# %%
# Scalability analysis
# Study how runtime scales with problem size.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:
    sns = None

from spectral.utils.plotting import get_repo_root

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_g"
save_dir = repo_root / "figures/A2/ex_g"
save_dir.mkdir(parents=True, exist_ok=True)

# %% Load data
print("Loading scalability data...")
timing_path = data_dir / "scalability_timing.parquet"
if timing_path.exists():
    try:
        df_timing = pd.read_parquet(timing_path)
    except ImportError:
        timing_path = timing_path.with_suffix(".csv")
        df_timing = pd.read_csv(timing_path)
else:
    timing_path = timing_path.with_suffix(".csv")
    df_timing = pd.read_csv(timing_path)

print(f"  Timing data: {df_timing.shape}")

# %% Create two-panel plot
print("\nCreating scalability analysis plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== Panel 1: Absolute performance (log-log) =====
# Plot data for each method
if sns:
    sns.lineplot(
        data=df_timing,
        x="N",
        y="time_per_step",
        hue="method",
        style="method",
        markers=True,
        markersize=8,
        ax=ax1,
    )
else:
    for method, subset in df_timing.groupby("method"):
        subset = subset.sort_values("N")
        ax1.plot(
            subset["N"],
            subset["time_per_step"],
            marker="o",
            label=method,
        )

# Add reference line: N log N scaling
N_ref = df_timing["N"].unique()
N_ref = np.sort(N_ref)
# Normalize to match first data point of RK3 (fastest method)
first_point = df_timing[(df_timing["N"] == N_ref[0]) & (df_timing["method"] == "RK3")][
    "time_per_step"
].values[0]
n_log_n = N_ref * np.log(N_ref)
n_log_n_scaled = first_point * (n_log_n / n_log_n[0])

ax1.plot(
    N_ref,
    n_log_n_scaled,
    "--",
    linewidth=2,
    alpha=0.7,
    color="gray",
    label=r"$\mathcal{O}(N \log N)$",
)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Number of grid points $N$")
ax1.set_ylabel("Time per timestep [s]")
ax1.set_title("Computational Complexity")
ax1.legend(title="Method", loc="best")
ax1.grid(True, alpha=0.3)

# ===== Panel 2: Normalized efficiency =====
# Compare full step timing vs direct RHS evaluations
methods = df_timing["method"].unique()
if sns:
    palette = sns.color_palette("deep", n_colors=len(methods))
else:
    palette = [plt.cm.tab10(i) for i in range(len(methods))]
colors = dict(zip(methods, palette))

for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method].sort_values("N")
    N_vals = subset["N"].values
    step_vals = subset["time_per_step"].values
    rhs_vals = subset["rhs_time"].values

    normalized_step = step_vals / (N_vals * np.log(N_vals))
    normalized_rhs = rhs_vals / (N_vals * np.log(N_vals))

    ax2.plot(
        N_vals,
        normalized_step,
        marker="o",
        markersize=7,
        linewidth=2,
        color=colors[method],
        alpha=0.85,
        label=f"{method} (step)",
    )

    ax2.plot(
        N_vals,
        normalized_rhs,
        marker="s",
        markersize=6,
        linewidth=1.8,
        linestyle="--",
        color=colors[method],
        alpha=0.85,
        label=f"{method} (rhs)",
    )

ax2.set_xlabel(r"Number of grid points $N$")
ax2.set_ylabel(r"Time / $(N \log N)$ [s]")
ax2.set_title("Scaling Efficiency (step vs. rhs)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.legend(title="Metric", fontsize=9)
ax2.grid(True, alpha=0.3)

# Overall title with parameters
L_val = df_timing["L"].iloc[0] if "L" in df_timing.columns else None
T_val = df_timing["T"].iloc[0] if "T" in df_timing.columns else None
if L_val and T_val:
    fig.suptitle(
        "KdV Scalability" + "\n" +
        rf"$L = {L_val:.1f}$, $T = {T_val:.1f}$",
        fontsize=14,
        y=1.02
    )
else:
    fig.suptitle("KdV Scalability", fontsize=14, y=1.02)

output = save_dir / "scalability_analysis.pdf"
fig.savefig(output, bbox_inches="tight")
print(f"  Saved: {output}")

# %% Summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\nTime per step at N=128:")
for method in df_timing["method"].unique():
    t = df_timing[(df_timing["method"] == method) & (df_timing["N"] == 128)][
        "time_per_step"
    ].values
    if len(t) > 0:
        print(f"  {method}: {t[0]:.6f} s")

print("\nScaling exponent (fit to N^α in log-log space):")
for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method].sort_values("N")
    subset_fit = subset[subset["N"] >= 128]
    if len(subset_fit) >= 2:
        subset = subset_fit
    log_N = np.log(subset["N"].values)
    log_t = np.log(subset["time_per_step"].values)
    # Linear fit in log-log space
    coef = np.polyfit(log_N, log_t, 1)
    print(f"  {method}: α = {coef[0]:.3f} (ideal: ~1.0-1.1 for N log N)")

print("\nRHS-only exponent (direct rhs_time fit):")
for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method].sort_values("N")
    subset_fit = subset[subset["N"] >= 128]
    if len(subset_fit) >= 2:
        subset = subset_fit
    log_N = np.log(subset["N"].values)
    log_rhs = np.log(subset["rhs_time"].values)
    coef = np.polyfit(log_N, log_rhs, 1)
    print(f"  {method}: α_rhs = {coef[0]:.3f}")

print("\nDecomposition: time ≈ a N log N + b N + c")
for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method].sort_values("N")
    subset_fit = subset[subset["N"] >= 128]
    if len(subset_fit) < 2:
        subset_fit = subset
    N_vals = subset_fit["N"].values
    time_vals = subset_fit["time_per_step"].values
    A = np.column_stack([N_vals * np.log(N_vals), N_vals, np.ones_like(N_vals)])
    coeff, *_ = np.linalg.lstsq(A, time_vals, rcond=None)
    a_fft, b_linear, c_const = coeff
    residual = time_vals - (b_linear * N_vals + c_const)
    mask = residual > 0
    fft_slope = (
        np.polyfit(np.log(N_vals[mask]), np.log(residual[mask]), 1)[0]
        if np.count_nonzero(mask) >= 2
        else float("nan")
    )
    print(
        f"  {method}: a={a_fft:.3e}, b={b_linear:.3e}, c={c_const:.3e}, "
        f"α_fft ≈ {fft_slope:.3f}"
    )

print("\nNote: Low scaling exponents are expected for small N ranges.")
print("The O(N log N) behavior from FFT becomes dominant at larger N.")

print("\nPlot created!")
