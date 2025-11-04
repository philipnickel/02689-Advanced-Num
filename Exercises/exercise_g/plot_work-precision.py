"""
Work-precision plots
====================

Visualize the trade-off between accuracy and computational work for RK3 vs RK4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root

sns.set_context("talk")

if __name__ == "__main__":
    repo_root = get_repo_root()
    data_dir = repo_root / "data/A2/ex_g"
    save_dir = repo_root / "figures/A2/ex_g"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "work_precision.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Work-precision data missing. Run compute_work-precision.py before plotting."
        )

    df = pd.read_parquet(data_path)
    df.sort_values(["method", "dt"], inplace=True)

    # ------------------------------------------------------------------ #
    # Build plots
    # ------------------------------------------------------------------ #
    params_text = (
        rf"$N = {int(df['N'].iloc[0])}$, "
        rf"$L = {df['L'].iloc[0]}$, "
        rf"$T = {df['T'].iloc[0]}$"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Work vs Precision (single soliton)\n{params_text}", fontsize=14)

    sns.lineplot(
        data=df,
        x="wall_time_s",
        y="error_l2",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        ax=ax1,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Wall time [s]")
    ax1.set_ylabel(r"$L^2$ error")
    ax1.set_title("Work vs precision")
    ax1.grid(True, which="both", alpha=0.3)

    # Annotate dt scales
    for method, sub in df.groupby("method"):
        for _, row in sub.iterrows():
            ax1.annotate(
                f"{row['dt_scale']:.2f}",
                (row["wall_time_s"], row["error_l2"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

    sns.lineplot(
        data=df,
        x="dt",
        y="error_l2",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        ax=ax2,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Time step $\Delta t$")
    ax2.set_ylabel(r"$L^2$ error")
    ax2.set_title("Convergence with timestep")
    ax2.grid(True, which="both", alpha=0.3)
    if ax2.legend_ is not None:
        ax2.legend_.set_title("Method")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = save_dir / "work_precision.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved work-precision analysis to {output}")
