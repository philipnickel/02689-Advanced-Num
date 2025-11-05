"""
Work-precision plots
====================

Visualize the trade-off between accuracy and computational work for RK3 vs RK4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns  # type: ignore

    sns.set_context("talk")
except ModuleNotFoundError:
    sns = None

from spectral.utils.plotting import get_repo_root

if __name__ == "__main__":
    repo_root = get_repo_root()
    data_dir = repo_root / "data/A2/ex_g"
    save_dir = repo_root / "figures/A2/ex_g"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "work_precision.parquet"
    if data_path.exists():
        try:
            df = pd.read_parquet(data_path)
        except ImportError:
            data_path = data_path.with_suffix(".csv")
            df = pd.read_csv(data_path)
    else:
        data_path = data_path.with_suffix(".csv")
        if not data_path.exists():
            raise FileNotFoundError(
                "Work-precision data missing. Run compute_work-precision.py before plotting."
            )
        df = pd.read_csv(data_path)
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
    fig.suptitle(f"KdV Work-Precision\n{params_text}", fontsize=14)

    if sns:
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
    else:
        for method, sub in df.groupby("method"):
            sub = sub.sort_values("wall_time_s")
            ax1.plot(
                sub["wall_time_s"],
                sub["error_l2"],
                marker="o",
                label=method,
            )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Wall time [s]")
    ax1.set_ylabel(r"$L^2$ error")
    ax1.set_title("Work vs precision")
    ax1.grid(True, which="both", alpha=0.3)
    if not sns:
        ax1.legend()

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

    if sns:
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
    else:
        for method, sub in df.groupby("method"):
            sub = sub.sort_values("dt")
            ax2.plot(
                sub["dt"],
                sub["error_l2"],
                marker="o",
                label=method,
            )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Time step $\Delta t$")
    ax2.set_ylabel(r"$L^2$ error")
    ax2.set_title("Convergence with timestep")
    ax2.grid(True, which="both", alpha=0.3)
    if ax2.legend_ is not None:
        ax2.legend_.set_title("Method")
    elif not sns:
        ax2.legend(title="Method")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = save_dir / "work_precision.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved work-precision analysis to {output}")
