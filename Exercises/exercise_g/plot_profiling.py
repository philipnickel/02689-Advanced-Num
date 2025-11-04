"""
Profiling plots
===============

Visualize timing statistics and profiler hotspots for RK3 vs RK4.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root

sns.set_context("talk")


def _format_function_label(func: str) -> str:
    """Shorten the ``module:lineno(function)`` label for plotting."""
    from pathlib import Path

    module, rest = func.split(":", maxsplit=1)
    lineno, func_name = rest.split("(")
    func_name = func_name.rstrip(")")
    module_short = Path(module).name
    if module_short.endswith(".py"):
        module_short = module_short[:-3]
    return f"{module_short}:{lineno} {func_name}()"


def _add_footer(fig: plt.Figure, text: str) -> None:
    """Add a centered footer annotation below the plot."""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=10)


if __name__ == "__main__":
    repo_root = get_repo_root()
    data_dir = repo_root / "data/A2/ex_g"
    save_dir = repo_root / "figures/A2/ex_g"
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "profiling_summary.parquet"
    functions_path = data_dir / "profiling_functions.parquet"

    if not summary_path.exists() or not functions_path.exists():
        raise FileNotFoundError(
            "Profiling data missing. Run compute_profiling.py before plotting."
        )

    df_summary = pd.read_parquet(summary_path)
    df_functions = pd.read_parquet(functions_path)

    # ------------------------------------------------------------------ #
    # Plot timing summary
    # ------------------------------------------------------------------ #
    params_text = (
        rf"$N = {df_summary['N'].iloc[0]}$, "
        rf"$L = {df_summary['L'].iloc[0]}$, "
        rf"$T = {df_summary['T'].iloc[0]}$"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Profiling RK3 vs RK4\n{params_text}", fontsize=14)

    method_order = df_summary["method"].tolist()
    sns.barplot(
        data=df_summary,
        x="method",
        y="wall_time_s",
        order=method_order,
        color="steelblue",
        ax=ax1,
    )
    ax1.set_ylabel("Wall time [s]")
    ax1.set_xlabel("")
    ax1.set_title("Total runtime")
    ax1.grid(axis="y", alpha=0.3)

    for patch, method in zip(ax1.patches, method_order):
        row = df_summary.loc[df_summary["method"] == method].iloc[0]
        height = patch.get_height()
        ax1.text(
            patch.get_x() + patch.get_width() / 2.0,
            height,
            f"{row['mean_step_time_s'] * 1e3:.2f} ms/step",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    # ------------------------------------------------------------------ #
    # Plot top functions
    # ------------------------------------------------------------------ #
    top_funcs = (
        df_functions.sort_values(["cumtime", "tottime"], ascending=False)
        .head(12)
        .copy()
    )
    top_funcs["label"] = top_funcs["function"].apply(_format_function_label)

    sns.barplot(
        data=top_funcs,
        y="label",
        x="cumtime",
        hue="method",
        palette="deep",
        ax=ax2,
    )
    ax2.set_xlabel("Cumulative time [s]")
    ax2.set_ylabel("")
    ax2.set_title("Top profiler entries")
    ax2.legend(title="Method")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.tight_layout()
    output = save_dir / "profiling_analysis.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved profiling analysis to {output}")
