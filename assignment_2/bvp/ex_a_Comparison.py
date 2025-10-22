# assignment_2/BVP/ex_a_Comparison_pd.py
"""Legendre Tau vs Collocation (ε-faceted with seaborn.relplot)."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from assignment_1.PolynomialMethods.exercise_j import generalized_vandermonde
from .ex_a_tau import solve_legendre_tau
from .ex_a_col import solve_legendre_collocation

plt.style.use("ana.mplstyle")


def exact_solution(x: np.ndarray, eps: float) -> np.ndarray:
    num = np.exp(-x / eps) + (x - 1.0) - np.exp(-1.0 / eps) * x
    den = np.exp(-1.0 / eps) - 1.0
    return num / den


def eval_legendre_series(c: np.ndarray, xi: np.ndarray) -> np.ndarray:
    V = generalized_vandermonde(xi, c.size - 1)
    return V @ c


def main() -> None:
    epsilons = (1e-1, 1e-2, 1e-3)
    N = 30
    xi = np.linspace(-1.0, 1.0, 2001)
    x = 0.5 * (xi + 1.0)

    # Compute coefficients once per epsilon
    coeff_tau = {e: solve_legendre_tau(e, N) for e in epsilons}
    coeff_col = {e: solve_legendre_collocation(e, N)[1] for e in epsilons}

    # === 1) Combined solution DataFrame (for all epsilons) =====================
    frames = []
    for e in epsilons:
        frames += [
            pd.DataFrame(
                {"x": x, "u": exact_solution(x, e), "method": "Exact", "epsilon": e}
            ),
            pd.DataFrame(
                {
                    "x": x,
                    "u": eval_legendre_series(coeff_tau[e], xi),
                    "method": "Tau",
                    "epsilon": e,
                }
            ),
            pd.DataFrame(
                {
                    "x": x,
                    "u": eval_legendre_series(coeff_col[e], xi),
                    "method": "Collocation",
                    "epsilon": e,
                }
            ),
        ]
    df_sol = pd.concat(frames, ignore_index=True)

    # Plot all ε side by side using seaborn facets
    g = sns.relplot(
        data=df_sol,
        x="x",
        y="u",
        hue="method",
        style="method",
        kind="line",
        col="epsilon",
        col_wrap=3,
        facet_kws=dict(sharey=False, sharex=True),
    )
    g.set_titles(r"$\epsilon={col_name:g}$")
    g.set_axis_labels("x", "u(x)")
    g.figure.suptitle(r"Accuracy of methods for different $\epsilon$ ")
    g.figure.savefig(
        "assignment_2/Figures/BVP/ex_a_solution_facet.pdf", bbox_inches="tight"
    )

    # === 2) Coefficient decay (single epsilon for clarity) ==========================
    ref = epsilons[0]
    k = np.arange(N)
    df_coef = pd.DataFrame(
        {
            "mode": np.r_[k[1:], k[1:]],
            "abs_c": np.r_[np.abs(coeff_tau[ref])[1:], np.abs(coeff_col[ref])[1:]],
            "method": ["Tau"] * (N - 1) + ["Collocation"] * (N - 1),
            "epsilon": ref,
        }
    )
    ax = sns.relplot(
        data=df_coef,
        x="mode",
        y="abs_c",
        hue="method",
        style="method",
        kind="line",
        facet_kws=dict(sharex=True),
    )
    ax.set(xscale="log", yscale="log", xlabel="Legendre mode n", ylabel=r"$|c_n|$")
    ax.figure.suptitle(rf"Coefficient decay, $\epsilon={ref}$", y=1.02)
    ax.figure.savefig(
        "assignment_2/Figures/BVP/ex_a_coefficients.pdf", bbox_inches="tight"
    )

    # === 3) Error profiles (all epsilon) ============================================
    frames = []
    for e in epsilons:
        u_ex = exact_solution(x, e)
        frames += [
            pd.DataFrame(
                {
                    "x": x,
                    "error": np.abs(eval_legendre_series(coeff_tau[e], xi) - u_ex) ** 2,
                    "method": "Tau",
                    "epsilon": e,
                }
            ),
            pd.DataFrame(
                {
                    "x": x,
                    "error": np.abs(eval_legendre_series(coeff_col[e], xi) - u_ex) ** 2,
                    "method": "Collocation",
                    "epsilon": e,
                }
            ),
        ]
    df_err = pd.concat(frames, ignore_index=True)

    g = sns.relplot(
        data=df_err,
        x="x",
        y="error",
        hue="method",
        style="method",
        kind="line",
        col="epsilon",
        col_wrap=3,
        facet_kws=dict(sharey=False),
    )
    g.set(yscale="log", xlabel="x", ylabel=r"$|u_{\rm num}-u_{\rm exact}|$")
    g.set_titles(r"$\epsilon={col_name:g}$")
    g.figure.suptitle("Error profiles for Tau vs Collocation", y=1.02)
    g.figure.savefig(
        "assignment_2/Figures/BVP/ex_a_errors_facet.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
