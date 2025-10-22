# assignment_2/BVP/ex_a_Comparison_pd.py
"""Legendre Tau vs Collocation"""

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


def main() -> None:
    eps = np.array([1e-1, 1e-2, 1e-3])  # (E,)
    N = 50
    xi = np.linspace(-1.0, 1.0, 2001)  # (M,)
    x = 0.5 * (xi + 1.0)  # (M,)
    E, M = eps.size, xi.size

    # --- Vandermonde once ---
    V = generalized_vandermonde(xi, N - 1)  # (M, N)

    # --- Coefficients stacked (vectorizable) ---
    coeff_tau = np.vstack([solve_legendre_tau(e, N) for e in eps])  # (E, N)
    coeff_col = np.vstack([solve_legendre_collocation(e, N)[1] for e in eps])  # (E, N)

    # --- Numerical solutions via one BLAS call each: (E, M) = (E, N) @ (N, M) ---
    VT = V.T  # (N, M)
    U_tau = coeff_tau @ VT  # (E, M)
    U_col = coeff_col @ VT  # (E, M)

    # --- Exact solution for all eps via broadcasting ---
    U_exact = exact_solution(x[None, :], eps[:, None])  # (E, M)

    # ================= 1) Solution comparison (faceted by ε) ===================
    # Build long DF without Python loops
    u_stacked = np.concatenate([U_exact, U_tau, U_col], axis=0)  # (3E, M)
    methods = np.array(["Exact", "Tau", "Collocation"])
    method_labels = np.repeat(methods, E)  # (3E,)
    eps_labels = np.tile(eps, 3)  # (3E,)

    df_sol = pd.DataFrame(
        {
            "x": np.tile(x, 3 * E),  # (3E*M,)
            "u": u_stacked.ravel(),  # (3E*M,)
            "method": np.repeat(method_labels, M),  # (3E*M,)
            "epsilon": np.repeat(eps_labels, M),  # (3E*M,)
        }
    )

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
    g.figure.suptitle(r"Accuracy of methods for different $\epsilon$", y=1.02)
    g.figure.savefig("assignment_2/Figures/BVP/ex_a_solution_facet.pdf")

    # ================= 2) Coefficient decay (all ε) ===========================
    k = np.arange(N)
    frames = []
    for i in range(len(eps)):
        frames.append(
            pd.DataFrame(
                {
                    "mode": k[1:],
                    "abs_c": np.abs(coeff_tau[i])[1:],
                    "method": "Tau",
                    "epsilon": eps[i],
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "mode": k[1:],
                    "abs_c": np.abs(coeff_col[i])[1:],
                    "method": "Collocation",
                    "epsilon": eps[i],
                }
            )
        )
    df_coef_all = pd.concat(frames, ignore_index=True)

    g2 = sns.relplot(
        data=df_coef_all.dropna(subset=["abs_c"]),
        x="mode",
        y="abs_c",
        hue="method",
        style="method",
        kind="line",
        col="epsilon",
        col_wrap=3,
        marker="o",
        dashes=False,
        errorbar=None,
        estimator=None,
    )
    g2.set(xscale="log", yscale="log", xlabel="Legendre mode n", ylabel=r"$|c_n|$")
    g2.figure.savefig(
        "assignment_2/Figures/BVP/ex_a_coefficients_facet.pdf", bbox_inches="tight"
    )

    # ================= 3) Error profiles (abs error, all ε) ====================
    err_tau = np.abs(U_tau - U_exact)  # (E, M)
    err_col = np.abs(U_col - U_exact)  # (E, M)
    err_stacked = np.concatenate([err_tau, err_col], axis=0)  # (2E, M)
    method_err_labels = np.repeat(np.array(["Tau", "Collocation"]), E)  # (2E,)
    eps_err_labels = np.tile(eps, 2)

    df_err = pd.DataFrame(
        {
            "x": np.tile(x, 2 * E),  # (2E*M,)
            "error": err_stacked.ravel(),  # (2E*M,)
            "method": np.repeat(method_err_labels, M),  # (2E*M,)
            "epsilon": np.repeat(eps_err_labels, M),  # (2E*M,)
        }
    )

    g3 = sns.relplot(
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
    g3.set(yscale="log", xlabel="x", ylabel=r"$|u_{\rm num}-u_{\rm exact}|$")
    g3.set_titles(r"$\epsilon={col_name:g}$")
    g3.figure.suptitle("Error profiles for Tau vs Collocation", y=1.02)
    g3.figure.savefig("assignment_2/Figures/BVP/ex_a_errors_facet.pdf")


if __name__ == "__main__":
    main()
