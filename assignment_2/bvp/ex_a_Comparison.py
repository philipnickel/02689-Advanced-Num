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


def eval_legendre_series(c: np.ndarray, xi: np.ndarray) -> np.ndarray:
    V = generalized_vandermonde(xi, c.size - 1)
    return V @ c


def main() -> None:
    epsilons = (1e-1, 1e-2, 1e-3)
    N = 30
    xi = np.linspace(-1.0, 1.0, 2001)
    x = 0.5 * (xi + 1.0)

    # Solve once per epsilon
    coeff_tau = {e: solve_legendre_tau(e, N) for e in epsilons}
    coeff_col = {e: solve_legendre_collocation(e, N)[1] for e in epsilons}

    # === 1) Solution comparison (single epsilon) ===============================
    ref = epsilons[0]
    df_sol = pd.DataFrame(
        {
            "x": np.tile(x, 3),
            "u": np.concatenate(
                [
                    exact_solution(x, ref),
                    eval_legendre_series(coeff_tau[ref], xi),
                    eval_legendre_series(coeff_col[ref], xi),
                ]
            ),
            "method": np.repeat(["Exact", "Tau", "Collocation"], len(x)),
        }
    )
    ax = sns.lineplot(data=df_sol, x="x", y="u", hue="method", style="method")
    ax.set(title=rf"Solution comparison, $\epsilon={ref}$", xlabel="x", ylabel="u(x)")
    plt.savefig("assignment_2/Figures/BVP/ex_a_solution.pdf")

    # === 2) Coefficient decay (single epsilon) =================================
    k = np.arange(N)
    df_coef = pd.DataFrame(
        {
            "mode": np.r_[k[1:], k[1:]],
            "abs_c": np.r_[np.abs(coeff_tau[ref])[1:], np.abs(coeff_col[ref])[1:]],
            "method": ["Tau"] * (N - 1) + ["Collocation"] * (N - 1),
        }
    )
    ax = sns.lineplot(data=df_coef, x="mode", y="abs_c", hue="method", style="method")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Legendre mode n",
        ylabel=r"$|c_n|$",
        title=rf"Coefficient decay, $\epsilon={ref}$",
    )
    plt.savefig("assignment_2/Figures/BVP/ex_a_coefficients.pdf")

    # === 3) Error profiles (all epsilons) ======================================
    frames = []
    for e in epsilons:
        u_ex = exact_solution(x, e)
        err_tau = np.abs(eval_legendre_series(coeff_tau[e], xi) - u_ex)
        err_col = np.abs(eval_legendre_series(coeff_col[e], xi) - u_ex)
        frames += [
            pd.DataFrame({"x": x, "error": err_tau, "method": "Tau", "epsilon": e}),
            pd.DataFrame(
                {"x": x, "error": err_col, "method": "Collocation", "epsilon": e}
            ),
        ]
    df_err = pd.concat(frames, ignore_index=True)

    ax = sns.lineplot(data=df_err, x="x", y="error", hue="epsilon", style="method")
    ax.set(
        yscale="log",
        xlabel="x",
        ylabel=r"$|u_{\rm num}-u_{\rm exact}|$",
        title="Error profiles for Tau vs Collocation",
    )
    plt.savefig("assignment_2/Figures/BVP/ex_a_errors.pdf")


if __name__ == "__main__":
    main()
