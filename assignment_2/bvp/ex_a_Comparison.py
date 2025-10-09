"""Comparison of Legendre Tau and collocation solvers for Assignment 2, Exercise 1a."""

from __future__ import annotations

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.rcParams.setdefault("text.usetex", False)

from assignment_1.PolynomialMethods.exercise_j import generalized_vandermonde
from assignment_2.bvp.ex_a_tau import solve_legendre_tau
from assignment_2.bvp.ex_a_col import solve_legendre_collocation
from utils import plot_style  # noqa: F401
plt.rcParams["text.usetex"] = False


def exact_solution(x: np.ndarray, epsilon: float) -> np.ndarray:
    numerator = np.exp(-x / epsilon) + (x - 1.0) - np.exp(-1.0 / epsilon) * x
    denominator = np.exp(-1.0 / epsilon) - 1.0
    return numerator / denominator


def reference_to_physical(xi: np.ndarray) -> np.ndarray:
    return 0.5 * (xi + 1.0)


def evaluate_legendre_series(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    degree = coeffs.size - 1
    vandermonde = generalized_vandermonde(xi, degree)
    return vandermonde @ coeffs


def main() -> None:
# %% Symbolic verification
    x_sym, eps_sym = sp.symbols("x eps", positive=True)
    exact_sym = (
        sp.exp(-x_sym / eps_sym)
        + (x_sym - 1)
        - sp.exp(-1 / eps_sym) * x_sym
    ) / (sp.exp(-1 / eps_sym) - 1)
    residual = sp.simplify(-eps_sym * sp.diff(exact_sym, x_sym, 2) - sp.diff(exact_sym, x_sym) - 1)
    bc_left = sp.simplify(exact_sym.subs(x_sym, 0))
    bc_right = sp.simplify(exact_sym.subs(x_sym, 1))
    print("Symbolic residual:", residual)
    print("u(0) =", bc_left, ", u(1) =", bc_right)

# %% Precompute coefficients
    epsilons = (1e-1, 1e-2, 1e-3)
    modes_plot = 30
    xi_dense = np.linspace(-1.0, 1.0, 2001)
    x_dense = reference_to_physical(xi_dense)
    coeffs_tau: dict[float, np.ndarray] = {}
    coeffs_coll: dict[float, np.ndarray] = {}
    for eps in epsilons:
        coeffs_tau[eps] = solve_legendre_tau(eps, modes_plot)
        coeffs_coll[eps] = solve_legendre_collocation(eps, modes_plot)[1]

# %% Solution profile for reference epsilon
    reference_eps = epsilons[0]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(x_dense, exact_solution(x_dense, reference_eps), label="Exact")
    ax.plot(x_dense, evaluate_legendre_series(coeffs_tau[reference_eps], xi_dense), "--", label="Legendre Tau")
    ax.plot(x_dense, evaluate_legendre_series(coeffs_coll[reference_eps], xi_dense), ":", label="Legendre collocation")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(rf"Solution comparison, $\epsilon={reference_eps}$")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig("assignment_2/Figures/ex_a_solution.pdf", bbox_inches="tight")

# %% Coefficient decay
    fig, ax = plt.subplots(figsize=(6, 3.2))
    modes = np.arange(modes_plot)
    ax.loglog(modes[1:], np.abs(coeffs_tau[reference_eps])[1:], marker="o", markersize=4, markerfacecolor="none", linestyle="-", label="Tau coefficients")
    ax.loglog(modes[1:], np.abs(coeffs_coll[reference_eps])[1:], marker="s", markersize=4, markerfacecolor="none", linestyle="--", label="Collocation coefficients")
    ax.set_xlabel("Legendre mode n")
    ax.set_ylabel(r"$|c_n|")
    ax.set_title(rf"Coefficient decay, $\epsilon={reference_eps}$")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig("assignment_2/Figures/ex_a_coefficients.pdf", bbox_inches="tight")

# %% Error profiles for all epsilons
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    tau_lines = []
    coll_lines = []
    for eps in epsilons:
        exact_vals = exact_solution(x_dense, eps)
        (tau_line,) = ax.semilogy(
            x_dense,
            np.abs(evaluate_legendre_series(coeffs_tau[eps], xi_dense) - exact_vals),
            linestyle="-",
            linewidth=1.6,
        )
        (coll_line,) = ax.semilogy(
            x_dense,
            np.abs(evaluate_legendre_series(coeffs_coll[eps], xi_dense) - exact_vals),
            color=tau_line.get_color(),
            linestyle="--",
            linewidth=1.6,
        )
        tau_lines.append((eps, tau_line))
        coll_lines.append((eps, coll_line))

    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u_{\mathrm{num}} - u_{\mathrm{exact}}|")
    ax.set_title("Error profiles for Tau vs Collocation")
    legend_eps = ax.legend(
        [line for _, line in tau_lines],
        [rf"$\epsilon={eps:g}$" for eps, _ in tau_lines],
        loc="upper right",
        frameon=False,
        title="Epsilon",
    )
    ax.add_artist(legend_eps)
    ax.legend(
        [tau_lines[0][1], coll_lines[0][1]],
        ["Tau", "Collocation"],
        loc="lower left",
        frameon=False,
        title="Method",
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 8.0)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig("assignment_2/Figures/ex_a_errors.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
