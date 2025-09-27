"""Legendre Vandermonde transforms and interpolation (exercise j)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre

from assignment_1.PolynomialMethods.exercise_h import legendre_polynomials


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR.parent / "Plots" / "PolynomialMethods" / "exercise_j"
PLOT_DIR.mkdir(parents=True, exist_ok=True)



def generalized_vandermonde(x: np.ndarray, degree: int | None = None) -> np.ndarray:
    if degree is None:
        degree = x.size - 1
    return legendre_polynomials(x, degree).T


def legendre_gauss_lobatto_nodes(num_nodes: int) -> np.ndarray:
    degree = num_nodes - 1
    roots = Legendre.basis(degree).deriv().roots()
    nodes = np.concatenate(([-1.0], roots, [1.0]))
    return np.sort(nodes)


def lagrange_on_grid(x_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    degree = x_nodes.size - 1
    V_nodes = generalized_vandermonde(x_nodes, degree)
    V_eval = generalized_vandermonde(x_eval, degree)
    identity = np.eye(degree + 1)
    return V_eval @ np.linalg.solve(V_nodes, identity)


def discrete_l2_error(f_exact: np.ndarray, f_num: np.ndarray, interval_length: float) -> float:
    diff = f_num - f_exact
    h = interval_length / f_exact.size
    return np.sqrt(h) * np.linalg.norm(diff)


def main() -> None:
    num_nodes=6
    x_nodes = legendre_gauss_lobatto_nodes(num_nodes)
    x_eval = np.linspace(-1.0, 1.0, 1000, endpoint=False)
    lagrange_vals = lagrange_on_grid(x_nodes, x_eval)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(num_nodes):
        ax.plot(x_eval, lagrange_vals[:, j], label=rf"$h_{{{j}}}(x)$")
    ax.plot(x_nodes, np.zeros_like(x_nodes), "ko", label="LGL nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Legendre-Gauss-Lobatto Lagrange polynomials (N=6)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exercise_j_lagrange.pdf", dpi=200)


    ## Convergence analysis 

    eval_points = 4000
    N_values = np.arange(4, 30, 2)

    x_eval = np.linspace(-1.0, 1.0, eval_points, endpoint=False)
    f_exact = np.sin(np.pi * x_eval)
    errors = []
    for N in N_values:
        x_nodes = legendre_gauss_lobatto_nodes(N)
        degree = N - 1
        V_nodes = generalized_vandermonde(x_nodes, degree)
        nodal_vals = np.sin(np.pi * x_nodes)
        modal = np.linalg.solve(V_nodes, nodal_vals)
        V_eval = generalized_vandermonde(x_eval, degree)
        f_approx = V_eval @ modal
        errors.append(discrete_l2_error(f_exact, f_approx, interval_length=2.0))

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_values, errors, "o-", label=r"$L_2$ error for $\sin(\pi x)$")

    ref = np.exp(1e-2) * N_values ** 2
    ax.loglog(N_values, ref, "--", color="0.6", label=rf"Fit $\mathcal{{O}}(N^{-2})$")
    ax.set_xlabel("Number of LGL nodes")
    ax.set_ylabel(r"$L_2$ error")
    ax.set_title(r"Legendre interpolation of $\sin(\pi x)$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exercise_j_convergence.pdf", dpi=200)

### Approx extrapolation plot

    x_ext = np.linspace(-1.5, 1.5, 400)
    N = 20
    x_nodes = legendre_gauss_lobatto_nodes(N)
    degree = N - 1
    V_nodes = generalized_vandermonde(x_nodes, degree)
    nodal_vals = np.sin(np.pi * x_nodes)
    modal = np.linalg.solve(V_nodes, nodal_vals)

    V_ext = generalized_vandermonde(x_ext, degree)
    approx_ext = V_ext @ modal
    exact_ext = np.sin(np.pi * x_ext)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_ext, exact_ext, label="Exact sin(πx)")
    ax.plot(x_ext, approx_ext, "--", label=f"Legendre modal degree {degree}")
    ax.axvspan(-1.0, 1.0, color="0.9", alpha=0.5, label="Interpolation domain")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Legendre polynomial extrapolation")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exercise_j_extrapolation.pdf", dpi=200)

    if len(errors) >= 2:
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        print("Error ratios N_k / N_{k+1}:", ratios)


if __name__ == "__main__":
    main()
