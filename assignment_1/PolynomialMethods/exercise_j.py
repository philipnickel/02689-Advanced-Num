"""Task j) from Assignment 1: Legendre Vandermonde transforms and interpolation.

The modal-to-nodal mapping follows Lecture 2 (Polynomial Methods), using
V_{ij} = P_j(x_i) with P_j the Legendre polynomials defined by the three-term
recurrence.  Legendre-Gauss-Lobatto nodes include the endpoints and solve
(1-x^2)P'_n(x)=0 as introduced on the interpolation slides.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre

PI = np.pi
BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR.parent / "Plots" / "PolynomialMethods"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def legendre_polynomials(x: np.ndarray, degree: int) -> np.ndarray:
    """Return Legendre polynomials P_0 ... P_degree evaluated at x."""
    x = np.asarray(x)
    vals = np.zeros((degree + 1, x.size))
    vals[0, :] = 1.0
    if degree >= 1:
        vals[1, :] = x
    for n in range(1, degree):
        vals[n + 1, :] = ((2 * n + 1) * x * vals[n, :] - n * vals[n - 1, :]) / (n + 1)
    return vals


def generalized_vandermonde(x: np.ndarray, degree: int | None = None) -> np.ndarray:
    if degree is None:
        degree = x.size - 1
    return legendre_polynomials(x, degree).T


def legendre_gauss_lobatto_nodes(num_nodes: int) -> np.ndarray:
    if num_nodes < 2:
        raise ValueError("Need at least two nodes for LGL grid")
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
    return np.sqrt(h * np.sum(diff ** 2))


def plot_lagrange(num_nodes: int = 6) -> None:
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
    fig.savefig(PLOT_DIR / "exercise_j_lagrange.png", dpi=200)


def convergence_test(N_values: np.ndarray, eval_points: int = 4000) -> np.ndarray:
    x_eval = np.linspace(-1.0, 1.0, eval_points, endpoint=False)
    f_exact = np.sin(PI * x_eval)
    errors = []
    for N in N_values:
        x_nodes = legendre_gauss_lobatto_nodes(N)
        degree = N - 1
        V_nodes = generalized_vandermonde(x_nodes, degree)
        nodal_vals = np.sin(PI * x_nodes)
        modal = np.linalg.solve(V_nodes, nodal_vals)
        V_eval = generalized_vandermonde(x_eval, degree)
        f_approx = V_eval @ modal
        errors.append(discrete_l2_error(f_exact, f_approx, interval_length=2.0))
    return np.array(errors)


def plot_convergence(N_values: np.ndarray, errors: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_values, errors, "o-", label=r"$L_2$ error for $\sin(\pi x)$")
    if errors.size >= 3:
        slope, intercept = np.polyfit(np.log(N_values[-3:]), np.log(errors[-3:]), 1)
        ref = np.exp(intercept) * N_values ** slope
        ax.loglog(N_values, ref, "--", color="0.6", label=rf"Fit $\mathcal{{O}}(N^{{{slope:.2f}}})$")
    ax.set_xlabel("Number of LGL nodes")
    ax.set_ylabel(r"$L_2$ error")
    ax.set_title(r"Legendre interpolation of $\sin(\pi x)$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exercise_j_convergence.png", dpi=200)


def extrapolation_plot(N: int, x_ext: np.ndarray) -> None:
    x_nodes = legendre_gauss_lobatto_nodes(N)
    degree = N - 1
    V_nodes = generalized_vandermonde(x_nodes, degree)
    nodal_vals = np.sin(PI * x_nodes)
    modal = np.linalg.solve(V_nodes, nodal_vals)

    V_ext = generalized_vandermonde(x_ext, degree)
    approx_ext = V_ext @ modal
    exact_ext = np.sin(PI * x_ext)

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
    fig.savefig(PLOT_DIR / "exercise_j_extrapolation.png", dpi=200)


def main() -> None:
    plot_lagrange(num_nodes=6)
    N_values = np.arange(4, 20, 2)
    errors = convergence_test(N_values)
    plot_convergence(N_values, errors)
    x_ext = np.linspace(-1.5, 1.5, 400)
    extrapolation_plot(N_values[-1], x_ext)
    if errors.size >= 2:
        ratios = errors[:-1] / errors[1:]
        print("Error ratios N_k / N_{k+1}:", ratios)


if __name__ == "__main__":
    main()
    plt.show()
