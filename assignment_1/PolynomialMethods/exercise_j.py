import matplotlib.pyplot as plt
import numpy as np

from assignment_1.PolynomialMethods.exercise_h import (
    legendre_polynomials_with_derivatives,
)
from utils.plotting import save_figure, setup_assignment_plotting, style_axes


def generalized_vandermonde(x: np.ndarray, degree: int | None = None) -> np.ndarray:
    x = np.asarray(x)
    if degree is None:
        degree = x.size - 1
    values, _ = legendre_polynomials_with_derivatives(x, degree)
    return values[: degree + 1].T


def jacobi_gauss_quadrature_nodes(alpha: float, beta: float, degree: int) -> np.ndarray:
    """Jacobi-Gauss nodes 

    Lecture 2 Polynomial Methods, slides 40–41.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree == 0:
        return np.array([-(alpha - beta) / (alpha + beta + 2.0)])

    n = degree
    k = np.arange(1, n + 1, dtype=float)
    two_k_ab = 2.0 * np.arange(n + 1, dtype=float) + alpha + beta

    diag = np.zeros(n + 1)
    if n >= 1:
        diag[1:] = (
            -0.5 * (alpha * alpha - beta * beta)
            / ((two_k_ab[1:] + 2.0) * two_k_ab[1:])
        )

    off_diag = (
        2.0
        / (two_k_ab[:-1] + 2.0)
        * np.sqrt(
            k
            * (k + alpha + beta)
            * (k + alpha)
            * (k + beta)
            / ((two_k_ab[:-1] + 1.0) * (two_k_ab[:-1] + 3.0))
        )
    )

    J = np.diag(diag) + np.diag(off_diag, 1)
    J = J + J.T

    nodes = np.linalg.eigvalsh(J)
    return np.sort(nodes)


def legendre_gauss_lobatto_nodes(num_nodes: int) -> np.ndarray:
    """Legendre-Gauss-Lobatto nodes
    Lecture 2 Polynomial Methods slide 46
    """
    if num_nodes < 2:
        raise ValueError("Need at least two nodes for LGL grid")
    if num_nodes == 2:
        return np.array([-1.0, 1.0])

    degree = num_nodes - 1
    interior = jacobi_gauss_quadrature_nodes(1.0, 1.0, degree - 2)
    nodes = np.empty(num_nodes)
    nodes[0], nodes[-1] = -1.0, 1.0
    nodes[1:-1] = interior
    return nodes


def legendre_gauss_lobatto_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Quadrature weights for Legendre-Gauss-Lobatto nodes.

   Lecture 2 Polynomial Methods  slide 46
    """
    degree = x_nodes.size - 1
    if degree < 1:
        return np.array([2.0])

    values, _ = legendre_polynomials_with_derivatives(x_nodes, degree)
    Pn = values[degree]
    return 2.0 / (degree * (degree + 1) * (Pn * Pn))


def lagrange_on_grid(x_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    #Lecture 2 slide 56

    degree = x_nodes.size - 1
    V_nodes = generalized_vandermonde(x_nodes, degree)
    V_eval = generalized_vandermonde(x_eval, degree)
    identity = np.eye(degree + 1)
    return V_eval @ np.linalg.solve(V_nodes, identity)


def discrete_l2_error(f_exact: np.ndarray, f_num: np.ndarray, weights: np.ndarray) -> float:
    diff = f_num - f_exact
    return np.sqrt(np.sum(weights * diff * diff))


def main() -> None:
    setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_j")

    num_nodes = 6
    x_nodes = legendre_gauss_lobatto_nodes(num_nodes)
    x_uniform = np.linspace(-1.0, 1.0, 100)
    lagrange_vals = lagrange_on_grid(x_nodes, x_uniform)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(num_nodes):
        ax.plot(x_uniform, lagrange_vals[:, j], label=rf"$h_{{{j}}}(x)$")
    ax.plot(x_nodes, np.zeros_like(x_nodes), "ko", label="LGL nodes")
    style_axes(
        ax,
        title="Legendre-Gauss-Lobatto Lagrange polynomials (N=6)",
        xlabel="x",
        ylabel="value",
        legend={"ncol": 2},
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_j_lagrange", fig=fig, dpi=200)

    eval_points = 200
    N_values = np.arange(4, 24, 2)

    x_eval = legendre_gauss_lobatto_nodes(eval_points)
    weights_eval = legendre_gauss_lobatto_weights(x_eval)
    f_exact = np.sin(np.pi * x_eval)
    errors: list[float] = []
    for N in N_values:
        x_nodes = legendre_gauss_lobatto_nodes(N)
        degree = N - 1
        V_nodes = generalized_vandermonde(x_nodes, degree)
        nodal_vals = np.sin(np.pi * x_nodes)
        modal = np.linalg.solve(V_nodes, nodal_vals)
        V_eval = generalized_vandermonde(x_eval, degree)
        f_approx = V_eval @ modal
        errors.append(discrete_l2_error(f_exact, f_approx, weights_eval))

    errors = np.array(errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_values, errors, "o-", label=r"$L_2$ error for $\sin(\pi x)$")

    ref = errors[0] * (N_values[0] / N_values) ** 2
    ax.loglog(N_values, ref, "--", color="0.6", label=r"Reference $N^{-2}$")
    style_axes(
        ax,
        title=r"Legendre interpolation of $\sin(\pi x)$",
        xlabel="Number of LGL nodes",
        ylabel=r"$L_2$ error",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_j_convergence", fig=fig, dpi=200)

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
    ax.plot(x_ext, exact_ext, label=r"Exact $\sin(\pi x)$")
    ax.plot(x_ext, approx_ext, "--", label=f"Legendre modal degree {degree}")
    ax.axvspan(-1.0, 1.0, color="0.9", alpha=0.5, label="Interpolation domain")
    style_axes(
        ax,
        title="Legendre polynomial extrapolation",
        xlabel="x",
        ylabel="value",
        legend=True,
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_j_extrapolation", fig=fig, dpi=200)

    if errors.size >= 2:
        ratios = errors[:-1] / errors[1:]
        print("Error ratios N_k / N_{k+1}:", ratios)


if __name__ == "__main__":
    main()

