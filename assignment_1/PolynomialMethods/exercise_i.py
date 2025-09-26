from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi

from utils.plotting import save_figure, setup_assignment_plotting, style_axes


setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_i")


def u(x: np.ndarray) -> np.ndarray:
    return 1.0 / (2.0 - np.cos(np.pi * x))


def legendre_coeffs(num_quad: int, num_modes: int = 200) -> np.ndarray:
    nodes, weights = leggauss(num_quad)
    coeffs = np.zeros(num_modes)
    values = u(nodes)
    for n in range(num_modes):
        Pn = eval_jacobi(n, 0, 0, nodes)
        integral_approx = np.sum(weights * values * Pn)
        coeffs[n] = (2 * n + 1) / 2 * integral_approx
    return coeffs


def synthesize(x_vals: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x_vals)
    for n, coef in enumerate(coeffs):
        result += coef * eval_jacobi(n, 0, 0, x_vals)
    return result


def plot_coeff_decay(coeffs: np.ndarray, num_quad: int) -> None:
    degrees = np.arange(coeffs.size)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(degrees, np.abs(coeffs), marker="o", linestyle="-", label=fr"$N={num_quad}$")
    style_axes(
        ax,
        title=r"Legendre coefficients of $u(x)=1/(2-\cos(\pi x))$",
        xlabel="Polynomial degree n",
        ylabel=r"$|c_n|$",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_i_coeff_decay", fig=fig, dpi=200)


def plot_synthesized_function(coeffs: np.ndarray) -> None:
    xs = np.linspace(-1.0, 1.0, 500)
    synthesized = synthesize(xs, coeffs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, synthesized, label="Modal synthesis")
    ax.plot(xs, u(xs), linestyle="--", label=r"Exact $u(x)$")
    style_axes(
        ax,
        title="Legendre series reconstruction",
        xlabel="x",
        ylabel="value",
        legend=True,
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_i_synthesis", fig=fig, dpi=200)


def main() -> None:
    num_quad = 200
    coeffs = legendre_coeffs(num_quad)
    plot_coeff_decay(coeffs, num_quad)
    plot_synthesized_function(coeffs)


if __name__ == "__main__":
    main()
