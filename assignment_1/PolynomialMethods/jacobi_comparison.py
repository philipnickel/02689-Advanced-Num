"""Comparison between custom Jacobi implementations and SciPy."""

from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jacobi

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/jacobi_comparison")


def a(alpha: float, beta: float, n1: Literal[-1, 0, 1], n2: int) -> float:
    if n1 == -1 and n2 == 0:
        return 0.0

    if n1 == 0 and n2 == 0:
        return 0.0

    match n1:
        case -1:
            return (2 * (n2 + alpha) * (n2 + beta)) / (
                (2 * n2 + alpha + beta + 1) * (2 * n2 + alpha + beta)
            )
        case 0:
            return (alpha**2 - beta**2) / (
                (2 * n2 + alpha + beta + 2) * (2 * n2 + alpha + beta)
            )
        case 1:
            return (2 * (n2 + 1) * (n2 + alpha + beta + 1)) / (
                (2 * n2 + alpha + beta) * (2 * n2 + alpha + beta + 1)
            )
        case _:
            raise ValueError("n1 must be -1, 0, or 1")


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    if n == 0:
        return np.ones_like(xs)

    if n == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2) * xs)

    return (
        (
            (a(alpha, beta, 0, 0) + xs) * jacobi_poly(xs, alpha, beta, n - 1)
            - a(alpha, beta, -1, n - 1) * jacobi_poly(xs, alpha, beta, n - 2)
        )
        / a(alpha, beta, 1, n)
    )


def main() -> None:
    x = np.linspace(-1, 1, 500)
    n_max = 4

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for n in range(n_max):
        y_custom = jacobi_poly(x, 0, 0, n)
        ax1.plot(x, y_custom, label=f"P{n}", linewidth=2)

    style_axes(
        ax1,
        title="Our Implementation (Legendre)",
        xlabel="x",
        ylabel="P_n(x)",
        legend=True,
        grid={"alpha": 0.3},
    )

    for n in range(n_max):
        y_scipy = jacobi(n, 0, 0)(x)
        ax2.plot(x, y_scipy, label=f"P{n}", linewidth=2, linestyle="--")

    style_axes(
        ax2,
        title="SciPy Implementation (Legendre)",
        xlabel="x",
        ylabel="P_n(x)",
        legend=True,
        grid={"alpha": 0.3},
    )

    for n in range(n_max):
        y_custom = jacobi_poly(x, 0, 0, n)
        y_scipy = jacobi(n, 0, 0)(x)
        difference = np.abs(y_custom - y_scipy)
        ax3.semilogy(x, difference, label=rf"$|\Delta P_{{{n}}}|$")

    style_axes(
        ax3,
        title="Absolute Difference (Legendre)",
        xlabel="x",
        ylabel="|Our - SciPy|",
        legend=True,
        grid={"alpha": 0.3},
    )

    save_figure("legendre_comparison", fig=fig, dpi=200)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for n in range(n_max):
        y_custom = jacobi_poly(x, -0.5, -0.5, n)
        ax1.plot(x, y_custom, label=f"T{n}", linewidth=2)

    style_axes(
        ax1,
        title="Our Implementation (Chebyshev)",
        xlabel="x",
        ylabel="T_n(x)",
        legend=True,
        grid={"alpha": 0.3},
    )

    for n in range(n_max):
        y_scipy = jacobi(n, -0.5, -0.5)(x)
        ax2.plot(x, y_scipy, label=f"T{n}", linewidth=2, linestyle="--")

    style_axes(
        ax2,
        title="SciPy Implementation (Chebyshev)",
        xlabel="x",
        ylabel="T_n(x)",
        legend=True,
        grid={"alpha": 0.3},
    )

    for n in range(n_max):
        y_custom = jacobi_poly(x, -0.5, -0.5, n)
        y_scipy = jacobi(n, -0.5, -0.5)(x)
        difference = np.abs(y_custom - y_scipy)
        ax3.semilogy(x, difference, label=rf"$|\Delta T_{{{n}}}|$")

    style_axes(
        ax3,
        title="Absolute Difference (Chebyshev)",
        xlabel="x",
        ylabel="|Our - SciPy|",
        legend=True,
        grid={"alpha": 0.3},
    )

    save_figure("chebyshev_comparison", fig=fig, dpi=200)

    print("Maximum absolute errors:")
    print("=" * 40)

    print("\nLegendre Polynomials (α=0, β=0):")
    for n in range(n_max):
        y_custom = jacobi_poly(x, 0, 0, n)
        y_scipy = jacobi(n, 0, 0)(x)
        max_error = np.max(np.abs(y_custom - y_scipy))
        print(f"  P_{n}: {max_error:.2e}")

    print("\nChebyshev Polynomials (α=-0.5, β=-0.5):")
    for n in range(n_max):
        y_custom = jacobi_poly(x, -0.5, -0.5, n)
        y_scipy = jacobi(n, -0.5, -0.5)(x)
        max_error = np.max(np.abs(y_custom - y_scipy))
        print(f"  T_{n}: {max_error:.2e}")


if __name__ == "__main__":
    main()
