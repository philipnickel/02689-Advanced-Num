from __future__ import annotations
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_h")


def jacobi_polynomials(xs: np.ndarray, alpha: float, beta: float, degree: int) -> np.ndarray:
    """Return values of Jacobi polynomials P_0..P_degree evaluated at xs."""

    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = np.asarray(xs, dtype=float)
    vals = np.zeros((degree + 1, x.size))
    vals[0, :] = 1.0
    if degree == 0:
        return vals

    vals[1, :] = 0.5 * ((2 + alpha + beta) * x + alpha - beta)
    for n in range(1, degree):
        two_n_ab = 2 * n + alpha + beta
        denom = 2 * (n + 1) * (n + alpha + beta + 1)
        A = (two_n_ab + 1) * ((two_n_ab + 2) * x + alpha - beta)
        B = 2 * (n + alpha) * (n + beta) * (two_n_ab + 2)
        vals[n + 1, :] = (A * vals[n, :] - B * vals[n - 1, :]) / denom
    return vals


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """Return Jacobi polynomial P_n^{(alpha, beta)} evaluated at xs."""

    return jacobi_polynomials(xs, alpha, beta, n)[n]


def legendre_polynomials_with_derivatives(
    xs: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre values and first derivatives up to ``degree``."""

    if degree < 0:
        raise ValueError("degree must be non-negative")

    x = np.asarray(xs, dtype=float)
    vals = np.zeros((degree + 1, x.size))
    derivs = np.zeros_like(vals)
    vals[0, :] = 1.0
    if degree == 0:
        return vals, derivs

    vals[1, :] = x
    derivs[1, :] = 1.0
    for n in range(1, degree):
        coeff = 2 * n + 1
        denom = n + 1
        vals[n + 1, :] = (coeff * x * vals[n, :] - n * vals[n - 1, :]) / denom
        derivs[n + 1, :] = (
            coeff * (vals[n, :] + x * derivs[n, :]) - n * derivs[n - 1, :]
        ) / denom
    return vals, derivs


def legendre_polynomials(xs: np.ndarray, degree: int) -> np.ndarray:
    """Convenience wrapper returning Legendre values only."""

    return legendre_polynomials_with_derivatives(xs, degree)[0]


def chebyshev_first_kind(xs: np.ndarray, degree: int) -> np.ndarray:
    """Return Chebyshev T_n polynomials via scaled Jacobi polynomials."""

    polys = jacobi_polynomials(xs, -0.5, -0.5, degree)
    for n in range(degree + 1):
        weight = gamma(n + 1) * gamma(0.5) / gamma(n + 0.5)
        polys[n, :] *= weight
    return polys


def plot_legendre(n_max: int = 5) -> Path:
    xs = np.linspace(-1.0, 1.0, 400)
    polys = legendre_polynomials(xs, n_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    for n in range(n_max + 1):
        ax.plot(xs, polys[n], label=rf"$P_{{{n}}}$")
    style_axes(ax, title="Legendre polynomials", xlabel="x", legend=True)
    return save_figure("exercise_h_legendre", fig=fig, dpi=200)


def plot_chebyshev(n_max: int = 5) -> Path:
    xs = np.linspace(-1.0, 1.0, 400)
    polys = chebyshev_first_kind(xs, n_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    for n in range(n_max + 1):
        ax.plot(xs, polys[n], label=rf"$P_{{{n}}}$")
    style_axes(ax, title="Chebyshev polynomials", xlabel="x", legend=True)
    return save_figure("exercise_h_chebyshev", fig=fig, dpi=200)


def main(n_max: int = 5) -> Iterable[Path]:
    return [plot_legendre(n_max), plot_chebyshev(n_max)]


if __name__ == "__main__":
    main()
