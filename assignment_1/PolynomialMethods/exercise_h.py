# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, factorial, gamma

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_h")


def legendre_polynomials_with_derivatives(xs: np.ndarray, degree: int):
    """Return Legendre polynomial values and derivatives up to 'degree'.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")

    xs = np.asarray(xs)
    num_points = xs.size

    values = np.zeros((degree + 1, num_points))
    derivatives = np.zeros_like(values)

    values[0] = 1.0
    if degree == 0:
        return values, derivatives

    values[1] = xs
    derivatives[1] = 1.0

    for n in range(2, degree + 1):
        two_n_minus_1 = 2 * n - 1
        inv_n = 1.0 / n
        values[n] = inv_n * (
            two_n_minus_1 * xs * values[n - 1] - (n - 1) * values[n - 2]
        )
        derivatives[n] = inv_n * (
            two_n_minus_1 * (values[n - 1] + xs * derivatives[n - 1])
            - (n - 1) * derivatives[n - 2]
        )

    return values, derivatives


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, N: int):
    jpm2 = xs**0
    jpm1 = 0.5 * (alpha - beta + (alpha + beta + 2) * xs)
    jpm0 = xs * 0

    if N == 0:
        return jpm2
    if N == 1:
        return jpm1

    for n in range(2, N + 1):
        am1 = (2 * ((n - 1) + alpha) * ((n - 1) + beta)) / (
            (2 * (n - 1) + alpha + beta + 1) * (2 * (n - 1) + alpha + beta)
        )
        a0 = (alpha**2 - beta**2) / (
            (2 * (n - 1) + alpha + beta + 2) * (2 * (n - 1) + alpha + beta)
        )
        ap1 = (2 * ((n - 1) + 1) * ((n - 1) + alpha + beta + 1)) / (
            (2 * (n - 1) + alpha + beta + 2) * (2 * (n - 1) + alpha + beta + 1)
        )

        jpm0 = ((a0 + xs) * jpm1 - am1 * jpm2) / ap1
        jpm2 = jpm1
        jpm1 = jpm0

    return jpm0


def normalized_jacobi_poly(xs: np.ndarray, alpha: float, beta: float, N: int):
    log_c = -0.5 * (
        np.log(2) * (alpha + beta + 1)
        + gammaln(N + alpha + 1)
        + gammaln(N + beta + 1)
        - gammaln(N + 1)
        - np.log(2 * N + alpha + beta + 1)
        - gammaln(N + alpha + beta + 1)
    )
    return np.exp(log_c) * jacobi_poly(xs, alpha, beta, N)


# %%
if __name__ == "__main__":
    xs = np.linspace(-1, 1)

    fig_legendre, ax_legendre = plt.subplots(figsize=(12, 5))
    for n in range(6):
        ys = jacobi_poly(xs, 0, 0, n)
        ax_legendre.plot(xs, ys, label=f"$P_{{{n}}}$")
    style_axes(
        ax_legendre,
        title="Legendre polynomials",
        xlabel="x",
        ylabel="value",
        legend=True,
    )
    save_figure("exercise_h_1", fig=fig_legendre, dpi=200)

    fig_chebyshev, ax_chebyshev = plt.subplots(figsize=(12, 5))
    for n in range(6):
        ws = gamma(n + 1) * gamma(1 / 2) / gamma(n + 1 / 2)
        ys = jacobi_poly(xs, -1 / 2, -1 / 2, n) * ws
        ax_chebyshev.plot(xs, ys, label=f"$P_{{{n}}}$")
    style_axes(
        ax_chebyshev,
        title="Chebyshev polynomials",
        xlabel="x",
        ylabel="value",
        legend=True,
    )
    save_figure("exercise_h_2", fig=fig_chebyshev, dpi=200)

    # %%
