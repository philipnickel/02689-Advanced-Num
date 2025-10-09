# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, factorial


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


def legendre_polynomials(xs: np.ndarray, degree: int) -> np.ndarray:
    """Return Legendre polynomials P_0..P_degree evaluated at xs."""
    xs = np.asarray(xs)
    polys = np.empty((degree + 1, xs.size))
    for n in range(degree + 1):
        polys[n] = jacobi_poly(xs, 0.0, 0.0, n)
    return polys


# %%
if __name__ == "__main__":
    plt.figure(figsize=(12, 5))
    for n in range(6):
        xs = np.linspace(-1, 1)

        ys = jacobi_poly(xs, 0, 0, n)

        plt.plot(xs, ys, label=f"$P_{{{n}}}$")

        # ys_scipy = eval_jacobi(n, 0, 0, xs)
        # plt.plot(xs, ys_scipy, "--", label=f"scipy $P_{{{n}}}$")
    plt.title("Legendre polynomials")
    plt.xlabel("x")
    plt.legend()
    plt.grid()
    plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_h_1.pdf")
    # %%

    plt.figure(figsize=(12, 5))
    for n in range(6):
        ws = gamma(n + 1) * gamma(1 / 2) / gamma(n + 1 / 2)

        xs = np.linspace(-1, 1)

        ys = jacobi_poly(xs, -1 / 2, -1 / 2, n) * ws

        plt.plot(xs, ys, label=f"$P_{{{n}}}$")
    plt.title("Chebyshev polynomials")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_h_2.pdf")

    # %%
