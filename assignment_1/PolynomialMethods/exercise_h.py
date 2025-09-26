# %%
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from scipy.special import gamma


def a(alpha: float, beta: float, n1: Literal[-1] | Literal[0] | Literal[1], n2: int):
    if n1 == -1 and n2 == 0:
        return 0

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
                (2 * n2 + alpha + beta + 2) * (2 * n2 + alpha + beta + 1)
            )


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, N: int):
    jpm2 = xs**0
    jpm1 = 0.5 * (alpha - beta + (alpha + beta + 2) * xs)
    jpm0 = xs * 0

    if N == 0:
        return jpm2
    if N == 1:
        return jpm1

    for n in range(2, N + 1):
        print(n)
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


# %%
if __name__ == "__main__":
    plt.figure(figsize=(12, 5))
    for n in range(6):
        xs = np.linspace(-1, 1)

        ys = jacobi_poly(xs, 0, 0, n)

        plt.plot(xs, ys, label=f"$P_{{{n}}}$")
    plt.title("Legendre polynomials")
    plt.xlabel("x")
    plt.legend()
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
