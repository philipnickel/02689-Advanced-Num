# %%
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


def a(alpha: float, beta: float, n1: Literal[-1] | Literal[0] | Literal[1], n2: int):
    if n1 == -1 and n2 == 0:
        return 0

    if n1 == 0 and n2 == 0:
        return 0

    match n1:
        case -1:
            return (2 * (n2-1 + alpha) * (n2-1 + beta)) / (
                (2 * (n2-1) + alpha + beta + 1) * (2 * (n2-1) + alpha + beta)
            )
        case 0:
            return (alpha**2 - beta**2) / (
                (2 * (n2-1) + alpha + beta + 2) * (2 * (n2-1) + alpha + beta)
            )
        case 1:
            return (2 * ((n2-1) + 1) * ((n2-1) + alpha + beta + 1)) / (
                (2 * (n2-1) + alpha + beta) * (2 * (n2-1) + alpha + beta + 1)
            )


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, n: int):
    if n == 0:
        return np.ones_like(xs)

    if n == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2) * xs)

    return (
        ((a(alpha, beta, 0, 0) + xs) * jacobi_poly(xs, alpha, beta, n - 1)
        - a(alpha, beta, -1, n - 1) * jacobi_poly(xs, alpha, beta, n - 2)
    )) / a(alpha, beta, 1, n)


# %%

for i in range(4):
    xs = np.linspace(-1, 1)

    ys = jacobi_poly(xs, 0, 0, i)

    plt.plot(xs, ys, label=f"P{i}")
plt.legend()
plt.show()
# %%
