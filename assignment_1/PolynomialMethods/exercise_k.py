# %%
from matplotlib import pyplot as plt
import numpy as np
from assignment_1.PolynomialMethods.exercise_h import (
    jacobi_poly,
    normalized_jacobi_poly,
)
from numpy.polynomial.legendre import leggauss

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_k")


# %%
def grad_jacobi_poly(xs: np.ndarray, alpha: float, beta: float, n: int):
    if n == 0:
        return 0
    return 0.5 * (alpha + beta + n + 1) * jacobi_poly(xs, alpha + 1, beta + 1, n - 1)


def vandermonde(xs: np.ndarray, alpha: float, beta: float):
    N = len(xs)
    V = np.zeros((N, N))

    for n in range(N):
        V[:, n] = jacobi_poly(xs, alpha, beta, n)

    return V


def vandermonde_normalized(xs: np.ndarray, alpha: float, beta: float):
    N = len(xs)
    V = np.zeros((N, N))

    for n in range(N):
        V[:, n] = normalized_jacobi_poly(xs, alpha, beta, n)

    return V


def vandermonde_x(xs: np.ndarray, alpha: float, beta: float):
    N = len(xs)
    Vx = np.zeros((N, N))

    for n in range(N):
        Vx[:, n] = grad_jacobi_poly(xs, alpha, beta, n)

    return Vx


def diff_matrix(xs):
    V = vandermonde(xs, 0, 0)
    Vx = vandermonde_x(xs, 0, 0)

    return Vx @ np.linalg.solve(V, np.identity(len(xs)))


def int_matrix(xs):
    V = vandermonde_normalized(xs, 0, 0)
    return np.linalg.inv(V @ V.T)


# %%

if __name__ == "__main__":

    xs, ws = leggauss(100)
    D = diff_matrix(xs)

    ys_g = np.sin(xs * np.pi)
    dys = D @ ys_g

    xs = np.linspace(-1, 1, 100)
    ys = np.sin(xs * np.pi)
    exact_dys = np.pi * np.cos(xs * np.pi)
    fig_diff, ax_diff = plt.subplots(figsize=(12, 5))
    ax_diff.plot(xs, ys, label=r"$\sin(x\pi)$")
    ax_diff.plot(xs, ys_g, ".", label=r"$\sin(x\pi)$ nodes")
    ax_diff.plot(xs, exact_dys, "--", label="Exact derivative")
    ax_diff.plot(xs, dys, ".", label="Spectral derivative")
    style_axes(
            ax_diff,
            title="Spectral derivative",
            xlabel="x",
            ylabel="value",
            legend=True,

        )

    save_figure("exercise_k_diff", fig=fig_diff, dpi=200)


    ys = np.sin(xs * np.pi)
    dys = D @ ys
    exact_dys = np.pi * np.cos(xs * np.pi)

    fig_diff, ax_diff = plt.subplots(figsize=(8, 5))
    ax_diff.plot(xs, ys, label=r"$u(x) = \sin(\pi x)$")
    ax_diff.plot(xs, dys, label=r"$D u$")
    ax_diff.plot(xs, exact_dys, label=r"$u'(x)$")

    style_axes(
        ax_diff,
        title="Spectral differentiation comparison",
        xlabel="x",
        ylabel="value",
        legend=True,
    )

    save_figure("exercise_k_diff_comparison", fig=fig_diff, dpi=200)

    errors = []
    Ns = np.arange(4, 30, 4)

    for N in Ns:
        xs, ws = leggauss(N)
        D = diff_matrix(xs)

        ys = np.sin(xs * np.pi)
        dys = D @ ys
        exact_dys = np.pi * np.cos(xs * np.pi)

        error = np.linalg.norm(dys - exact_dys, ord=np.inf)
        errors.append(error)

    fig_conv, ax_conv = plt.subplots(figsize=(8, 5))
    ax_conv.loglog(Ns, errors, marker="o", label="Error")
    ax_conv.loglog(
        Ns,
        errors[0] * (Ns/ Ns[0]) ** -2,
        'k--',
        label=r"$O(N^{-2})$",
        alpha=0.5
    )


    style_axes(
        ax_conv,
        title="Convergence Test for N",
        xlabel="N",
        ylabel="Error (Infinity Norm)",
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
        legend=True,
    )
    save_figure("exercise_k", fig=fig_conv, dpi=200)

    xs, ws = leggauss(30)
    M = int_matrix(xs)
    ys = np.sin(xs * np.pi)
    ys @ (M @ ys)
# %%

# %%
