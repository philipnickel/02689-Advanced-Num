# %%
from cProfile import label
from matplotlib import pyplot as plt
import numpy as np
from assignment_1.PolynomialMethods.exercise_h import (
    jacobi_poly,
    normalized_jacobi_poly,
)
from numpy.polynomial.legendre import leggauss


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
    xs_g, ws = leggauss(20)
    D = diff_matrix(xs_g)

    ys_g = np.sin(xs_g * np.pi)
    dys = D @ ys_g

    xs = np.linspace(-1, 1, 100)
    ys = np.sin(xs * np.pi)
    exact_dys = np.pi * np.cos(xs * np.pi)
    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, label="$sin(x\pi)$")
    plt.plot(xs_g, ys_g, ".", label="$sin(x\pi)$ nodes")
    plt.plot(xs, exact_dys, "--", label="Exact derivative")
    plt.plot(xs_g, dys, ".", label="Spectral derivative")
    plt.grid()
    plt.legend()
    plt.title("Spectral derivative")
    plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_k_1.pdf")
    # %%

    errors = []
    errors_spec = []
    Ns = np.arange(1, 50, 2)

    for N in Ns:
        xs, ws = leggauss(N)
        D = diff_matrix(xs)
        M = int_matrix(xs)

        ys = np.sin(xs * np.pi)
        dys = D @ ys
        exact_dys = np.pi * np.cos(xs * np.pi)

        # error = np.linalg.norm(dys - exact_dys, ord=np.inf)
        error_spec = (dys - exact_dys) @ M @ (dys - exact_dys)
        # errors.append(error)
        errors_spec.append(error_spec)

    plt.figure(figsize=(12, 5))
    # plt.semilogy(Ns, errors, marker="o", label="$L_\infty$")
    plt.semilogy(Ns, errors_spec, marker="o")
    plt.xlabel("N")
    plt.ylabel("Error $L_2$")
    plt.title("Convergence Test for N")
    plt.grid()

    plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_k_2.pdf")
# %%
