# %%
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
    xs, ws = leggauss(100)
    D = diff_matrix(xs)

    ys = np.sin(xs * np.pi)
    dys = D @ ys
    exact_dys = np.pi * np.cos(xs * np.pi)

    plt.plot(xs, ys)
    plt.plot(xs, dys)
    plt.plot(xs, exact_dys)
    plt.grid()
    # %%

    errors = []
    Ns = 2 ** np.arange(0, 10)

    for N in Ns:
        xs, ws = leggauss(N)
        D = diff_matrix(xs)

        ys = np.sin(xs * np.pi)
        dys = D @ ys
        exact_dys = np.pi * np.cos(xs * np.pi)

        error = np.linalg.norm(dys - exact_dys, ord=np.inf)
        errors.append(error)

    plt.loglog(Ns, errors, marker="o")
    plt.xlabel("N")
    plt.ylabel("Error (Infinity Norm)")
    plt.title("Convergence Test for N")
    plt.grid()
    plt.show()

    plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_k.pdf")

    # %%
    xs, ws = leggauss(30)
    M = int_matrix(xs)
    ys = np.sin(xs * np.pi)
    ys @ (M @ ys)
# %%

# %%
