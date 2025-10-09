"""Legendre Tau method utilities for Assignment 2, Exercise 1a."""

from __future__ import annotations

import numpy as np


def legendre_tau_derivative_matrices(num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse derivative matrices based on assignment remark relations."""
    n = np.arange(num_modes, dtype=float)[:, None]
    p = np.arange(num_modes, dtype=float)[None, :]

    mask_d1 = (p > n) & ((p + n) % 2 == 1)
    D1 = np.where(mask_d1, 2.0 * n + 1.0, 0.0)

    mask_d2 = (p >= n + 2) & ((p + n) % 2 == 0)
    factor2 = n + 0.5
    n_term = n * (n + 1.0)
    D2 = np.where(
        mask_d2,
        factor2 * (p * (p + 1.0) - n_term),
        0.0,
    )
    return D1, D2

def legendre_tau_system(epsilon: float, num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    D1, D2 = legendre_tau_derivative_matrices(num_modes)
    operator = -4.0 * epsilon * D2 - 2.0 * D1

    rhs = np.zeros(num_modes)
    rhs[0] = 1.0  # coefficient for constant one

    system = np.zeros((num_modes, num_modes))
    system[:-2, :] = operator[:-2, :]
    rhs_mod = rhs.copy()

    system[-2, :] = (-1.0) ** np.arange(num_modes)
    system[-1, :] = 1.0
    rhs_mod[-2:] = 0

    return system, rhs_mod


def solve_legendre_tau(epsilon: float, num_modes: int) -> np.ndarray:
    """Solve the BVP using the Legendre Tau method."""
    system, rhs = legendre_tau_system(epsilon, num_modes)
    coeffs = np.linalg.solve(system, rhs)
    return coeffs

## % visualize sparse matrix 

def solve_legendre_collocation(epsilon: float, num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve the BVP using the Legendre Tau method."""
    system, rhs = legendre_tau_system(epsilon, num_modes)
    coeffs = np.linalg.solve(system, rhs)
    return system, coeffs


if __name__ == "__main__":

# %% Imports 

    import sympy as sp
    import matplotlib.pyplot as plt
    plt.rcParams.setdefault("text.usetex", False)

    from utils import plot_style  # noqa: F401
    plt.rcParams["text.usetex"] = False

# %% Visualize system matrix
    num_modes = 25  # example value
    epsilon = 0.1
    sys_matrix, rhs = legendre_tau_system(1e-2, num_modes)

    plt.figure(figsize=(10, 5))

    plt.title("Sparcity System Matrix")
    plt.spy(sys_matrix, markersize=5)

    plt.figure(figsize=(10, 5))

    plt.matshow(sys_matrix)
    plt.title("Sparcity System Matrix")

    plt.colorbar()



