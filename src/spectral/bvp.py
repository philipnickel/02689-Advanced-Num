"""Boundary value problem solvers used in Assignment 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss

from .polynomial import legendre_gauss_lobatto_nodes, vandermonde
from .spectral import (
    FourierEquispacedBasis,
    LegendreLobattoBasis,
    legendre_diff_matrix,
)


@dataclass
class BvpProblem:
    """Dense linear BVP system with convenience helpers."""

    operator: np.ndarray
    rhs: np.ndarray

    def apply_dirichlet(self, indices: Iterable[int], values: Iterable[float]) -> None:
        """
        Apply Dirichlet boundary conditions in-place.

        Parameters
        ----------
        indices : Iterable[int]
            Row indices where boundary conditions are applied
        values : Iterable[float]
            Boundary condition values
        """
        for idx, value in zip(indices, values):
            self.operator[idx, :] = 0.0
            self.operator[idx, idx] = 1.0
            self.rhs[idx] = value

    def solve(self) -> np.ndarray:
        """
        Solve the linear BVP system.

        Returns
        -------
        np.ndarray
            Solution vector
        """
        return np.linalg.solve(self.operator, self.rhs)


# =============================================================================
# Legendre Tau Method
# =============================================================================


def legendre_tau_derivative_matrices(num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return first and second derivative matrices in Legendre modal space.

    Parameters
    ----------
    num_modes : int
        Number of Legendre modes

    Returns
    -------
    D1 : np.ndarray
        First derivative matrix of shape (num_modes, num_modes)
    D2 : np.ndarray
        Second derivative matrix of shape (num_modes, num_modes)
    """
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


def legendre_tau_problem(epsilon: float, num_modes: int) -> BvpProblem:
    """
    Assemble Legendre tau method system as a BVP problem.

    The tau method is a modal spectral method that approximates the solution
    as a truncated series of Legendre polynomials. Boundary conditions are
    enforced by replacing the last rows of the discrete operator.

    Parameters
    ----------
    epsilon : float
        Diffusion parameter
    num_modes : int
        Number of Legendre modes

    Returns
    -------
    BvpProblem
        Assembled BVP system with boundary conditions
    """
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

    return BvpProblem(operator=system, rhs=rhs_mod)


def solve_legendre_tau(epsilon: float, num_modes: int) -> np.ndarray:
    """
    Solve the Legendre tau formulation.

    Parameters
    ----------
    epsilon : float
        Diffusion parameter
    num_modes : int
        Number of Legendre modes

    Returns
    -------
    np.ndarray
        Modal coefficients

    Notes
    -----
    The tau method does not require the residual to vanish pointwise,
    but instead enforces the differential equation in a weighted sense
    with boundary conditions imposed through row replacement.

    References
    ----------
    Engsig-Karup, "Lecture 5: Boundary Value Problems", p. 14
    Kopriva (2009), "Implementing Spectral Methods for PDEs", p. 107
    """
    problem = legendre_tau_problem(epsilon, num_modes)
    return problem.solve()


# =============================================================================
# Legendre-Gauss-Lobatto Collocation
# =============================================================================


def solve_legendre_collocation(
    epsilon: float, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve boundary value problem using Legendre-Gauss-Lobatto collocation.

    The collocation method enforces the differential equation exactly at
    the collocation nodes (Legendre-Gauss-Lobatto points). This nodal
    approach naturally incorporates boundary conditions at the endpoints.

    Parameters
    ----------
    epsilon : float
        Diffusion parameter
    num_nodes : int
        Number of collocation nodes

    Returns
    -------
    xi : np.ndarray
        Collocation nodes
    coeffs : np.ndarray
        Modal coefficients

    Notes
    -----
    Legendre-Gauss-Lobatto points include the domain endpoints, making
    them natural for imposing Dirichlet boundary conditions. The spectral
    differentiation matrix is constructed directly at these nodes.

    References
    ----------
    Engsig-Karup, "Lecture 5: Boundary Value Problems", p. 39
    """
    if num_nodes < 3:
        msg = "Collocation scheme requires at least three nodes."
        raise ValueError(msg)

    xi = legendre_gauss_lobatto_nodes(num_nodes)
    basis = LegendreLobattoBasis(domain=(0.0, 1.0))
    D_x = basis.diff_matrix(xi)
    D2_x = D_x @ D_x

    operator = -epsilon * D2_x - D_x
    rhs = np.ones(num_nodes)

    problem = BvpProblem(operator=operator, rhs=rhs)
    problem.apply_dirichlet(indices=(0, num_nodes - 1), values=(0.0, 0.0))

    u_nodes = problem.solve()
    coeffs = np.linalg.solve(vandermonde(xi, 0, 0), u_nodes)
    return xi, coeffs


# =============================================================================
# Polar coordinate Laplace problem
# =============================================================================


def solve_polar_bvp(
    r1: float, r2: float, Nr: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Laplace's equation in polar coordinates using spectral collocation.

    Parameters
    ----------
    r1 : float
        Inner radius
    r2 : float
        Outer radius
    Nr : int
        Number of radial collocation points

    Returns
    -------
    Phi : np.ndarray
        Analytical solution on the grid
    Phi_hat : np.ndarray
        Numerical solution on the grid
    Rs : np.ndarray
        Radial meshgrid
    Theta : np.ndarray
        Angular meshgrid
    """
    xs, ws = leggauss(Nr)
    rs = 0.5 * (r2 - r1) * (xs + 1) + r1

    theta_basis = FourierEquispacedBasis(domain=(0.0, 2.0 * np.pi))
    Ntheta = Nr
    thetas = theta_basis.nodes(Ntheta)

    Rs, Theta = np.meshgrid(rs, thetas)
    Phi = (Rs + (r1**2 / Rs)) * np.cos(Theta)

    Dtheta = theta_basis.diff_matrix(thetas)
    Dr = (2 / (r2 - r1)) * legendre_diff_matrix(xs)
    Dtheta2 = Dtheta @ Dtheta

    Lr_block = Dr @ Dr + np.diag(1 / rs) @ Dr
    Ltheta_block = Dtheta2

    Lr = np.kron(Lr_block, np.eye(Ntheta))
    Ltheta = np.kron(np.diag(1 / rs**2), Ltheta_block)

    L = Ltheta + Lr

    b = np.zeros_like(Phi)
    b[:, 0] = 1
    b[:, -1] = 1
    b_flat = b.flatten(order="F")
    indices = np.where(b_flat == 1)[0]
    L[indices, :] = 0
    L[indices, indices] = 1
    b_flat[indices] = Phi.flatten(order="F")[indices]

    Phi_hat = np.linalg.solve(L, b_flat).reshape(Phi.shape, order="F")

    return Phi, Phi_hat, Rs, Theta


def solve_transport_spacetime(
    Nx: int,
    Nt: int,
    x1: float,
    x2: float,
    t1: float,
    t2: float,
    a: float,
    exact_solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve transport equation using space-time Legendre collocation.

    Solves ∂u/∂t + a ∂u/∂x = 0 with boundary conditions:
    - Initial condition at t=t1
    - Inflow condition at x=x1 (a > 0)

    Parameters
    ----------
    Nx : int
        Number of spatial collocation points
    Nt : int
        Number of temporal collocation points
    x1, x2 : float
        Spatial domain [x1, x2]
    t1, t2 : float
        Temporal domain [t1, t2]
    a : float
        Wave speed
    exact_solution : np.ndarray
        Exact solution on the grid (for boundary conditions), shape (Nx, Nt)

    Returns
    -------
    Phi : np.ndarray
        Exact solution on the grid
    Phi_hat : np.ndarray
        Numerical solution on the grid
    Ts : np.ndarray
        Temporal meshgrid
    Xs : np.ndarray
        Spatial meshgrid
    """
    # Get Legendre-Gauss-Lobatto nodes
    ys_x = legendre_gauss_lobatto_nodes(Nx)
    ys_t = legendre_gauss_lobatto_nodes(Nt)

    # Map to physical domains
    xs = 0.5 * (x2 - x1) * (ys_x + 1) + x1
    ts = 0.5 * (t2 - t1) * (ys_t + 1) + t1

    # Create meshgrid
    Ts, Xs = np.meshgrid(ts, xs)

    # Build spectral differentiation matrices
    Dx = (2 / (x2 - x1)) * legendre_diff_matrix(ys_x)
    Dt = (2 / (t2 - t1)) * legendre_diff_matrix(ys_t)

    # Construct space-time operator: ∂u/∂t + a ∂u/∂x = 0
    Lt_block = Dt
    Lx_block = (a * np.eye(Dx.shape[0])) @ Dx

    Lt = np.kron(Lt_block, np.eye(Nx))
    Lx = np.kron(np.eye(Nt), Lx_block)
    L = Lx + Lt

    # Build boundary condition mask and RHS
    b = np.zeros(Nx * Nt)
    b[:Nx] = 1  # Initial condition at t=t1
    b[::Nx] = 1  # Left boundary at x=x1

    # Apply boundary conditions
    indices = np.where(b == 1)[0]
    L[indices, :] = 0
    L[indices, indices] = 1
    b[indices] = exact_solution.flatten(order="F")[indices]

    # Solve
    Phi_hat = np.linalg.solve(L, b).reshape(exact_solution.shape, order="F")

    return exact_solution, Phi_hat, Ts, Xs


# Backwards-compatible alias retained for assignment scripts
solve_bvp = solve_polar_bvp
