"""Spectral basis utilities shared across Assignment 2."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .polynomial import (
    legendre_gauss_lobatto_nodes,
    vandermonde,
    vandermonde_normalized,
    vandermonde_x,
)


def legendre_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Return Legendre spectral differentiation matrix at arbitrary nodes.

    Constructs the spectral differentiation matrix D such that D @ u
    approximates du/dx at the collocation nodes. The matrix is computed
    using Vandermonde matrices without requiring explicit quadrature.

    Parameters
    ----------
    nodes : np.ndarray
        Collocation nodes

    Returns
    -------
    np.ndarray
        Differentiation matrix of shape (N, N)

    Notes
    -----
    The differentiation matrix is constructed as D = V_x @ V^(-1), where
    V is the Vandermonde matrix and V_x contains derivatives of the basis
    polynomials. This approach works for arbitrary node distributions.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods"
    """
    V = vandermonde(nodes, 0.0, 0.0)
    Vx = vandermonde_x(nodes, 0.0, 0.0)
    identity = np.eye(nodes.size)
    return Vx @ np.linalg.solve(V, identity)


def legendre_mass_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Return Legendre spectral mass matrix using normalized basis.

    Parameters
    ----------
    nodes : np.ndarray
        Collocation nodes

    Returns
    -------
    np.ndarray
        Mass matrix of shape (N, N)
    """
    V_norm = vandermonde_normalized(nodes, 0.0, 0.0)
    return np.linalg.inv(V_norm @ V_norm.T)


def fourier_diff_matrix_cotangent(N: int) -> np.ndarray:
    """
    Construct Fourier differentiation matrix using cotangent identity.

    Computes the spectral differentiation matrix for periodic functions
    on an equispaced grid using the cotangent formula. The matrix entries
    are constructed directly without FFT operations.

    Parameters
    ----------
    N : int
        Number of grid points

    Returns
    -------
    np.ndarray
        Fourier differentiation matrix of shape (N, N)

    Notes
    -----
    The diagonal entries are set to ensure that differentiating a constant
    function yields zero, which is enforced by requiring each row sum to
    be zero. This construction is exact for the Fourier collocation method
    on periodic domains.

    References
    ----------
    Engsig-Karup, "Lecture 1: Fourier Methods"
    Kopriva (2009), "Implementing Spectral Methods for PDEs"
    """
    indices = np.arange(N)
    diff = indices[:, None] - indices[None, :]
    D = np.zeros((N, N), dtype=float)

    mask = diff != 0
    angles = np.pi * diff[mask] / N
    parity = (-1) ** (indices[:, None] + indices[None, :])

    cot_vals = np.cos(angles) / np.sin(angles)
    D[mask] = 0.5 * parity[mask] * cot_vals

    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return D


def fourier_diff_matrix_on_interval(
    N: int, a: float = -2.0, b: float = 2.0
) -> np.ndarray:
    """
    Fourier differentiation matrix rescaled to periodic interval [a, b].

    Parameters
    ----------
    N : int
        Number of grid points
    a : float, optional
        Left endpoint (default: -2.0)
    b : float, optional
        Right endpoint (default: 2.0)

    Returns
    -------
    np.ndarray
        Rescaled Fourier differentiation matrix of shape (N, N)
    """
    scale = 2 * np.pi / (b - a)
    return scale * fourier_diff_matrix_cotangent(N)


class SpectralBasis(ABC):
    """Abstract interface for nodal spectral bases."""

    def __init__(self, domain: tuple[float, float] | None = None):
        self.domain = domain

    @abstractmethod
    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return nodal points for the basis.

        Parameters
        ----------
        num_points : int
            Number of collocation points

        Returns
        -------
        np.ndarray
            Nodal points in the configured domain
        """

    @abstractmethod
    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return differentiation matrix evaluated at `nodes`.

        Parameters
        ----------
        nodes : np.ndarray
            Collocation nodes

        Returns
        -------
        np.ndarray
            Differentiation matrix of shape (N, N)
        """

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return mass (quadrature) matrix for `nodes`.

        Subclasses can override when a closed-form expression is available.
        """
        raise NotImplementedError("Basis does not define a mass matrix.")


class LegendreLobattoBasis(SpectralBasis):
    """Legendre-Gauss-Lobatto nodal polynomial basis."""

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        super().__init__(domain=domain)

    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return nodes mapped to the configured domain.

        Parameters
        ----------
        num_points : int
            Number of Legendre-Gauss-Lobatto nodes

        Returns
        -------
        np.ndarray
            LGL nodes mapped to the physical domain
        """
        xi = legendre_gauss_lobatto_nodes(num_points)
        if self.domain == (-1.0, 1.0):
            return xi
        a, b = self.domain
        return 0.5 * (b - a) * (xi + 1.0) + a

    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return derivative matrix scaled to the physical domain.

        Parameters
        ----------
        nodes : np.ndarray
            Physical domain nodes

        Returns
        -------
        np.ndarray
            Scaled differentiation matrix of shape (N, N)
        """
        xi = legendre_gauss_lobatto_nodes(nodes.size)
        D_xi = legendre_diff_matrix(xi)
        a, b = self.domain
        scale = 2.0 / (b - a)
        return scale * D_xi

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return mass matrix associated with Legendre basis.

        Parameters
        ----------
        nodes : np.ndarray
            Physical domain nodes

        Returns
        -------
        np.ndarray
            Scaled mass matrix of shape (N, N)
        """
        xi = legendre_gauss_lobatto_nodes(nodes.size)
        M = legendre_mass_matrix(xi)
        a, b = self.domain
        return 0.5 * (b - a) * M


class FourierEquispacedBasis(SpectralBasis):
    """Equispaced Fourier basis on a periodic interval."""

    def __init__(self, domain: tuple[float, float] = (0.0, 2.0 * np.pi)):
        super().__init__(domain=domain)

    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return equispaced nodes on the periodic domain.

        Parameters
        ----------
        num_points : int
            Number of equispaced nodes

        Returns
        -------
        np.ndarray
            Equispaced nodes on the periodic interval
        """
        a, b = self.domain
        return np.linspace(a, b, num_points, endpoint=False)

    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return Fourier differentiation matrix.

        Parameters
        ----------
        nodes : np.ndarray
            Fourier collocation nodes

        Returns
        -------
        np.ndarray
            Fourier differentiation matrix of shape (N, N)
        """
        a, b = self.domain
        return fourier_diff_matrix_on_interval(nodes.size, a=a, b=b)

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return diagonal mass matrix for trapezoidal quadrature.

        Parameters
        ----------
        nodes : np.ndarray
            Fourier collocation nodes

        Returns
        -------
        np.ndarray
            Diagonal mass matrix of shape (N, N)
        """
        a, b = self.domain
        return np.eye(nodes.size) * ((b - a) / nodes.size)
