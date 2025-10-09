"""Legendre collocation utilities for Assignment 2, Exercise 1a."""

from __future__ import annotations

import numpy as np

from assignment_1.PolynomialMethods.exercise_k import diff_matrix, vandermonde
from assignment_1.PolynomialMethods.exercise_j import legendre_gauss_lobatto_nodes


def solve_legendre_collocation(epsilon: float, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return LGL nodes and modal coefficients for the collocation scheme."""
    if num_nodes < 3:
        msg = "Collocation scheme requires at least three nodes."
        raise ValueError(msg)

    xi = legendre_gauss_lobatto_nodes(num_nodes)
    D_xi = diff_matrix(xi)
    D_x = 2.0 * D_xi  # chain rule d/dx = 2 d/dξ
    D2_x = D_x @ D_x

    operator = -epsilon * D2_x - D_x
    rhs = np.ones(num_nodes)

    # Impose Dirichlet conditions at x=0 and x=1
    operator[0, :] = 0.0
    operator[0, 0] = 1.0
    rhs[0] = 0.0

    operator[-1, :] = 0.0
    operator[-1, -1] = 1.0
    rhs[-1] = 0.0

    u_nodes = np.linalg.solve(operator, rhs)
    coeffs = np.linalg.solve(vandermonde(xi, 0, 0), u_nodes)
    return xi, coeffs
