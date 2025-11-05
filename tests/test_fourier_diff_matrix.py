import numpy as np
from numpy.testing import assert_allclose

from spectral import (
    FourierEquispacedBasis,
    fourier_diff_matrix_cotangent,
    fourier_diff_matrix_complex,
    fourier_diff_matrix_on_interval,
)


def _sin_mode(x: np.ndarray, mode: int, interval: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic sine mode and exact derivative on [a, b)."""
    a, b = interval
    length = b - a
    phase = 2.0 * np.pi * mode * (x - a) / length
    u = np.sin(phase)
    dudx = (2.0 * np.pi * mode / length) * np.cos(phase)
    return u, dudx


def test_real_and_complex_agree_on_unit_circle() -> None:
    N = 32
    interval = (0.0, 2.0 * np.pi)
    nodes = np.linspace(*interval, N, endpoint=False)
    u, exact = _sin_mode(nodes, mode=3, interval=interval)

    D_real = fourier_diff_matrix_on_interval(N, *interval, representation="real")
    D_complex = fourier_diff_matrix_on_interval(N, *interval, representation="complex")

    ux_real = D_real @ u
    ux_complex = D_complex @ u

    assert_allclose(ux_real, exact, atol=1e-12, rtol=1e-12)
    assert_allclose(ux_complex.real, exact, atol=1e-12, rtol=1e-12)
    assert_allclose(ux_complex.imag, 0.0, atol=1e-12, rtol=1e-12)

    # Matrices should agree up to numerical precision
    assert_allclose(D_complex.real, D_real, atol=1e-12, rtol=1e-12)


def test_rescaled_interval_matches_expectations() -> None:
    N = 40
    interval = (-1.5, 3.0)
    nodes = np.linspace(*interval, N, endpoint=False)
    u, exact = _sin_mode(nodes, mode=5, interval=interval)

    D_real = fourier_diff_matrix_on_interval(N, *interval, representation="real")
    D_complex = fourier_diff_matrix_on_interval(N, *interval, representation="complex")

    assert_allclose(D_real @ u, exact, atol=1e-11, rtol=1e-11)
    assert_allclose((D_complex @ u).real, exact, atol=1e-11, rtol=1e-11)
    assert_allclose((D_complex @ u).imag, 0.0, atol=1e-11, rtol=1e-11)


def test_basis_parameter_selects_representation() -> None:
    N = 16
    interval = (0.0, 2.0 * np.pi)

    basis_real = FourierEquispacedBasis(domain=interval, representation="real")
    basis_complex = FourierEquispacedBasis(domain=interval, representation="complex")

    nodes = basis_real.nodes(N)
    assert_allclose(nodes, basis_complex.nodes(N))

    D_real_basis = basis_real.diff_matrix(nodes)
    D_complex_basis = basis_complex.diff_matrix(nodes)

    assert_allclose(D_real_basis, fourier_diff_matrix_cotangent(N), atol=1e-12, rtol=1e-12)
    assert_allclose(
        D_complex_basis,
        fourier_diff_matrix_complex(N),
        atol=1e-12,
        rtol=1e-12,
    )


if __name__ == "__main__":
    # Run tests manually when invoked as a script
    test_real_and_complex_agree_on_unit_circle()
    test_rescaled_interval_matches_expectations()
    test_basis_parameter_selects_representation()
    print("All Fourier differentiation matrix tests passed.")
