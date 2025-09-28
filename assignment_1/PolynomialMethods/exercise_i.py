from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi

from utils.plotting import save_figure, setup_assignment_plotting, style_axes


setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_i")

QUADRATURE_SIZES = [10, 20, 40, 80, 100, 200]


def u(x: np.ndarray) -> np.ndarray:
    return 1.0 / (2.0 - np.cos(np.pi * x))


def legendre_coeffs(num_quad: int, num_modes: int = 200) -> np.ndarray:
    nodes, weights = leggauss(num_quad)
    coeffs = np.zeros(num_modes)
    values = u(nodes)
    for n in range(num_modes):
        Pn = eval_jacobi(n, 0, 0, nodes)
        integral_approx = np.sum(weights * values * Pn)
        coeffs[n] = (2 * n + 1) / 2 * integral_approx
    return coeffs


def reconstruct(x_vals: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x_vals)
    for n, coef in enumerate(coeffs):
        result += coef * eval_jacobi(n, 0, 0, x_vals)
    return result


def _draw_coeff_decay(
    ax: plt.Axes,
    coeffs: np.ndarray,
    num_quad: int,
    legend: bool,
    title: str | None = None,
) -> None:
    degrees = np.arange(coeffs.size)
    ax.semilogy(
        degrees,
        np.abs(coeffs),
        marker="o",
        markersize=1.5,
        linestyle="",
        label=fr"$N={num_quad}$",
    )
    style_axes(
        ax,
        title=title or rf"Legendre coefficients of $u(x)$ (N={num_quad})",
        xlabel="Polynomial degree n",
        ylabel=r"$|c_n|$",
        legend=legend,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )


def plot_coeff_decay(coeffs: np.ndarray, num_quad: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_coeff_decay(ax, coeffs, num_quad, legend=True)
    save_figure(f"exercise_i_coeff_decay_N{num_quad}", fig=fig, dpi=200)
    plt.close(fig)


def _draw_reconstruction(
    ax: plt.Axes,
    coeffs: np.ndarray,
    num_quad: int,
    legend: bool,
    title: str | None = None,
) -> None:
    xs = np.linspace(-1.0, 1.0, 500)
    reconstruction= reconstruct(xs, coeffs)

    ax.plot(xs, reconstruction, label="Reconstruction")
    ax.plot(xs, u(xs), linestyle="--", label=r"Exact $u(x)$")
    style_axes(
        ax,
        title=title or rf"Legendre series reconstruction (N={num_quad})",
        xlabel="x",
        ylabel="value",
        legend=legend,
        grid={"linestyle": ":", "linewidth": 0.5},
    )


def plot_reconstructed_function(coeffs: np.ndarray, num_quad: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_reconstruction(ax, coeffs, num_quad, legend=True)
    save_figure(f"exercise_i_reconstruction_N{num_quad}", fig=fig, dpi=200)
    plt.close(fig)


def plot_summary(coeffs: np.ndarray, num_quad: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _draw_coeff_decay(
        axes[0], coeffs, num_quad, legend=False, title=rf"Coeff. decay (N={num_quad})"
    )
    _draw_reconstruction(
        axes[1], coeffs, num_quad, legend=True, title=rf"Reconstruction (N={num_quad})"
    )

    fig.tight_layout()
    save_figure(f"exercise_i_summary_N{num_quad}", fig=fig, dpi=200)
    plt.close(fig)


def compute_reconstruction_errors(coeffs: np.ndarray, x_vals: np.ndarray) -> tuple[float, float]:
    """Compute L2 and L∞ errors between reconstruction and exact function."""
    reconstruction = reconstruct(x_vals, coeffs)
    exact = u(x_vals)
    diff = reconstruction - exact

    l2_error = np.sqrt(np.trapezoid(diff**2, x_vals))
    linf_error = np.max(np.abs(diff))

    return l2_error, linf_error


def compute_convergence_rate(quad_sizes: list[int], errors: list[float]) -> tuple[np.ndarray, float]:
    """Compute convergence rate using linear regression in log space."""
    log_h = np.log(1.0 / np.array(quad_sizes))  # h = 1/N for spectral methods
    log_errors = np.log(np.array(errors))

    # Linear regression: log(error) = slope * log(h) + intercept
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    slope, intercept = np.linalg.lstsq(A, log_errors, rcond=None)[0]

    # Convergence rate is -slope (since error ~ h^p where p is convergence rate)
    convergence_rate = -slope

    # Synthesized errors for reference line
    fitted_errors = np.exp(intercept) * (1.0 / np.array(quad_sizes))**(-convergence_rate)

    return fitted_errors, convergence_rate


def analyze_convergence() -> dict:
    """Analyze convergence of Legendre series reconstruction."""
    x_test = np.linspace(-1, 1, 1000)  # High-resolution test grid
    l2_errors = []
    linf_errors = []

    print("Computing convergence analysis...")
    print("N\t\tL2 Error\t\tLinf Error")
    print("-" * 45)

    for num_quad in QUADRATURE_SIZES:
        coeffs = legendre_coeffs(num_quad)
        l2_error, linf_error = compute_reconstruction_errors(coeffs, x_test)
        l2_errors.append(l2_error)
        linf_errors.append(linf_error)
        print(f"{num_quad}\t\t{l2_error:.2e}\t\t{linf_error:.2e}")

    # Compute convergence rates
    fitted_l2_errors, l2_conv_rate = compute_convergence_rate(QUADRATURE_SIZES, l2_errors)
    fitted_linf_errors, linf_conv_rate = compute_convergence_rate(QUADRATURE_SIZES, linf_errors)

    print(f"\nEstimated L2 convergence rate: {l2_conv_rate:.2f}")
    print(f"Estimated L∞ convergence rate: {linf_conv_rate:.2f}")

    return {
        'quad_sizes': QUADRATURE_SIZES,
        'l2_errors': l2_errors,
        'linf_errors': linf_errors,
        'fitted_l2_errors': fitted_l2_errors,
        'fitted_linf_errors': fitted_linf_errors,
        'l2_convergence_rate': l2_conv_rate,
        'linf_convergence_rate': linf_conv_rate
    }


def plot_convergence_analysis(conv_data: dict) -> None:
    """Plot convergence analysis with both L2 and L∞ errors in same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    quad_sizes = conv_data['quad_sizes']
    l2_errors = conv_data['l2_errors']
    linf_errors = conv_data['linf_errors']

    # Plot both error types
    ax.loglog(quad_sizes, l2_errors, 'o-', linewidth=1, markersize=2,
              label='L2 reconstruction error')
    ax.loglog(quad_sizes, linf_errors, 's-', linewidth=1, markersize=2,
              label='$L_\\infty$ reconstruction error')

    # Add N^-2 reference line
    N_ref = np.array(quad_sizes)
    n2_ref = l2_errors[0] * (N_ref[0]/N_ref)**2  # O(N^-2) reference
    ax.loglog(N_ref, n2_ref, 'k--', alpha=0.6,
              label='$O(N^{-2})$ ')

    style_axes(
        ax,
        title='Error compared to quadrature size',
        xlabel='Number of quadrature points (N)',
        ylabel='Error',
        legend=True,
        grid={'which': 'both', 'linestyle': ':', 'linewidth': 0.5}
    )

    save_figure('exercise_i_convergence_analysis', fig=fig, dpi=200)
    plt.close(fig)


def main() -> None:
    # Generate individual plots for each quadrature size
    for num_quad in QUADRATURE_SIZES:
        coeffs = legendre_coeffs(num_quad)
        #plot_coeff_decay(coeffs, num_quad)
        #plot_reconstructed_function(coeffs, num_quad)
        plot_summary(coeffs, num_quad)

    # Perform convergence analysis
    print("\n" + "="*50)
    print("CONVERGENCE ANALYSIS")
    print("="*50)

    conv_data = analyze_convergence()
    plot_convergence_analysis(conv_data)

    print(f"\nConvergence analysis plot saved as 'exercise_i_convergence_analysis'")
    print("="*50)


if __name__ == "__main__":
    main()
