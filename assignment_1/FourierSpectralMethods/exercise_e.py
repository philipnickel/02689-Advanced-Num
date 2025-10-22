"""Task e) from Assignment 1: Fourier differentiation accuracy for regularity ladder.

The differentiation matrix follows the nodal formulation in Lecture 1 (Fourier Methods),
where differentiation is effected through a dense matrix with cotangent entries.  We reuse
`fourier_diff_matrix` from exercise d) and rescale it to the physical interval [-2, 2].
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercise_d import fourier_diff_matrix

DOMAIN_A, DOMAIN_B = -2.0, 2.0
LENGTH = DOMAIN_B - DOMAIN_A
PI = np.pi
BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR.parent / "Plots" / "FourierSpectralMethods"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def fourier_diff_matrix_on_interval(N, a=DOMAIN_A, b=DOMAIN_B):
    """Return first-derivative Fourier matrix rescaled to a periodic interval [a, b]."""
    scale = 2 * np.pi / (b - a)
    return scale * fourier_diff_matrix(N)


def w0(x):
    return np.where(x < 0.0, -np.cos(PI * x), np.cos(PI * x))


def w1(x):
    base = np.sin(PI * x) / PI
    return np.where(x < 0.0, -base, base)


def w2(x):
    pos = (1.0 - np.cos(PI * x)) / (PI**2)
    neg = (np.cos(PI * x) - 1.0) / (PI**2)
    return np.where(x < 0.0, neg, pos)


def w3(x):
    pos = (x / (PI**2)) - (np.sin(PI * x) / (PI**3))
    neg = (np.sin(PI * x) / (PI**3)) - (x / (PI**2))
    return np.where(x < 0.0, neg, pos)


W_FUNCTIONS = [w0, w1, w2, w3]


def discrete_l2_norm(values, h):
    """Approximate L2 norm, composite trapezoidal rule"""
    return np.sqrt(h * np.sum(np.abs(values) ** 2))


def compute_errors(N_values):
    errors = {1: [], 2: [], 3: []}
    for N in N_values:
        x = np.linspace(DOMAIN_A, DOMAIN_B, N, endpoint=False)
        D = fourier_diff_matrix_on_interval(N)
        h = LENGTH / N
        for i in (1, 2, 3):
            w_vals = W_FUNCTIONS[i](x)
            derivative_numeric = D @ w_vals
            derivative_exact = W_FUNCTIONS[i - 1](x)
            err = discrete_l2_norm(derivative_numeric - derivative_exact, h)
            errors[i].append(err)
    return errors


def plot_functions():
    x_fine = np.linspace(DOMAIN_A, DOMAIN_B, 2000, endpoint=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    fig.suptitle(r"Functions $w_i(x)$ on $[-2, 2]$")
    labels = [r"$w_0$", r"$w_1$", r"$w_2$", r"$w_3$"]
    for idx, ax in enumerate(axes.flat):
        ax.plot(x_fine, W_FUNCTIONS[idx](x_fine), label=labels[idx])
        ax.axvline(0.0, color="k", linewidth=0.5, linestyle="--")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("value")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(PLOT_DIR / "exercise_e_functions.png", dpi=200)


def plot_convergence(N_values, errors):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {1: "o", 2: "s", 3: "^"}
    for i in (1, 2, 3):
        ax.loglog(
            N_values, errors[i], marker=markers[i], label=rf"$w_{i}$ derivative error"
        )
        tail = min(4, len(N_values))
        slope, _ = np.polyfit(np.log(N_values[-tail:]), np.log(errors[i][-tail:]), 1)
        ax.text(
            N_values[-1],
            errors[i][-1],
            rf"$\mathcal{{O}}(N^{{{slope:.2f}}})$",
            fontsize=10,
            ha="right",
            va="bottom",
        )
    ax.set_xlabel("Number of modes N")
    ax.set_ylabel(r"$L_2$ error of $D w_i - w_{i-1}$")
    ax.set_title("Fourier differentiation errors vs. grid resolution")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exercise_e_convergence.png", dpi=200)


def main():
    N_values = 2 ** np.arange(4, 11)  # 16 .. 1024
    plot_functions()
    errors = compute_errors(N_values)
    plot_convergence(N_values, errors)
    for i in (1, 2, 3):
        tail = min(4, len(N_values))
        slope, _ = np.polyfit(np.log(N_values[-tail:]), np.log(errors[i][-tail:]), 1)
        # As discussed in Lecture 1, each integration step improves regularity by one
        # power of x and the Fourier coefficients decay like k^{-(r+1)} for C^r data,
        # so we expect roughly |error| ~ N^{-(i-0.5)}.  The measured slopes confirm
        # this regularity ladder behaviour.
        print(f"Estimated convergence rate for w_{i}: N^{slope:.2f}")


if __name__ == "__main__":
    main()
