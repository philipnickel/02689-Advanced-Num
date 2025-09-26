"""Task e: 

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from assignment_1.FourierSpectralMethods.exercise_d import fourier_diff_matrix
from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_e")

DOMAIN_A, DOMAIN_B = -2.0, 2.0
LENGTH = DOMAIN_B - DOMAIN_A


def w_functions(order: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.zeros((order + 1, x.size))
    w[0] = np.where(x < 0.0, -np.cos(np.pi * x), np.cos(np.pi * x))

    for i in range(1, order + 1):
        mask = x >= 0.0
        if i == 1:
            derivative = np.zeros_like(x)
            derivative[mask] = np.sin(np.pi * x[mask]) / np.pi
            derivative[~mask] = -np.sin(np.pi * x[~mask]) / np.pi
        elif i == 2:
            derivative = np.zeros_like(x)
            derivative[mask] = (1.0 - np.cos(np.pi * x[mask])) / (np.pi**2)
            derivative[~mask] = (np.cos(np.pi * x[~mask]) - 1.0) / (np.pi**2)
        elif i == 3:
            derivative = np.zeros_like(x)
            derivative[mask] = x[mask] / (np.pi**2) - np.sin(np.pi * x[mask]) / (np.pi**3)
            derivative[~mask] = np.sin(np.pi * x[~mask]) / (np.pi**3) - x[~mask] / (np.pi**2)
        else:
            raise ValueError("w_functions defined up to order 3")
        w[i] = derivative
    return w


def fourier_diff_matrix_on_interval(N: int, a: float = DOMAIN_A, b: float = DOMAIN_B) -> np.ndarray:
    scale = 2 * np.pi / (b - a)
    return scale * fourier_diff_matrix(N)


def discrete_l2_norm(values: np.ndarray, h: float) -> float:
    return np.sqrt(h * np.sum(np.abs(values) ** 2))


def main() -> None:
    x_fine = np.linspace(DOMAIN_A, DOMAIN_B, 2000, endpoint=False)
    ladder_fine = w_functions(3, x_fine)

    fig_funcs, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    titles = {
        0: r"$w_0$ (jump at $x=0$)",
        1: r"$w_1$ ($C^0$)",
        2: r"$w_2$ ($C^1$)",
        3: r"$w_3$ ($C^2$)",
    }
    for idx, ax in enumerate(axes.flat):
        ax.plot(x_fine, ladder_fine[idx])
        ax.axvline(0.0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
        style_axes(
            ax,
            title=titles[idx],
            xlabel="x" if idx // 2 else None,
            ylabel="value" if idx % 2 == 0 else None,
            legend=False,
            grid={"linestyle": ":", "linewidth": 0.5},
        )
    save_figure("exercise_e_functions", fig=fig_funcs, dpi=200, tight_layout_kwargs={"rect": [0, 0.03, 1, 0.95]})

    N_values = 2 ** np.arange(4, 11)
    errors = {1: [], 2: [], 3: []}
    for N in N_values:
        x = np.linspace(DOMAIN_A, DOMAIN_B, N, endpoint=False)
        ladder = w_functions(3, x)
        D = fourier_diff_matrix_on_interval(N)
        h = LENGTH / N
        for i in (1, 2, 3):
            derivative_numeric = D @ ladder[i]
            derivative_exact = ladder[i - 1]
            errors[i].append(discrete_l2_norm(derivative_numeric - derivative_exact, h))

    fig_conv, ax = plt.subplots(figsize=(8, 5))
    markers = {1: "o", 2: "s", 3: "^"}
    for i in (1, 2, 3):
        ax.loglog(N_values, errors[i], marker=markers[i], label=rf"$w_{i}$ derivative error")
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
        print(f"Estimated convergence rate for w_{i}: N^{slope:.2f}")
    style_axes(
        ax,
        title="Fourier differentiation errors vs. grid resolution",
        xlabel="Number of modes N",
        ylabel=r"$L_2$ error of $D w_i - w_{i-1}$",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_e_convergence", fig=fig_conv, dpi=200)


if __name__ == "__main__":
    main()
