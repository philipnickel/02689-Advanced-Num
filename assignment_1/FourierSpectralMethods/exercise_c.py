# %%
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_c")


def fourier_nodes(N: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)


def lagrange_basis(N: int, x_eval: np.ndarray) -> np.ndarray:

    x_nodes = fourier_nodes(N)
    dx = x_eval[:, None] - x_nodes[None, :]
    sin_half = np.sin(0.5 * dx)
    mask = np.abs(sin_half) > 1e-14
    cot_half = np.zeros_like(dx)
    cot_half[mask] = np.cos(0.5 * dx[mask]) / sin_half[mask]

    lag_vals = np.zeros_like(dx)
    lag_vals[mask] = (1.0 / N) * np.sin(0.5 * N * dx[mask]) * cot_half[mask]
    coincident = ~mask
    if np.any(coincident):
        lag_vals[coincident] = 0.0
        rows, cols = np.nonzero(coincident)
        for r, c in zip(rows, cols):
            if np.isclose(x_eval[r], x_nodes[c]):
                lag_vals[r, c] = 1.0
    return lag_vals


def fourier_diff_matrix(N: int) -> np.ndarray:

    indices = np.arange(N)
    I = indices[:, None]
    J = indices[None, :]
    diff = I - J
    D = np.zeros((N, N), dtype=float)
    mask = diff != 0
    angles = 0.5 * (I - J)[mask] * (2.0 * np.pi / N)
    cot_values = np.cos(angles) / np.sin(angles)
    parity = (-1.0) ** (I + J)
    D[mask] = 0.5 * parity[mask] * cot_values
    D -= np.diag(np.sum(D, axis=1))
    return D


def main() -> None:
    N = 6
    x_plot = np.linspace(0.0, 2.0 * np.pi, 300, endpoint=False)
    lag_vals = lagrange_basis(N, x_plot)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(N):
        ax.plot(x_plot, lag_vals[:, j], label=rf"$h_{{{j}}}(x)$")
    style_axes(
        ax,
        title=r"Fourier Lagrange polynomials on $[0, 2\pi)$ (N=6)",
        xlabel="x",
        ylabel="value",
        legend=True,
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_c_lagrange", fig=fig)

    N_test = 30
    x_nodes = np.linspace(0.0, 2.0, N_test, endpoint=False)
    v_vals = np.exp(np.sin(np.pi * x_nodes))
    D = np.pi * fourier_diff_matrix(N_test)
    derivative = D @ v_vals

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(x_nodes, v_vals, label=r"$e^{\sin(\pi x)}$")
    ax2.plot(x_nodes, derivative, label=r"$D_N e^{\sin(\pi x)}$")
    style_axes(
        ax2,
        title=r"Differentiation matrix applied to $e^{\sin(\pi x)}$",
        xlabel="x",
        ylabel="value",
        legend=True,
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_c_diff_matrix", fig=fig2)


if __name__ == "__main__":
    main()
