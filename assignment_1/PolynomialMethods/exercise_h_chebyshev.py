"""Standalone plot of Chebyshev (Jacobi) polynomials for exercise h."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jacobi

from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/PolynomialMethods/exercise_h_chebyshev")


def main() -> None:
    x = np.linspace(-1.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(-1.1, 1.1)
    for n in range(6):
        ax.plot(x, jacobi(n, -0.5, -0.5)(x), label=rf"$n={n}$")
    style_axes(
        ax,
        title=r"Jacobi polynomials (Chebyshev) $P_n^{(-1/2,-1/2)}$",
        xlabel="x",
        ylabel="value",
        legend={"loc": "best"},
        grid={"linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_h_chebyshev_alt", fig=fig, dpi=200)


if __name__ == "__main__":
    main()
