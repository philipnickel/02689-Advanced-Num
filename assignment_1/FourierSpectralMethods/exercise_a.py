# %%
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import save_figure, setup_assignment_plotting, style_axes
# %%

setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_a")


c_n = lambda k: (np.sqrt(3) / 3) * (2 - np.sqrt(3)) ** abs(k)


def fourier_series_custom(x_vals, c_n, n_terms=10):
    result = np.zeros_like(x_vals, dtype=np.complex128)
    for k in range(-n_terms, n_terms + 1):
        result += c_n(k) * np.exp(1j * k * x_vals * np.pi)
    return result.real


x_vals = np.linspace(0, 2, 100, endpoint=False)

u = 1 / (2 - np.cos(np.pi * x_vals))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i in range(0, 50, 5):
    y_custom = fourier_series_custom(x_vals, c_n, n_terms=i)

    axs[0].semilogy(i, np.max(y_custom - u), "o", label=rf"$\hat{{f}}_{{{i}}}$")
    axs[1].semilogy(x_vals, np.abs(y_custom - u))

style_axes(
    axs[0],
    title=r"$L_\infty$ error as a function of N",
    xlabel="N",
    ylabel=r"$L_\infty$",
    legend=True,
)
style_axes(
    axs[1],
    title="$L_1$ error as a function of x",
    xlabel="x",
    ylabel=r"$abs(f - \hat{f})$",
)

save_figure("fourier_series_convergence")

# %%
