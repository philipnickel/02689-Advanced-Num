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
reference = 1 / (2 - np.cos(np.pi * x_vals))

N_values = np.arange(4, 40, 4)
errors = []
pointwise_results = {}

for n_terms in N_values:
    approximation = fourier_series_custom(x_vals, c_n, n_terms=int(n_terms))
    pointwise_error = approximation - reference
    pointwise_results[int(n_terms)] = pointwise_error
    max_error = np.max(np.abs(pointwise_error))
    errors.append(max_error)

errors = np.array(errors)

fig_sup, ax_sup = plt.subplots(figsize=(6, 4))
ax_sup.semilogy(N_values, errors, "o-", label="Max error")
# Reference line of O(N**2)
ax_sup.loglog(
    N_values,
    errors[0] * (N_values / N_values[0]) ** -2,
    'k--',
    label=r"$O(N^{-2})$",
    alpha=0.5
)


style_axes(
    ax_sup,
    title="Max error vs N",
    xlabel="N",
    ylabel="Max error",
    legend=True,
)
save_figure("fourier_series_max_error", fig=fig_sup)

fig_pointwise, ax_pointwise = plt.subplots(figsize=(6, 4))
for n_terms in N_values:
    ax_pointwise.semilogy(
        x_vals,
        np.abs(pointwise_results[int(n_terms)]),
        label=f"N={int(n_terms)}",
    )

style_axes(
    ax_pointwise,
    title="Pointwise error vs x",
    xlabel="x",
    ylabel="|Error|",
    legend=True,
)
save_figure("fourier_series_pointwise_error", fig=fig_pointwise)
