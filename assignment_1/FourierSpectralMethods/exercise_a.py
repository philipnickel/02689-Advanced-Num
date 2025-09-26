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

N_values = np.arange(2, 35, 2)
errors = []
pointwise_results = {}

for n_terms in N_values:
    approximation = fourier_series_custom(x_vals, c_n, n_terms=int(n_terms))
    max_error = np.max(np.abs(approximation - reference))
    errors.append(max_error)

errors = np.array(errors)

fig_sup, ax_sup = plt.subplots(figsize=(6, 4))
ax_sup.semilogy(N_values, errors, "o", label="Max error")

r = 2 - np.sqrt(3)
A = np.sqrt(3) / 3
ref_errors = 2 * A * r ** (N_values + 1) / (1 - r)
ax_sup.semilogy(N_values, ref_errors, "--", label="Analytic bound")
style_axes(
    ax_sup,
    title="Max error vs N",
    xlabel="N",
    ylabel="Max error",
    legend=True,
)
save_figure("fourier_series_max_error", fig=fig_sup)

