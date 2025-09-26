import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_b")
#%% Parameters
N_VALUES = [4, 8, 16, 32, 64]

#%% Functions
def c_analytical(k: int) -> float:
    return 1 / (np.sqrt(3) * (2 + np.sqrt(3)) ** abs(k))


fig, axes = plt.subplots(len(N_VALUES), 1, figsize=(10, 3 * len(N_VALUES)))
if len(N_VALUES) == 1:
    axes = [axes]

errors = []

for ax, N in zip(axes, N_VALUES):
    x = np.linspace(0, 2, N, endpoint=False)
    u = 1 / (2 - np.cos(np.pi * x))

    coeffs_fft = fft(u) / N
    coeffs_shifted = fftshift(coeffs_fft)
    k_vals = np.arange(-N // 2, N // 2)
    coeffs_analytical = np.array([c_analytical(k) for k in k_vals])

    ax.plot(k_vals, np.abs(coeffs_shifted), 'o-', label='FFT', linewidth=1.5)
    ax.plot(k_vals, coeffs_analytical, '--', label='Analytical', linewidth=1.5)
    style_axes(
        ax,
        title=f"Comparison of coefficients (N={N})",
        xlabel="k",
        ylabel="|c_k|",
        legend=True,
    )
    errors.append(np.max(np.abs(np.abs(coeffs_shifted) - coeffs_analytical)))

errors = np.array(errors)
Ns = np.array(N_VALUES, dtype=float)

fig_error, ax_error = plt.subplots(figsize=(8, 5))
ax_error.loglog(Ns, errors, "o-", label=r"$\max(|c_k^{\mathrm{FFT}}| - |c_k^{\mathrm{exact}}|)$")
style_axes(
    ax_error,
    title="Max absolute difference between FFT and analytic coefficients",
    xlabel="N",
    ylabel="max error",
    legend=True,
    grid={"which": "both"},
)
fig.tight_layout()
save_figure("exercise_b_coefficients", fig=fig)
save_figure("exercise_b_coeff_diff", fig=fig_error)
