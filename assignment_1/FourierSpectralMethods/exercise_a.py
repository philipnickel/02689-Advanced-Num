# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

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

axs[0].set_title(r"$L_\infty$ error as a function of N")
axs[0].set_xlabel(r"N")
axs[0].set_ylabel(r"$L_\infty$")
axs[0].legend()
axs[0].grid()
axs[1].set_title(r"$L_1$ error as a function of x")
axs[1].set_xlabel(r"x")
axs[1].set_ylabel(r"$abs(f - \hat{f})$")
axs[1].grid()

plt.tight_layout()
plt.savefig("assignment_1/Plots/FourierSpectralMethods/exercise_a.pdf")

# %%
