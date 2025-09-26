import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from utils.plotting import save_figure, setup_assignment_plotting, style_axes

setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_b")
#%% Parameters
N = 4
x = np.linspace(0, 2, N, endpoint=False)
u = 1 / (2 - np.cos(np.pi * x))


#%% Functions
def c_analytical(k):
    return 1 / (np.sqrt(3) * (2 + np.sqrt(3))**abs(k))


#%% Numerical FFT
coeffs_fft = fft(u) / N   # normalize to match Fourier series definition
coeffs_shifted = fftshift(coeffs_fft)

# k values (from -N/2 to N/2-1)
k_vals = np.arange(-N//2, N//2)
coeffs_analytical = np.array([c_analytical(k) for k in k_vals])

#%% Plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
ax.stem(k_vals, np.abs(coeffs_shifted), basefmt=" ", linefmt="b-", markerfmt="bo", label="Numerical FFT")
ax.stem(k_vals, coeffs_analytical, basefmt=" ", linefmt="r--", markerfmt="rx", label="Analytical")
style_axes(
    ax,
    title="Comparison of Fourier Coefficients",
    xlabel="k",
    ylabel="|c_k|",
    legend=True,
)
save_figure("exercise_b_coefficients", fig=fig)
