# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Parameters
N = 64
x = np.linspace(0, 2, N, endpoint=False)
u = 1 / (2 - np.cos(np.pi * x))

# Numerical FFT
coeffs_fft = fft(u) / N   # normalize to match Fourier series definition
coeffs_shifted = fftshift(coeffs_fft)

# k values (from -N/2 to N/2-1)
k_vals = np.arange(-N//2, N//2)

# Analytical coefficients
def c_analytical(k):
    return 1 / (np.sqrt(3) * (2 + np.sqrt(3))**abs(k))

coeffs_analytical = np.array([c_analytical(k) for k in k_vals])

# Plot comparison
plt.figure(figsize=(10,5))
plt.stem(k_vals, np.abs(coeffs_shifted), basefmt=" ", linefmt="b-", markerfmt="bo", label="Numerical FFT")
plt.stem(k_vals, coeffs_analytical, basefmt=" ", linefmt="r--", markerfmt="rx", label="Analytical")
plt.xlabel("k")
plt.ylabel("|c_k|")
plt.title("Comparison of Fourier Coefficients")
plt.legend()
plt.grid(True)
plt.show()
# %%
