# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from numba import njit
import time

from utils.plotting import save_figure, setup_assignment_plotting, style_axes


setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_f")

# ======================================================
# Differentiation methods
# ======================================================

# --- FFT differentiation ---
def fft_diff(v, L):
    N = len(v)
    v_hat = fft(v)
    k = fftfreq(N, d=L/N) * 2*np.pi
    vprime_hat = 1j * k * v_hat
    return np.real(ifft(vprime_hat))

# --- Fourier differentiation matrix ---
@njit
def cot(x):
    return 1.0 / np.tan(x)

@njit
def fourier_diff_matrix(N):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = 0.5 * (-1)**(i + j) * cot(np.pi * (i - j) / N)
        D[i, i] = -np.sum(D[i, :])  # negative sum trick
    return D

def matrix_diff(v, L):
    N = len(v)
    D = (np.pi) * fourier_diff_matrix(N)  # scaling for domain
    return D @ v

# --- Test function and exact derivative ---
def u(x):
    return np.exp(np.sin(np.pi * x))

def u_prime_exact(x):
    return np.pi * np.cos(np.pi * x) * np.exp(np.sin(np.pi * x))

# ======================================================
# Convergence test
# ======================================================
def convergence_test(Ns, L=2):
    err_fft, err_mat = [], []
    for N in Ns:
        x = np.linspace(0, L, N, endpoint=False)
        v = u(x)
        vprime_exact = u_prime_exact(x)

        vprime_fft = fft_diff(v, L)
        vprime_mat = matrix_diff(v, L)

        err_fft.append(np.linalg.norm(vprime_fft - vprime_exact, np.inf))
        err_mat.append(np.linalg.norm(vprime_mat - vprime_exact, np.inf))

    return np.array(err_fft), np.array(err_mat)

# ======================================================
# Performance benchmark
# ======================================================
def benchmark(Ns, L=2):
    t_fft, t_mat = [], []
    for N in Ns:
        x = np.linspace(0, L, N, endpoint=False)
        v = u(x)

        start = time.time()
        fft_diff(v, L)
        t_fft.append(time.time() - start)

        start = time.time()
        matrix_diff(v, L)
        t_mat.append(time.time() - start)

    return np.array(t_fft), np.array(t_mat)

# ======================================================
# Main execution
# ======================================================
if __name__ == "__main__":
    L = 2
    N = 64  # default resolution for solution printout
    x = np.linspace(0, L, N, endpoint=False)
    v = u(x)
    vprime_exact = u_prime_exact(x)

    # --- Compute derivatives ---
    vprime_fft = fft_diff(v, L)
    vprime_mat = matrix_diff(v, L)

    # --- Print solutions ---
    #print("x values:\n", x)
    #print("\nFunction v(x) = exp(sin(pi*x)):\n", v)
    #print("\nExact derivative v'(x):\n", vprime_exact)
    #print("\nFFT derivative:\n", vprime_fft)
    #print("\nMatrix derivative:\n", vprime_mat)

    # --- Plot solution comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, vprime_exact, linewidth=2, label="Exact derivative")
    ax.plot(x, vprime_fft, linestyle='', marker='o', label="FFT derivative")
    ax.plot(x, vprime_mat, linestyle='--', label="Matrix derivative")
    style_axes(
        ax,
        title="Spectral Differentiation: FFT vs Matrix vs Exact",
        xlabel="x",
        ylabel="v'(x)",
        legend=True,
    )
    save_figure("exercise_f_derivative_comparison", fig=fig)

    # --- Convergence study ---
    Ns_conv = [8, 16, 32, 64, 128, 256, 512]
    err_fft, err_mat = convergence_test(Ns_conv, L)

    fig_conv, ax_conv = plt.subplots(figsize=(8, 5))
    ax_conv.loglog(Ns_conv, err_fft, marker='o', label="FFT differentiation")
    ax_conv.loglog(Ns_conv, err_mat, marker='s', label="Matrix differentiation")
    style_axes(
        ax_conv,
        title="Convergence of spectral differentiation",
        xlabel="N (grid points)",
        ylabel="Infinity norm error",
        legend=True,
        grid={'which': 'both'},
    )
    save_figure("exercise_f_convergence", fig=fig_conv)

    # --- Performance study ---
    Ns_perf = [16, 32, 64, 128, 256, 512, 1024]
    t_fft, t_mat = benchmark(Ns_perf, L)

    fig_perf, ax_perf = plt.subplots(figsize=(8, 5))
    ax_perf.loglog(Ns_perf, t_fft, marker='o', label="FFT differentiation")
    ax_perf.loglog(Ns_perf, t_mat, marker='s', label="Matrix differentiation")
    style_axes(
        ax_perf,
        title="Performance: FFT vs Differentiation Matrix",
        xlabel="N (grid points)",
        ylabel="Time [s]",
        legend=True,
        grid={'which': 'both'},
    )
    save_figure("exercise_f_performance", fig=fig_perf)
