# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import time

from utils.plotting import save_figure, setup_assignment_plotting, style_axes
from assignment_1.FourierSpectralMethods.exercise_d import fourier_diff_matrix


setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_f")

# ======================================================
# Differentiation methods
# ======================================================


# --- FFT differentiation ---
def fft_diff(v, L):
    N = len(v)
    v_hat = fft(v)
    k = fftfreq(N, d=L / N) * 2 * np.pi
    vprime_hat = 1j * k * v_hat
    return np.real(ifft(vprime_hat))


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
    # print("x values:\n", x)
    # print("\nFunction v(x) = exp(sin(pi*x)):\n", v)
    # print("\nExact derivative v'(x):\n", vprime_exact)
    # print("\nFFT derivative:\n", vprime_fft)
    # print("\nMatrix derivative:\n", vprime_mat)

    # --- Plot solution comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, vprime_exact, linewidth=1, alpha=0.5, label="Exact derivative")

    ax.plot(x, vprime_fft, linestyle="", marker="x", label="FFT derivative")
    ax.plot(x, vprime_mat, linestyle="", marker="+", label="Matrix derivative")
    style_axes(
        ax,
        title="Spectral Differentiation: FFT vs Matrix vs Exact",
        xlabel="x",
        ylabel="v'(x)",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_f_derivative_comparison", fig=fig)

    # --- Convergence study ---
    # Ns_conv = [8, 16, 32, 64, 128, 256, 512]

    # Ns_conv= [4, 8, 16, 32, 64, 128]
    Ns_conv = np.arange(2, 64 * 70, 10)

    err_fft, err_mat = convergence_test(Ns_conv, L)

    fig_conv, ax_conv = plt.subplots(figsize=(8, 5))
    ax_conv.loglog(Ns_conv, err_fft, marker="o", label="FFT differentiation")
    ax_conv.loglog(
        Ns_conv, err_mat, linestyle="--", marker="s", label="Matrix differentiation"
    )

    # Add reference line for algebraic convergence (for comparison)
    N_ref = np.array(Ns_conv)
    algebraic_ref = 1e-2 * (N_ref[0] / N_ref) ** 2  # O(N^-2) reference
    ax_conv.loglog(N_ref, algebraic_ref, "k--", alpha=0.5, label="$O(N^{-2})$")

    style_axes(
        ax_conv,
        title="Convergence of spectral differentiation",
        xlabel="N (grid points)",
        ylabel="Infinity norm error",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_f_convergence", fig=fig_conv)

    # --- Performance study ---
    # Ns_perf = [16, 32, 64, 128, 256, 512, 1024]
    Ns_perf = np.logspace(2, 11, num=20, base=2, dtype=int)
    # np.arange(16, 512, 10)

    t_fft, t_mat = benchmark(Ns_perf, L)

    fig_perf, ax_perf = plt.subplots(figsize=(8, 5))
    ax_perf.loglog(Ns_perf, t_fft, marker="o", label="FFT differentiation")
    ax_perf.loglog(Ns_perf, t_mat, marker="s", label="Matrix differentiation")

    # Add theoretical complexity reference lines
    N_ref = np.array(Ns_perf)

    # O(N log N) reference for FFT
    fft_ref = t_fft[0] * (N_ref * np.log2(N_ref)) / (Ns_perf[0] * np.log2(Ns_perf[0]))
    ax_perf.loglog(N_ref, fft_ref, "k--", alpha=0.5, label="$O(N \\log N)$")

    # O(N^2) reference for matrix multiplication
    mat_ref = t_mat[0] * (N_ref**2) / (Ns_perf[0] ** 2)
    ax_perf.loglog(N_ref, mat_ref, "k:", alpha=0.5, label="$O(N^2)$")

    style_axes(
        ax_perf,
        title="Performance: FFT vs Differentiation Matrix",
        xlabel="N (grid points)",
        ylabel="Time [s]",
        legend=True,
        grid={"which": "both", "linestyle": ":", "linewidth": 0.5},
    )
    save_figure("exercise_f_performance", fig=fig_perf)

# %%
