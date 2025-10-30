# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Fourier Differentiation Matrix
from assignment_1.FourierSpectralMethods.exercise_d import fourier_diff_matrix

# %%

# %% Main Execution
if __name__ == "__main__":
    L = 20
    N = 20000
    x = (np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi) * L
    D1 = (1 / L) * fourier_diff_matrix(N)
    D2 = D1 @ D1
    D3 = D2 @ D1

    c = 0.25
    u1 = lambda x, t: 0.5 * c * np.cosh(0.5 * np.sqrt(c) * (x - c * t)) ** (-2)
    u2 = (
        lambda x, t: 0.5
        * (-c)
        * np.cosh(0.5 * np.sqrt(c) * (x - (-c) * t - 40)) ** (-2)
    )
    u = lambda x, t: u1(x, t) + u2(x, t)

    # Compute eigenvalues with proper frozen-coefficient KdV operator
    u_max = np.max(np.abs(u1(x, 0)))

    # Construct frozen-coefficient linearized system matrix
    A = -6 * u_max * D1 - D3

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    delta_t = 1.73 / max_eig
    print(f"The maximum eigenvalue is {max_eig}")
    print(f"Timestep should be under: {delta_t}")

    # Plot the steady KdV solution
    plt.figure()
    plt.plot(x, u(x, 0))
    plt.title("Steady KdV Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)

    # Plot eigenvalues
    plt.figure()
    plt.scatter(eigvals.real, eigvals.imag, s=10)
    plt.title("Eigenvalues (Frozen Coefficient Approximation for KdV)")
    plt.xlabel("Real(λ)")
    plt.ylabel("Imag(λ)")
    plt.grid(True)
    plt.show()
    # %%

    def F(U: np.ndarray):
        U_hat = np.fft.fft(U)
        k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1] - x[0]))
        dU_hat = 1j * k * U_hat
        d3U_hat = (1j * k) ** 3 * U_hat

        dU = np.fft.ifft(dU_hat)
        d3U = np.fft.ifft(d3U_hat)
        return -6 * U * dU - d3U

    def dealias_mult(u_hat: np.ndarray, v_hat: np.ndarray):
        N = len(u_hat)
        M = int(3 / 2 * N)
        u_hat_pad = np.array([*u_hat[: N // 2], *np.zeros(M - N), *u_hat[N // 2 :]])
        v_hat_pad = np.array([*v_hat[: N // 2], *np.zeros(M - N), *v_hat[N // 2 :]])
        u_pad = np.fft.ifft(u_hat_pad)
        v_pad = np.fft.ifft(v_hat_pad)
        w_pad = u_pad * v_pad
        w_pad_hat = np.fft.fft(w_pad)
        w_hat = 3 / 2 * np.array([*w_pad_hat[: N // 2], *w_pad_hat[(M - N // 2) : M]])
        return w_hat

    def F_dealias(U: np.ndarray):
        U_hat = np.fft.fft(U)
        k = 2 * np.pi * np.fft.fftfreq(N, d=(x[1] - x[0]))
        dU_hat = 1j * k * U_hat
        d3U_hat = (1j * k) ** 3 * U_hat

        NL_hat = dealias_mult(U_hat, dU_hat)
        NL = np.fft.ifft(NL_hat)
        d3U = np.fft.ifft(d3U_hat)
        return -6 * NL - d3U

    # %%
    steps = 1000
    u_sol = np.zeros((steps, N), dtype=float)
    u_sol[0] = u(x, 0)
    delta_t = 0.00001

    for i in range(1, steps):
        U = u_sol[i - 1]
        G = F(U)
        U = U + (1 / 3) * delta_t * G
        G = -(5 / 9) * G + F(U)
        U = U + (15 / 16) * delta_t * G
        G = -(153 / 128) * G + F(U)
        u_sol[i] = U + (8 / 15) * delta_t * G

    # %%
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(x, u_sol[-1])

    axs[1].plot(x, u(x, steps * delta_t))

# %%
