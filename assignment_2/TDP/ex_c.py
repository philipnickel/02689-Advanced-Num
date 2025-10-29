# %%
import numpy as np
import matplotlib.pyplot as plt

# %% Fourier Differentiation Matrix
from assignment_1.FourierSpectralMethods.exercise_d import fourier_diff_matrix

# %% Main Execution
if __name__ == "__main__":
    L = 5
    N = 200
    x = (np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi) * L
    D1 = (1 / L) * fourier_diff_matrix(N)
    D2 = D1 @ D1
    D3 = D2 @ D1

    c = 2
    u0 = 0.5 * c * np.cosh(0.5 * np.sqrt(c) * x) ** (-2)

    # Compute eigenvalues with proper frozen-coefficient KdV operator
    u_max = np.max(np.abs(u0))

    # Construct frozen-coefficient linearized system matrix
    A = -6 * u_max * D1 - D3

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    print(f"The maximum eigenvalue is {max_eig}")
    print(f"Timestep should be under: {1.73 / max_eig}")

    # Plot the steady KdV solution
    plt.figure()
    plt.plot(x, u0)
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
        return -(6 * U * D1 + D3) @ U

    def dealias_mult(u_hat: np.ndarray, v_hat: np.ndarray):
        N = len(u_hat)
        M = int(3 / 2 * N)
        u_hat_pad = np.array([*u_hat[: N // 2], *np.zeros(M - N), *u_hat[N // 2 :]])
        v_hat_pad = np.array([*v_hat[: N // 2], *np.zeros(M - N), *v_hat[N // 2 :]])
        u_pad = np.fft.ifft(u_hat_pad)
        v_pad = np.fft.ifft(v_hat_pad)
        w_pad = u_pad * v_pad
        w_pad_hat = np.fft.fft(w_pad)
        w_hat = np.array([*w_pad_hat[: N // 2], *w_pad_hat[(M - N // 2) : M]])
        return w_hat

    def F_dealias(U: np.ndarray):
        U_hat = np.fft.fft(U)
        dU_hat = np.fft.fft(D1 @ U)
        NL = np.fft.ifft(dealias_mult(U_hat, dU_hat))
        return -6 * NL - D3 @ U

    # %%
    steps = 40000
    u = np.zeros((steps, N), dtype=float)
    u[0] = u0
    delta_t = 0.0002
    for i in range(1, steps):
        U = u[i - 1]
        G = F_dealias(U)
        U = U + (1 / 3) * delta_t * G
        G = -(5 / 9) * G + F_dealias(U)
        U = U + (15 / 16) * delta_t * G
        G = -(153 / 128) * G + F_dealias(U)
        u[i] = U + (8 / 15) * delta_t * G
    plt.matshow(u)

    # %%
    u_last = 0.5 * c * np.cosh(0.5 * np.sqrt(c) * (x - c * steps * delta_t)) ** (-2)

    # plt.plot(x, u[0])
    plt.plot(x, u[-1])
    plt.plot(x, u_last)

# %%
