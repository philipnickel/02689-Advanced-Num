#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Fourier Differentiation Matrix
def fourier_diff_matrix(x):
    N = len(x)
    L = x[-1] - x[0] + (x[1] - x[0])
    k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    ik = 1j * k
    D = np.real(np.fft.ifft(np.diag(ik) @ np.fft.fft(np.eye(N)), axis=0))
    return D

#%% Stationary KdV Solver
def solve_stationary_kdv(N=128, L=10.0, max_iter=2000, dt=0.001, tol=1e-10):
    x = np.linspace(-L, L, N, endpoint=False)
    D1 = fourier_diff_matrix(x)
    D2 = D1 @ D1
    D3 = D2 @ D1

    u = np.cosh(x)**-2  # initial guess

    for it in range(max_iter):
        ux = D1 @ u
        uxxx = D3 @ u
        residual = 6 * u * ux + uxxx
        norm = np.linalg.norm(residual)

        if it % 100 == 0:
            print(f"Iteration {it:4d}: residual = {norm:.3e}, max(u) = {u.max():.3f}")
        if norm < tol:
            print(f"Converged at iteration {it} with residual {norm:.3e}")
            break

    return x, u, D1, D3

#%% Frozen Coefficient Eigenvalue Analysis (for KdV)
def frozen_coeff_eigenvalues_kdv(u, D1, D3):
    """
    Apply the method of frozen coefficients for the stationary KdV:
        L_N ≈ 6 * max(|u|) * D1 + D3
    """
    u_max = np.max(np.abs(u))
    print(f"Frozen coefficient (max|u|): {u_max:.4f}")

    # Construct frozen-coefficient linearized system matrix
    A = 6 * u_max * D1 + D3

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A)
    return np.sort_complex(eigvals)

#%% Main Execution
if __name__ == "__main__":
    # Correct unpacking: returns D3, not D2
    x, u, D1, D3 = solve_stationary_kdv()
    
    # Compute eigenvalues with proper frozen-coefficient KdV operator
    eigvals = frozen_coeff_eigenvalues_kdv(u, D1, D3)

    # Plot the steady KdV solution
    plt.figure()
    plt.plot(x, u)
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