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

#%% Analytical KdV Soliton
def kdv_soliton(x, c=1.0, x0=0.0, t=0.0):
    """
    Analytical KdV soliton solution:
        u(x, t) = (1/2)*c * sech^2( (1/2)*sqrt(c)*(x - c*t - x0) )
    """
    arg = 0.5 * np.sqrt(c) * (x - c * t - x0)
    return 0.5 * c * (1 / np.cosh(arg))**2

#%% "Solver" using analytical initial condition
def solve_stationary_kdv(N=256, L=20.0, c=1.0, x0=0.0):
    """
    For this version, we don't numerically solve the steady KdV.
    We use the analytical soliton as the stationary/traveling solution.
    """
    x = np.linspace(-L, L, N, endpoint=False)

    # Compute differentiation matrices
    D1 = fourier_diff_matrix(x)
    D2 = D1 @ D1
    D3 = D2 @ D1

    # Analytical solution (at t=0)
    u = kdv_soliton(x, c=c, x0=x0, t=0.0)

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
    # Parameters for soliton
    c = 1.0    # wave speed
    x0 = 0.0   # initial center position

    # Use analytical soliton as solution
    x, u, D1, D3 = solve_stationary_kdv(c=c, x0=x0)

    # Compute eigenvalues via frozen coefficient method
    eigvals = frozen_coeff_eigenvalues_kdv(u, D1, D3)

    # Plot the analytical soliton
    plt.figure(figsize=(7, 4))
    plt.plot(x, u, label="Analytical Soliton", color="C0")
    plt.title("Analytical Solitary Wave Solution of the KdV Equation")
    plt.xlabel("x")
    plt.ylabel("u(x, 0)")
    plt.grid(True)
    plt.legend()

    # Plot eigenvalues in the complex plane
    plt.figure(figsize=(6, 5))
    plt.scatter(eigvals.real, eigvals.imag, s=12, color="C1")
    plt.title("Eigenvalues (Frozen Coefficient Approximation for KdV)")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.grid(True)
    plt.show()
# %%