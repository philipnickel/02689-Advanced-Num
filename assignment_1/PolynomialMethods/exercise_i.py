# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi


# Function
def u(x):
    return 1.0 / (2.0 - np.cos(np.pi * x))


# --- Nodal -> Modal ---
def nodal_to_modal(u_func, N, M):
    """
    Project u(x) onto Legendre basis using Gauss–Legendre quadrature.
    N : number of quadrature nodes
    M : number of Legendre modes (coefficients)
    """
    xj, wj = leggauss(N)          # quadrature nodes and weights
    uj = u_func(xj)               # evaluate function
    coeffs = np.zeros(M)
    for n in range(M):
        Pn = eval_jacobi(n, 0, 0, xj)
        coeffs[n] = (2*n + 1)/2 * np.sum(wj * uj * Pn)
    return coeffs


# --- Modal -> Nodal ---
def modal_to_nodal(x, coeffs):
    """
    Reconstruct function from Legendre coefficients at points x.
    """
    result = np.zeros_like(x)
    for n, cn in enumerate(coeffs):
        Pn = eval_jacobi(n, 0, 0, x)
        result += cn * Pn
    return result


# --- Error computation ---
def compute_errors(u_func, coeffs, xs):
    true_values = u_func(xs)
    synth_values = modal_to_nodal(xs, coeffs)
    L2_error = np.sqrt(np.trapz((synth_values - true_values)**2, xs))
    Linfty_error = np.max(np.abs(synth_values - true_values))
    return L2_error, Linfty_error, true_values, synth_values


# ======================================================
# Main execution
# ======================================================
if __name__ == "__main__":
    M_modes = 200                 # number of Legendre coefficients
    Ns = [10,40,80,100,200]  # quadrature sizes to experiment with

    xs = np.linspace(-1, 1, 500)  # fine grid for error computation
    errors_L2, errors_Linf = [], []

    plt.figure(figsize=(10, 6))
    for N in Ns:
        coeffs = nodal_to_modal(u, N, M_modes)
        L2_error, Linfty_error, true_values, synth_values = compute_errors(u, coeffs, xs)

        errors_L2.append(L2_error)
        errors_Linf.append(Linfty_error)

        # plot synthesized function for each quadrature size
        plt.plot(xs, synth_values, label=f"Synthesized N={N}")

    # --- Plot exact function ---
    plt.plot(xs, true_values, "k--", lw=2, label="Exact function")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Synthesized functions for different quadrature sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot coefficient decay ---
    plt.figure(figsize=(10, 6))
    for N in Ns:
        coeffs = nodal_to_modal(u, N, M_modes)
        plt.semilogy(range(M_modes), np.abs(coeffs), label=f"N={N}")
    plt.xlabel("Mode n")
    plt.ylabel(r"$|c_n|$")
    plt.title("Legendre coefficient decay for different quadrature sizes")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # --- Error vs quadrature size ---
    plt.figure(figsize=(8, 5))
    plt.loglog(Ns, errors_L2, "bo-", label="L2 error")
    plt.loglog(Ns, errors_Linf, "rs-", label="L∞ error")
    plt.xlabel("Quadrature size N")
    plt.ylabel("Error")
    plt.title("Error vs quadrature size")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
# %%
