import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi

# Function
def u(x):
    return 1.0 / (2.0 - np.cos(np.pi * x))

# Compute Legendre coefficients using Gauss–Legendre quadrature
def legendre_coeffs(N, M=200):
    # Gauss–Legendre nodes and weights
    xj, wj = leggauss(N)
    coeffs = np.zeros(M+1)
    
    for n in range(M+1):
        # Legendre polynomial P_n(x) = Jacobi(n,0,0)
        Pn = eval_jacobi(n, 0, 0, xj)
        integral_approx = np.sum(wj * u(xj) * Pn)
        coeffs[n] = (2*n + 1)/2 * integral_approx
    return coeffs

# Experiment with different quadrature sizes
Ns = [10,40,80,100,200]
coeff_dict = {N: legendre_coeffs(N, M=200) for N in Ns}

# --- Plot decay of coefficients ---
plt.figure(figsize=(10,6))
for N in Ns:
    plt.semilogy(range(201), np.abs(coeff_dict[N]), label=f"N={N}")
plt.xlabel("n (Polynomial degree)")
plt.ylabel(r"$|c_n|$")
plt.title("Legendre coefficients of $u(x) = 1/(2 - cos(πx))$")
plt.legend()
plt.grid(True, which="both")
plt.show()