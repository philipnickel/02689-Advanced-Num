# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi


# Function
def u(x):
    return 1.0 / (2.0 - np.cos(np.pi * (x)))


# Compute Legendre coefficients using Gauss–Legendre quadrature
def legendre_coeffs(N, M=200):
    # Gauss–Legendre nodes and weights
    xj, wj = leggauss(N)
    coeffs = np.zeros(M)

    for n in range(M):
        # Legendre polynomial P_n(x) = Jacobi(n,0,0)
        Pn = eval_jacobi(n, 0, 0, xj)
        integral_approx = np.sum(wj * u(xj + 1) * Pn)
        coeffs[n] = (2 * n + 1) / 2 * integral_approx
    return coeffs, xj


# Experiment with different quadrature sizes
Ns = [200]
coeff_dict = {N: legendre_coeffs(N, M=200)[0] for N in Ns}

# --- Plot decay of coefficients ---
plt.figure(figsize=(10, 6))
for N in Ns:
    plt.semilogy(range(200), np.abs(coeff_dict[N]), label=f"N={N}")
plt.xlabel("n (Polynomial degree)")
plt.ylabel(r"$|c_n|$")
plt.title("Legendre coefficients of $u(x) = 1/(2 - cos(πx))$")
plt.legend()
plt.grid(True, which="both")
plt.show()
# %%


def synthesize(xj: np.ndarray, coeffs: np.ndarray):
    N = len(coeffs)
    result = np.zeros_like(xj)
    for n in range(N):
        Pn = eval_jacobi(n, 0, 0, xj)
        result += coeffs[n] * Pn
    return result


# %%

# --- Plot synthesized function ---
xs = np.linspace(-1, 1, 500)
coeffs = coeff_dict[200]

synth_values = synthesize(xs, coeffs)
synth_values = synthesize(xs, coeffs)

plt.figure(figsize=(10, 6))
plt.plot(xs + 1, synth_values, label="Synthesized function")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Synthesized function from Legendre coefficients")
plt.legend()
plt.grid(True)
plt.show()
# %%
