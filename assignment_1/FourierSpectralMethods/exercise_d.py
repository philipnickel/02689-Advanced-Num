# %% Imports 
import numpy as np
import matplotlib.pyplot as plt
from numba import njit 

# %% Fourier differentiation matrix D 
# using Algorithm 18 from 'Implementing Spectral Methods for Partial Differential Equations' by David A. Kopriva

@njit
def cot(x):
    return 1 / np.tan(x)


# Using negative Sum trick 
@njit
def fourier_diff_matrix(N):
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = 0
        for j in range(N):
            if i != j:
                D[i, j] = 0.5 * (-1)**(i + j) * cot(np.pi * (i - j) / N)
                D[i, i] -= D[i, j]
    return D




# %% Discrete derivative of v(x) = exp(sin(pi x)) on x in [0,2)

# exact derivative

# %% Convergence rate

def convergence_rate(Ns): 
    errors = np.zeros(len(Ns))
    for i in range(0,len(Ns)):
        N = Ns[i]
        x = np.linspace(0, 2, N, endpoint=False)
        v = np.exp(np.sin(np.pi * x))
        D = np.pi * fourier_diff_matrix(N)
        D_v = D @ v
        dv_exact = np.pi * np.cos(np.pi * x) * v
        errors[i] = np.linalg.norm(D_v - dv_exact, np.inf)

    return errors


Ns = np.linspace(4, 10000, 100, dtype=int)
errors = convergence_rate(Ns)
plt.figure(figsize=(10, 5))
plt.loglog(Ns, errors, 'bo-', label='Error')
# Reference line of O(N**2)
plt.loglog(Ns, errors[0]*(Ns/Ns[0])**-2, 'r--', label='O(N$^{-2}$)')
plt.xlabel('N')
plt.ylabel('Infinity Norm of Error')
plt.title('Convergence Rate of Fourier Spectral Differentiation')
plt.legend()
plt.grid(True)

# plot solutions 
"""
plt.figure(figsize=(10,5))
Ns = [2, 4, 8, 16, 32]
x_fine = np.linspace(0, 2, 400, endpoint=False)
v_exact = np.exp(np.sin(np.pi * x_fine))
dv_exact = np.pi * np.cos(np.pi * x_fine) * v_exact
plt.plot(x_fine, dv_exact, 'k-', label='Exact Derivative', linewidth=2)
for N in Ns:
    x = np.linspace(0, 2, N, endpoint=False)
    v = np.exp(np.sin(np.pi * x))
    D = np.pi * fourier_diff_matrix(N)
    D_v = D @ v
    plt.plot(x, D_v, 'o-', label=f'N={N}')
plt.xlabel('x')
plt.ylabel("v'(x)")
plt.title("Fourier Spectral Differentiation")
plt.legend()
plt.grid(True)
"""
plt.show()
    



#plt.savefig("assignment_1/Plots/FourierSpectralMethods/convRateMatrixD.pdf")


