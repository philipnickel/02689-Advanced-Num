# %%
import numpy as np
from numpy.polynomial.legendre import leggauss
from assignment_1.PolynomialMethods.exercise_k import int_matrix

# %%
N = 20
xs, ws = leggauss(N)
u1 = (xs + 1) ** 0
u2 = np.sin((xs + 1))

M = int_matrix(xs)

print(u1 @ M @ u1)
print(u2 @ M @ u2)

# %%

# Convergence test for N -> infinity
Ns = np.arange(1, 20, 1)
error_u1 = []
error_u2 = []

u1_exact = 2
u2_exact = 1 - np.sin(4) / 4

for N in Ns:
    xs, ws = leggauss(N)
    u1 = (xs + 1) ** 0
    u2 = np.sin((xs + 1))
    M = int_matrix(xs)

    error_u1.append(np.abs(u1 @ M @ u1 - u1_exact))
    error_u2.append(np.abs(u2 @ M @ u2 - u2_exact))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].semilogy(Ns, error_u1, ".-", label="Error in u1")
axs[0].set_title("Convergence of u1")
axs[0].set_xlabel("N")
axs[0].set_ylabel("Error")
axs[0].legend()
axs[0].grid()

axs[1].semilogy(Ns, error_u2, ".-", label="Error in u2", color="orange")
axs[1].set_title("Convergence of u2")
axs[1].set_xlabel("N")
axs[1].set_ylabel("Error")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig("./assignment_1/Plots/PolynomialMethods/exercise_l.pdf")

# %%
