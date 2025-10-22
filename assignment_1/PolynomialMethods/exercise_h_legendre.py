import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jacobi

x = np.arange(-1, 1.0, 0.01)
fig, ax = plt.subplots()
ax.set_ylim(-1.0, 1.0)
ax.set_title(r"Jacobi polynomials (Legrendre) $P_n^{(0, 0)}$")


# Legrendre polynomials
for n in np.arange(0, 4, 1):
    ax.plot(x, jacobi(n, 0, 0)(x), label=rf"$n={n}$")
plt.legend(loc="best")
