import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jacobi

x = np.arange(0, 1.0, 0.01)
fig, ax = plt.subplots()
ax.set_ylim(-1.0, 1.0)
ax.set_title(r'Jacobi polynomials (Chebyshev) $P_n^{(-0.5, -0.5)}$')


# Legrendre polynomials
for n in np.arange(0, 6, 1):
    ax.plot(x, jacobi(n, -0.5, -0.5)(x), label=rf'$n={n}$')
plt.legend(loc='best')
plt.show()
