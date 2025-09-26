# %%
import numpy as np
from numpy.polynomial.legendre import leggauss
from assignment_1.PolynomialMethods.exercise_k import int_matrix

# %%
N = 100
xs, ws = leggauss(N)
u1 = (xs + 1) ** 0
u2 = np.sin((xs + 1))

M = int_matrix(xs)

print(u1 @ M @ u1)
print(u2 @ M @ u2)

# %%
